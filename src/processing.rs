//! Shared sequential and pipelined execution, independent of media backends.
//! Factories create media objects on the thread that uses and drops them.
//! Inference and progress always stay on the calling thread, in frame order.
use crate::{
    decoder::VideoDecoder,
    encoder::VideoEncoder,
    engine::Engine,
    frame::{MediaTime, VideoFrame},
};
use anyhow::{bail, Context};
use std::{sync::mpsc::sync_channel, thread};

const FRAME_QUEUE_CAPACITY: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Sequential,
    Pipelined,
}

enum Message {
    Duration(Option<MediaTime>),
    Frame(VideoFrame),
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Progress {
    Started(Option<MediaTime>),
    Frame(MediaTime, u64),
}

/// Factories construct the decoder and encoder on their owning threads.
/// Neither media object nor the engine needs to implement Send.
pub(crate) fn run(
    mode: Mode,
    open_decoder: impl FnOnce() -> anyhow::Result<Box<dyn VideoDecoder>> + Send,
    open_encoder: impl FnOnce() -> anyhow::Result<Box<dyn VideoEncoder>> + Send,
    engine: &mut dyn Engine,
    progress: impl FnMut(Progress),
) -> anyhow::Result<u64> {
    match mode {
        Mode::Sequential => run_sequential(open_decoder()?, open_encoder()?, engine, progress),
        Mode::Pipelined => run_pipelined(open_decoder, open_encoder, engine, progress),
    }
}

fn run_sequential(
    mut decoder: Box<dyn VideoDecoder>,
    mut encoder: Box<dyn VideoEncoder>,
    engine: &mut dyn Engine,
    mut progress: impl FnMut(Progress),
) -> anyhow::Result<u64> {
    let duration = decoder.duration();
    encoder.set_expected_duration(duration)?;
    progress(Progress::Started(duration));
    let mut frames = 0;
    while let Some(frame) = decoder.read_frame()? {
        let mask = engine.segment(&frame)?;
        encoder.send_frame(&mask)?;
        frames += 1;
        progress(Progress::Frame(frame.time, frames));
    }
    if frames == 0 {
        bail!("input video did not produce any frames");
    }
    encoder.finalize()?;
    Ok(frames)
}

fn run_pipelined(
    open_decoder: impl FnOnce() -> anyhow::Result<Box<dyn VideoDecoder>> + Send,
    open_encoder: impl FnOnce() -> anyhow::Result<Box<dyn VideoEncoder>> + Send,
    engine: &mut dyn Engine,
    mut progress: impl FnMut(Progress),
) -> anyhow::Result<u64> {
    thread::scope(|scope| {
        let (decoded_tx, decoded_rx) = sync_channel(FRAME_QUEUE_CAPACITY);
        let (mask_tx, mask_rx) = sync_channel(FRAME_QUEUE_CAPACITY);
        let decode = scope.spawn(move || -> anyhow::Result<()> {
            let mut decoder = open_decoder()?;
            if decoded_tx
                .send(Message::Duration(decoder.duration()))
                .is_err()
            {
                return Ok(());
            }
            while let Some(frame) = decoder.read_frame()? {
                if decoded_tx.send(Message::Frame(frame)).is_err() {
                    break;
                }
            }
            Ok(())
        });
        let encode = scope.spawn(move || -> anyhow::Result<u64> {
            let Ok(Message::Duration(duration)) = mask_rx.recv() else {
                return Ok(0);
            };
            let mut encoder = open_encoder()?;
            encoder.set_expected_duration(duration)?;
            let mut frames = 0;
            while let Ok(Message::Frame(frame)) = mask_rx.recv() {
                encoder.send_frame(&frame)?;
                frames += 1;
            }
            if frames > 0 {
                encoder.finalize()?;
            }
            Ok(frames)
        });

        let processing = (|| -> anyhow::Result<u64> {
            let Ok(Message::Duration(duration)) = decoded_rx.recv() else {
                return Ok(0);
            };
            if mask_tx.send(Message::Duration(duration)).is_err() {
                return Ok(0);
            }
            progress(Progress::Started(duration));
            let mut frames = 0;
            while let Ok(Message::Frame(frame)) = decoded_rx.recv() {
                let mask = engine.segment(&frame)?;
                if mask_tx.send(Message::Frame(mask)).is_err() {
                    break;
                }
                frames += 1;
                progress(Progress::Frame(frame.time, frames));
            }
            Ok(frames)
        })();
        // Disconnect before joining on ALL exit paths, including an inference
        // error. Otherwise a producer blocked on a full queue could deadlock.
        drop(decoded_rx);
        drop(mask_tx);
        let decoded = decode
            .join()
            .map_err(|_| anyhow::anyhow!("video decoder worker panicked"));
        let encoded = encode
            .join()
            .map_err(|_| anyhow::anyhow!("video encoder worker panicked"));
        let frames = processing?;
        decoded?.context("video decoder worker failed")?;
        let encoded = encoded?.context("video encoder worker failed")?;
        if frames == 0 {
            bail!("input video did not produce any frames");
        }
        if encoded != frames {
            bail!("encoder wrote {encoded} frames, expected {frames}");
        }
        Ok(frames)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        cell::RefCell,
        rc::Rc,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };

    struct Decoder {
        next: i64,
        count: i64,
        fail_at: Option<i64>,
    }
    impl VideoDecoder for Decoder {
        fn read_frame(&mut self) -> anyhow::Result<Option<VideoFrame>> {
            if self.fail_at == Some(self.next) {
                bail!("injected decoder error");
            }
            if self.next == self.count {
                return Ok(None);
            }
            let frame = VideoFrame::new_bgra(
                2,
                2,
                8,
                MediaTime::new(self.next, 30)?,
                vec![self.next as u8; 16],
            )?;
            self.next += 1;
            Ok(Some(frame))
        }
    }
    struct Segmenter {
        next: i64,
        fail_at: Option<i64>,
    }
    impl Engine for Segmenter {
        fn segment(&mut self, frame: &VideoFrame) -> anyhow::Result<VideoFrame> {
            assert_eq!(
                frame.time.value, self.next,
                "inference must retain temporal order"
            );
            if self.fail_at == Some(self.next) {
                bail!("injected inference error");
            }
            self.next += 1;
            Ok(frame.clone())
        }
    }
    struct Encoder {
        frames: Arc<Mutex<Vec<i64>>>,
        fail_at: Option<i64>,
        fail_finalize: bool,
    }
    impl VideoEncoder for Encoder {
        fn send_frame(&mut self, frame: &VideoFrame) -> anyhow::Result<()> {
            if self.fail_at == Some(frame.time.value) {
                bail!("injected encoder error");
            }
            self.frames.lock().unwrap().push(frame.time.value);
            Ok(())
        }
        fn finalize(&mut self) -> anyhow::Result<()> {
            if self.fail_finalize {
                bail!("injected finalize error");
            }
            self.frames.lock().unwrap().push(-1);
            Ok(())
        }
    }

    #[test]
    fn both_modes_keep_all_frames_in_order_and_finalize() {
        for mode in [Mode::Sequential, Mode::Pipelined] {
            let frames = Arc::new(Mutex::new(Vec::new()));
            let captured = frames.clone();
            let mut engine = Segmenter {
                next: 0,
                fail_at: None,
            };
            let count = run(
                mode,
                || {
                    Ok(Box::new(Decoder {
                        next: 0,
                        count: 100,
                        fail_at: None,
                    }))
                },
                move || {
                    Ok(Box::new(Encoder {
                        frames: captured,
                        fail_at: None,
                        fail_finalize: false,
                    }))
                },
                &mut engine,
                |_| {},
            )
            .unwrap();
            assert_eq!(count, 100);
            assert_eq!(
                *frames.lock().unwrap(),
                (0..100).chain([-1]).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn worker_errors_and_inference_errors_disconnect_queues_and_join() {
        for mode in [Mode::Sequential, Mode::Pipelined] {
            for (decode_failure, inference_failure, encode_failure, finalize_failure, expected) in [
                (Some(10), None, None, false, "injected decoder error"),
                (None, Some(10), None, false, "injected inference error"),
                (None, None, Some(10), false, "injected encoder error"),
                (None, None, None, true, "injected finalize error"),
            ] {
                let mut engine = Segmenter {
                    next: 0,
                    fail_at: inference_failure,
                };
                let error = run(
                    mode,
                    move || {
                        Ok(Box::new(Decoder {
                            next: 0,
                            count: 100,
                            fail_at: decode_failure,
                        }))
                    },
                    move || {
                        Ok(Box::new(Encoder {
                            frames: Arc::default(),
                            fail_at: encode_failure,
                            fail_finalize: finalize_failure,
                        }))
                    },
                    &mut engine,
                    |_| {},
                )
                .unwrap_err();
                assert!(format!("{error:#}").contains(expected), "{error:#}");
            }
        }
    }

    #[test]
    fn opening_failures_and_empty_input_terminate_cleanly() {
        for mode in [Mode::Sequential, Mode::Pipelined] {
            let mut engine = Segmenter {
                next: 0,
                fail_at: None,
            };
            let error = run(
                mode,
                || bail!("injected open error"),
                || unreachable!(),
                &mut engine,
                |_| {},
            )
            .unwrap_err();
            assert!(format!("{error:#}").contains("injected open error"));
            let error = run(
                mode,
                || {
                    Ok(Box::new(Decoder {
                        next: 0,
                        count: 100,
                        fail_at: None,
                    }))
                },
                || bail!("injected encoder open error"),
                &mut engine,
                |_| {},
            )
            .unwrap_err();
            assert!(format!("{error:#}").contains("injected encoder open error"));
            let error = run(
                mode,
                || {
                    Ok(Box::new(Decoder {
                        next: 0,
                        count: 0,
                        fail_at: None,
                    }))
                },
                || {
                    Ok(Box::new(Encoder {
                        frames: Arc::default(),
                        fail_at: None,
                        fail_finalize: false,
                    }))
                },
                &mut engine,
                |_| {},
            )
            .unwrap_err();
            assert!(error.to_string().contains("did not produce any frames"));
        }
    }

    #[test]
    fn encoder_setup_failure_disconnects_a_busy_decoder() {
        struct FailingEncoder;
        impl VideoEncoder for FailingEncoder {
            fn set_expected_duration(&mut self, _: Option<MediaTime>) -> anyhow::Result<()> {
                bail!("injected duration error");
            }
            fn send_frame(&mut self, _: &VideoFrame) -> anyhow::Result<()> {
                unreachable!()
            }
            fn finalize(&mut self) -> anyhow::Result<()> {
                unreachable!()
            }
        }
        for mode in [Mode::Sequential, Mode::Pipelined] {
            let error = run(
                mode,
                || {
                    Ok(Box::new(Decoder {
                        next: 0,
                        count: 100,
                        fail_at: None,
                    }))
                },
                || Ok(Box::new(FailingEncoder)),
                &mut Segmenter {
                    next: 0,
                    fail_at: None,
                },
                |_| {},
            )
            .unwrap_err();
            assert!(format!("{error:#}").contains("injected duration error"));
        }
    }

    // Rc deliberately makes each stage !Send. Factories must construct it on
    // its owning thread; only frames may move between stages.
    struct ThreadBoundStage {
        owner: thread::ThreadId,
        dropped: Arc<AtomicUsize>,
        _not_send: Rc<()>,
        frames: u64,
    }
    impl ThreadBoundStage {
        fn new(dropped: Arc<AtomicUsize>) -> Self {
            Self {
                owner: thread::current().id(),
                dropped,
                _not_send: Rc::new(()),
                frames: 0,
            }
        }
        fn check_thread(&self) {
            assert_eq!(self.owner, thread::current().id());
        }
    }
    impl Drop for ThreadBoundStage {
        fn drop(&mut self) {
            self.check_thread();
            self.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl VideoDecoder for ThreadBoundStage {
        fn duration(&self) -> Option<MediaTime> {
            self.check_thread();
            Some(MediaTime::new(2, 30).unwrap())
        }
        fn read_frame(&mut self) -> anyhow::Result<Option<VideoFrame>> {
            self.check_thread();
            if self.frames == 2 {
                return Ok(None);
            }
            let frame = VideoFrame::new_bgra(
                2,
                2,
                8,
                MediaTime::new(self.frames as i64, 30)?,
                vec![0; 16],
            )?;
            self.frames += 1;
            Ok(Some(frame))
        }
    }
    impl VideoEncoder for ThreadBoundStage {
        fn set_expected_duration(&mut self, duration: Option<MediaTime>) -> anyhow::Result<()> {
            self.check_thread();
            assert_eq!(duration, Some(MediaTime::new(2, 30)?));
            Ok(())
        }
        fn send_frame(&mut self, _: &VideoFrame) -> anyhow::Result<()> {
            self.check_thread();
            self.frames += 1;
            Ok(())
        }
        fn finalize(&mut self) -> anyhow::Result<()> {
            self.check_thread();
            assert_eq!(self.frames, 2);
            Ok(())
        }
    }
    impl Engine for ThreadBoundStage {
        fn segment(&mut self, frame: &VideoFrame) -> anyhow::Result<VideoFrame> {
            self.check_thread();
            Ok(frame.clone())
        }
    }

    #[test]
    fn both_modes_preserve_thread_affinity_progress_and_lifetimes() {
        for mode in [Mode::Sequential, Mode::Pipelined] {
            let caller = thread::current().id();
            let dropped = Arc::new(AtomicUsize::new(0));
            let decoder_drops = dropped.clone();
            let encoder_drops = dropped.clone();
            let mut engine = ThreadBoundStage::new(dropped.clone());
            let events = Rc::new(RefCell::new(Vec::new()));
            let count = run(
                mode,
                move || {
                    assert_eq!(thread::current().id() == caller, mode == Mode::Sequential);
                    Ok(Box::new(ThreadBoundStage::new(decoder_drops)))
                },
                move || {
                    assert_eq!(thread::current().id() == caller, mode == Mode::Sequential);
                    Ok(Box::new(ThreadBoundStage::new(encoder_drops)))
                },
                &mut engine,
                |event| {
                    assert_eq!(thread::current().id(), caller);
                    events.borrow_mut().push(event);
                },
            )
            .unwrap();
            assert_eq!(count, 2);
            assert_eq!(
                *events.borrow(),
                vec![
                    Progress::Started(Some(MediaTime::new(2, 30).unwrap())),
                    Progress::Frame(MediaTime::new(0, 30).unwrap(), 1),
                    Progress::Frame(MediaTime::new(1, 30).unwrap(), 2),
                ]
            );
            assert_eq!(dropped.load(Ordering::SeqCst), 2);
            drop(engine);
            assert_eq!(dropped.load(Ordering::SeqCst), 3);
        }
    }
}
