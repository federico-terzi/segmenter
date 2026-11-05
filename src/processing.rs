//! Shared sequential and pipelined execution, independent of media backends.
//! Factories create media objects on the thread that uses and drops them.
//! Inference and progress always stay on the calling thread, in frame order.
use crate::{
    decoder::VideoDecoder,
    encoder::VideoEncoder,
    engine::Engine,
    frame::{MediaTime, VideoFrame, VideoFrameInfo},
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
    Metadata {
        duration: Option<MediaTime>,
        frame_rate: Option<f64>,
    },
    Frame(VideoFrame),
    Prepare(VideoFrameInfo),
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
    encoder.set_expected_frame_rate(decoder.nominal_frame_rate())?;
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
            let mut work = std::time::Duration::ZERO;
            let start = std::time::Instant::now();
            if decoded_tx
                .send(Message::Metadata {
                    duration: decoder.duration(),
                    frame_rate: decoder.nominal_frame_rate(),
                })
                .is_err()
            {
                return Ok(());
            }
            loop {
                let t = std::time::Instant::now();
                let frame = decoder.read_frame()?;
                work += t.elapsed();
                let Some(frame) = frame else { break };
                if decoded_tx.send(Message::Frame(frame)).is_err() {
                    break;
                }
            }
            if std::env::var_os("SEGMENTER_METAL_PROFILE").is_some() {
                eprintln!("DECODE: work={work:?} total={:?}", start.elapsed());
            }
            Ok(())
        });
        let encode = scope.spawn(move || -> anyhow::Result<u64> {
            let Ok(Message::Metadata {
                duration,
                frame_rate,
            }) = mask_rx.recv()
            else {
                return Ok(0);
            };
            let mut encoder = open_encoder()?;
            encoder.set_expected_duration(duration)?;
            encoder.set_expected_frame_rate(frame_rate)?;
            let mut frames = 0;
            let mut work = std::time::Duration::ZERO;
            let start = std::time::Instant::now();
            while let Ok(message) = mask_rx.recv() {
                let t = std::time::Instant::now();
                match message {
                    Message::Prepare(info) => encoder.prepare(info)?,
                    Message::Frame(frame) => {
                        encoder.send_frame(&frame)?;
                        frames += 1;
                    }
                    Message::Metadata { .. } => {
                        bail!("unexpected metadata after encoder initialization")
                    }
                }
                work += t.elapsed();
            }
            if frames > 0 {
                let t = std::time::Instant::now();
                encoder.finalize()?;
                if std::env::var_os("SEGMENTER_METAL_PROFILE").is_some() {
                    eprintln!(
                        "ENCODE: work={work:?} finalize={:?} total={:?}",
                        t.elapsed(),
                        start.elapsed()
                    );
                }
            }
            Ok(frames)
        });

        let processing = (|| -> anyhow::Result<u64> {
            let Ok(Message::Metadata {
                duration,
                frame_rate,
            }) = decoded_rx.recv()
            else {
                return Ok(0);
            };
            if mask_tx
                .send(Message::Metadata {
                    duration,
                    frame_rate,
                })
                .is_err()
            {
                return Ok(0);
            }
            progress(Progress::Started(duration));
            let mut frames = 0;
            let mut submitted = 0;
            while let Ok(Message::Frame(frame)) = decoded_rx.recv() {
                if submitted == 0 {
                    if let Some(info) = engine.output_frame_info(&frame) {
                        if mask_tx.send(Message::Prepare(info)).is_err() {
                            return Ok(0);
                        }
                    }
                }
                submitted += 1;
                if let Some(mask) = engine.enqueue(&frame)? {
                    let time = mask.time;
                    if mask_tx.send(Message::Frame(mask)).is_err() {
                        return Ok(frames);
                    }
                    frames += 1;
                    progress(Progress::Frame(time, frames));
                }
            }
            while let Some(mask) = engine.drain()? {
                let time = mask.time;
                if mask_tx.send(Message::Frame(mask)).is_err() {
                    return Ok(frames);
                }
                frames += 1;
                progress(Progress::Frame(time, frames));
            }
            if frames != submitted {
                bail!("inference returned {frames} masks for {submitted} input frames");
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

    struct DeferredEngine {
        inner: Segmenter,
        pending: std::collections::VecDeque<VideoFrame>,
        fail_drain: bool,
    }

    #[test]
    fn encoder_preparation_overlaps_inference_and_propagates_errors() {
        use std::sync::Condvar;
        struct PredictableEngine(Arc<(Mutex<bool>, Condvar)>);
        impl Engine for PredictableEngine {
            fn output_frame_info(&self, input: &VideoFrame) -> Option<VideoFrameInfo> {
                Some(input.info())
            }
            fn segment(&mut self, input: &VideoFrame) -> anyhow::Result<VideoFrame> {
                let (lock, ready) = &*self.0;
                let (prepared, timeout) = ready
                    .wait_timeout_while(
                        lock.lock().unwrap(),
                        std::time::Duration::from_secs(2),
                        |p| !*p,
                    )
                    .unwrap();
                assert!(
                    *prepared && !timeout.timed_out(),
                    "encoder must prepare while inference runs"
                );
                Ok(input.clone())
            }
        }
        struct PreparingEncoder {
            ready: Arc<(Mutex<bool>, Condvar)>,
            fail: bool,
            sent: usize,
        }
        impl VideoEncoder for PreparingEncoder {
            fn prepare(&mut self, info: VideoFrameInfo) -> anyhow::Result<()> {
                assert_eq!((info.width, info.height, info.time.value), (2, 2, 0));
                let (lock, ready) = &*self.ready;
                let mut prepared = lock.lock().unwrap();
                assert!(!*prepared, "prepare must run only once");
                *prepared = true;
                ready.notify_all();
                if self.fail {
                    bail!("injected encoder preparation failure");
                }
                Ok(())
            }
            fn send_frame(&mut self, frame: &VideoFrame) -> anyhow::Result<()> {
                assert!(*self.ready.0.lock().unwrap());
                assert_eq!(frame.time.value, self.sent as i64);
                self.sent += 1;
                Ok(())
            }
            fn finalize(&mut self) -> anyhow::Result<()> {
                assert_eq!(self.sent, 17);
                Ok(())
            }
        }
        for fail in [false, true] {
            let ready = Arc::new((Mutex::new(false), Condvar::new()));
            let encoder_ready = ready.clone();
            let mut engine = PredictableEngine(ready);
            let result = run(
                Mode::Pipelined,
                || {
                    Ok(Box::new(Decoder {
                        next: 0,
                        count: 17,
                        fail_at: None,
                    }))
                },
                move || {
                    Ok(Box::new(PreparingEncoder {
                        ready: encoder_ready,
                        fail,
                        sent: 0,
                    }))
                },
                &mut engine,
                |_| {},
            );
            if fail {
                assert!(format!("{:#}", result.unwrap_err())
                    .contains("injected encoder preparation failure"));
            } else {
                assert_eq!(result.unwrap(), 17);
            }
        }
    }

    impl Engine for DeferredEngine {
        fn segment(&mut self, frame: &VideoFrame) -> anyhow::Result<VideoFrame> {
            self.inner.segment(frame)
        }

        fn enqueue(&mut self, frame: &VideoFrame) -> anyhow::Result<Option<VideoFrame>> {
            self.pending.push_back(self.inner.segment(frame)?);
            Ok(if self.pending.len() == 3 {
                self.pending.pop_front()
            } else {
                None
            })
        }

        fn drain(&mut self) -> anyhow::Result<Option<VideoFrame>> {
            if self.fail_drain {
                bail!("injected drain error");
            }
            Ok(self.pending.pop_front())
        }
    }

    #[test]
    fn queued_inference_drains_short_clips_and_reports_output_timestamps() {
        for count in [1, 2, 3, 4, 17] {
            let frames = Arc::new(Mutex::new(Vec::new()));
            let captured = frames.clone();
            let mut progress_times = Vec::new();
            let mut engine = DeferredEngine {
                inner: Segmenter {
                    next: 0,
                    fail_at: None,
                },
                pending: Default::default(),
                fail_drain: false,
            };
            let result = run(
                Mode::Pipelined,
                || {
                    Ok(Box::new(Decoder {
                        next: 0,
                        count,
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
                |event| {
                    if let Progress::Frame(time, _) = event {
                        progress_times.push(time.value)
                    }
                },
            )
            .unwrap();
            assert_eq!(result, count as u64);
            assert_eq!(progress_times, (0..count).collect::<Vec<_>>());
            assert_eq!(
                *frames.lock().unwrap(),
                (0..count).chain([-1]).collect::<Vec<_>>()
            );
            assert!(engine.pending.is_empty());
        }
    }

    #[test]
    fn queued_inference_errors_disconnect_workers_with_pending_frames() {
        for (decode, infer, encode, drain, expected) in [
            (Some(1), None, None, false, "injected decoder error"),
            (None, Some(1), None, false, "injected inference error"),
            (None, None, Some(1), false, "injected encoder error"),
            (None, None, None, true, "injected drain error"),
        ] {
            let mut engine = DeferredEngine {
                inner: Segmenter {
                    next: 0,
                    fail_at: infer,
                },
                pending: Default::default(),
                fail_drain: drain,
            };
            let error = run(
                Mode::Pipelined,
                || {
                    Ok(Box::new(Decoder {
                        next: 0,
                        count: 17,
                        fail_at: decode,
                    }))
                },
                || {
                    Ok(Box::new(Encoder {
                        frames: Arc::default(),
                        fail_at: encode,
                        fail_finalize: false,
                    }))
                },
                &mut engine,
                |_| {},
            )
            .unwrap_err();
            assert!(format!("{error:#}").contains(expected), "{error:#}");
        }
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
        fn nominal_frame_rate(&self) -> Option<f64> {
            self.check_thread();
            Some(30000.0 / 1001.0)
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
        fn set_expected_frame_rate(&mut self, rate: Option<f64>) -> anyhow::Result<()> {
            self.check_thread();
            assert_eq!(self.frames, 0);
            assert_eq!(rate, Some(30000.0 / 1001.0));
            Ok(())
        }
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
