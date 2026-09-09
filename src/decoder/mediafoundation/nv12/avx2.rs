//! Eight exact integer NV12 -> BGRA conversions per vector. No float math or
//! change to clipping/rounding. Called only after runtime AVX2 detection.
use super::{convert, Layout};
use std::arch::x86_64::*;

/// # Safety
/// AVX2 must be available. The NV12 layout and source/destination lengths must
/// be validated before calling; each vector reads 8 bytes and writes 32 bytes.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn convert_frame(
    source: &[u8],
    layout: Layout,
    uv_offset: usize,
    coefficients: [i32; 4],
    output: &mut [u8],
) {
    let width = layout.width as usize;
    for y in 0..layout.height as usize {
        let luma = &source[y * layout.stride..][..width];
        let uv = &source[uv_offset + (y / 2) * layout.stride..][..layout.stride];
        let row = &mut output[y * width * 4..][..width * 4];
        let vector_width = width & !7;
        for x in (0..vector_width).step_by(8) {
            // Each load is eight bytes; x + 8 <= visible width <= stride.
            convert_eight(
                luma.as_ptr().add(x),
                uv.as_ptr().add(x),
                coefficients,
                row.as_mut_ptr().add(x * 4),
            );
        }
        for x in vector_width..width {
            row[x * 4..][..4].copy_from_slice(&convert(
                luma[x],
                uv[x & !1],
                uv[(x & !1) + 1],
                coefficients,
            ));
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn convert_eight(y: *const u8, uv: *const u8, [rv, gu, gv, bu]: [i32; 4], output: *mut u8) {
    let luma = _mm256_sub_epi32(
        _mm256_cvtepu8_epi32(_mm_loadl_epi64(y.cast())),
        _mm256_set1_epi32(16),
    );
    let uv = _mm_loadl_epi64(uv.cast());
    let u = _mm256_sub_epi32(
        _mm256_cvtepu8_epi32(_mm_shuffle_epi8(
            uv,
            _mm_setr_epi8(0, 0, 2, 2, 4, 4, 6, 6, -1, -1, -1, -1, -1, -1, -1, -1),
        )),
        _mm256_set1_epi32(128),
    );
    let v = _mm256_sub_epi32(
        _mm256_cvtepu8_epi32(_mm_shuffle_epi8(
            uv,
            _mm_setr_epi8(1, 1, 3, 3, 5, 5, 7, 7, -1, -1, -1, -1, -1, -1, -1, -1),
        )),
        _mm256_set1_epi32(128),
    );
    let y = _mm256_add_epi32(
        _mm256_mullo_epi32(luma, _mm256_set1_epi32(298)),
        _mm256_set1_epi32(128),
    );
    let b = pack(_mm256_srai_epi32::<8>(_mm256_add_epi32(
        y,
        _mm256_mullo_epi32(u, _mm256_set1_epi32(bu)),
    )));
    let g = pack(_mm256_srai_epi32::<8>(_mm256_sub_epi32(
        _mm256_sub_epi32(y, _mm256_mullo_epi32(u, _mm256_set1_epi32(gu))),
        _mm256_mullo_epi32(v, _mm256_set1_epi32(gv)),
    )));
    let r = pack(_mm256_srai_epi32::<8>(_mm256_add_epi32(
        y,
        _mm256_mullo_epi32(v, _mm256_set1_epi32(rv)),
    )));
    let bg = _mm_unpacklo_epi8(b, g);
    let ra = _mm_unpacklo_epi8(r, _mm_set1_epi8(-1));
    _mm_storeu_si128(output.cast(), _mm_unpacklo_epi16(bg, ra));
    _mm_storeu_si128(output.add(16).cast(), _mm_unpackhi_epi16(bg, ra));
}

#[target_feature(enable = "avx2")]
unsafe fn pack(channel: __m256i) -> __m128i {
    // Packing is lane-local. Move [c4..c7] beside [c0..c3] before narrowing
    // again; signed saturation then unsigned saturation exactly implements clamp.
    let words = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(channel, channel));
    let words = _mm256_castsi256_si128(words);
    _mm_packus_epi16(words, words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_rows_match_scalar_with_padding_and_tails() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for width in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let layout = Layout {
                width,
                height: 7,
                stride: (width as usize + 9) & !1,
                coded_height: 12,
                matrix: super::super::Matrix::Bt709,
                mixed_interlace: false,
            };
            let uv_offset = layout.stride * layout.coded_height;
            let source: Vec<u8> = (0..uv_offset * 3 / 2)
                .map(|i| ((i * 73 + i / 7) % 256) as u8)
                .collect();
            let coefficients = [459, 55, 136, 541];
            let mut output = vec![0; width as usize * layout.height as usize * 4];
            unsafe {
                convert_frame(&source, layout, uv_offset, coefficients, &mut output);
            }
            for y in 0..layout.height as usize {
                for x in 0..width as usize {
                    let chroma = uv_offset + (y / 2) * layout.stride + (x & !1);
                    assert_eq!(
                        &output[(y * width as usize + x) * 4..][..4],
                        convert(
                            source[y * layout.stride + x],
                            source[chroma],
                            source[chroma + 1],
                            coefficients
                        )
                    );
                }
            }
        }
    }

    #[test]
    fn vector_conversion_matches_scalar_for_all_yuv_values() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut y = [0; 8];
        let mut uv = [0; 8];
        let mut output = [0; 32];
        for coefficients in [[409, 100, 208, 516], [459, 55, 136, 541]] {
            for u in 0..=255 {
                for v in 0..=255 {
                    for pair in uv.chunks_exact_mut(2) {
                        pair.copy_from_slice(&[u, v]);
                    }
                    for first_y in (0..256).step_by(8) {
                        for (i, y) in y.iter_mut().enumerate() {
                            *y = (first_y + i) as u8;
                        }
                        unsafe {
                            convert_eight(
                                y.as_ptr(),
                                uv.as_ptr(),
                                coefficients,
                                output.as_mut_ptr(),
                            );
                        }
                        for (y, pixel) in y.iter().zip(output.chunks_exact(4)) {
                            assert_eq!(pixel, convert(*y, u, v, coefficients));
                        }
                    }
                }
            }
        }
    }
}
