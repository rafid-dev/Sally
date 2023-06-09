#pragma once

#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX2__) || defined(__AVX__)
#define BIT_ALIGNMENT (256)
#endif

#define STRIDE_16_BIT (BIT_ALIGNMENT / 16)
#define ALIGNMENT     (BIT_ALIGNMENT / 8)
#define REG_COUNT (16)

#if defined(__AVX2__) || defined(__AVX__)
using avx_register_type_16 = __m256i;
using avx_register_type_32 = __m256i;
#define avx_madd_epi16 _mm256_madd_epi16
#define avx_load_reg   _mm256_load_si256
#define avx_store_reg  _mm256_store_si256
#define avx_add_epi32  _mm256_add_epi32
#define avx_sub_epi32  _mm256_sub_epi32
#define avx_add_epi16  _mm256_add_epi16
#define avx_sub_epi16  _mm256_sub_epi16
#define avx_max_epi16  _mm256_max_epi16
#define avx_zero _mm256_setzero_si256
#endif


#if defined(__AVX__) || defined(__AVX2__)
inline int32_t sumRegisterEpi32(avx_register_type_32& reg) {
    // first summarize in case of avx512 registers into one 256 bit register
#if defined(__AVX2__) || defined(__AVX__)
    const __m256i reduced_8 = reg;
#endif
    const __m128i reduced_4 =
        _mm_add_epi32(_mm256_castsi256_si128(reduced_8), _mm256_extractf128_si256(reduced_8, 1));

    __m128i vsum = _mm_add_epi32(reduced_4, _mm_srli_si128(reduced_4, 8));
    vsum         = _mm_add_epi32(vsum, _mm_srli_si128(vsum, 4));
    int32_t sums = _mm_cvtsi128_si32(vsum);
    return sums;
}
#endif