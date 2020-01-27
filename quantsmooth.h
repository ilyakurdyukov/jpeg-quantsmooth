/*
 * Copyright (C) 2016-2020 Kurdyukov Ilya
 *
 * This file is part of jpeg quantsmooth
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifdef NO_MATH_LIB
#define roundf(x) (float)((int)((x) < 0 ? (x) - 0.5f : (x) + 0.5f))
#endif

#ifdef WITH_LOG
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
// conflict with libjpeg typedef
#define INT32 INT32_WIN
#include <windows.h>
static inline int64_t get_time_usec() {
    LARGE_INTEGER freq, perf;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&perf);
    double d = (double)perf.QuadPart * 1000000.0 / (double)freq.QuadPart;
    return d;
}
#else
#include <time.h>
#include <sys/time.h>
static inline int64_t get_time_usec() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec * (int64_t)1000000 + time.tv_usec;
}
#endif
#endif

#if defined(__SSE4_1__)
#define USE_SSE2
#define USE_SSE4
#include <smmintrin.h>
#elif defined(__SSE2__)
#define USE_SSE2
#include <emmintrin.h>

#define _mm_cvtepu8_epi16(a) _mm_unpacklo_epi8(a, _mm_setzero_si128())
// _mm_cmplt_epi16(a, _mm_setzero_si128()) or _mm_srai_epi16(a, 15)
#define _mm_cvtepi16_epi32(a) _mm_unpacklo_epi16(a, _mm_srai_epi16(a, 15))
#define _mm_abs_epi16(a) ({ \
    __m128i __tmp = _mm_srai_epi16(a, 15); \
    _mm_xor_si128(_mm_add_epi16(a, __tmp), __tmp); })
#endif

#ifdef __AVX2__
#define USE_AVX2
#include <immintrin.h>
#endif

#if defined(__ARM_NEON__) || defined(__aarch64__)
#define USE_NEON
#include <arm_neon.h>
// for testing on x86
#elif 0 && defined(__SSSE3__)
#define USE_NEON
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsequence-point"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#include "NEON_2_SSE.h"
#pragma GCC diagnostic pop
#warning NEON test build on x86
#elif defined(__arm__)
#warning compiling for ARM without NEON support
#endif

#define ALIGN(n) __attribute__((aligned(n)))

#include "idct.h"

static float ALIGN(32) quantsmooth_tables[DCTSIZE2][3*DCTSIZE2 + DCTSIZE];

static void quantsmooth_init(float border) {
    int i; JCOEF coef[DCTSIZE2];

    range_limit_init();

    for (i = 0; i < DCTSIZE2; i++) {
        float *tab = quantsmooth_tables[i];
        float bcoef = border;
        int x, y, p0, p1;
        memset(coef, 0, sizeof(coef)); coef[i] = 1;
        idct_fslow(coef, tab);

#define M1(xx, yy, j) \
    p0 = y*DCTSIZE+x; p1 = (yy)*DCTSIZE+(xx); \
    tab[j*DCTSIZE2+p0] = tab[p0] - tab[p1];
        for (y = 0; y < DCTSIZE; y++) {
            for (x = 0; x < DCTSIZE-1; x++) { M1(x+1,y,1) }
            tab[DCTSIZE2+y*DCTSIZE+x] = tab[y*DCTSIZE+x] * bcoef;
        }
        for (y = 0; y < DCTSIZE-1; y++)
        for (x = 0; x < DCTSIZE; x++) { M1(x,y+1,2) }
#undef M1
        for (x = 0; x < DCTSIZE2; x++) tab[x] *= bcoef;
        for (y = 0; y < DCTSIZE; y++) tab[3*DCTSIZE2+y] = tab[y*DCTSIZE];
    }
}

// When compiling with libjpeg-turbo and static linking, you can use
// optimized idct_islow from library. Which will be faster if you don't have
// processor with AVX2 support, but SSE2 is supported.

#if defined(USE_JSIMD) && defined(LIBJPEG_TURBO_VERSION)
EXTERN(void) jsimd_idct_islow_sse2(void *dct_table, JCOEFPTR coef_block, JSAMPARRAY output_buf, JDIMENSION output_col);
#define idct_islow(coef, buf, st) jsimd_idct_islow_sse2(dct_table1, coef, output_buf, output_col)
#define M1 1,1,1,1, 1,1,1,1
static short dct_table1[DCTSIZE2] = { M1, M1, M1, M1, M1, M1, M1, M1 };
#undef M1
#endif

static const char zigzag_refresh[DCTSIZE2] = {
    0, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 1,
};

static void quantsmooth_block(JCOEFPTR coef, UINT16 *quantval, JSAMPROW image, int stride, float gain) {

    int k, x, y, flag = 1;
    JSAMPLE ALIGN(32) buf[DCTSIZE2+DCTSIZE*2];
#ifdef USE_JSIMD
    JSAMPROW output_buf[8] = { buf+8*0, buf+8*1, buf+8*2, buf+8*3, buf+8*4, buf+8*5, buf+8*6, buf+8*7 };
    int output_col = 0;
#endif

    (void)x;
    for (y = 0; y < DCTSIZE; y++) {
        buf[DCTSIZE2+y] = image[y*stride-1];
        buf[DCTSIZE2+y+DCTSIZE] = image[y*stride+8];
    }

    for (k = DCTSIZE2-1; k > 0; k--) {
        float *tab, a2, a3;
        int i = jpeg_natural_order[k];
        int range = (int)((float)quantval[i] * gain + 0.5f);

        tab = quantsmooth_tables[i];
        a2 = a3 = 0;
#if 1
        if (flag != 0 && zigzag_refresh[k]) {
            idct_islow(coef, buf, DCTSIZE);
            flag = 0;
        }
#endif

#if 1 && defined(USE_NEON)
#define VINIT \
    int16x8_t v0, v5; uint16x8_t v6 = vdupq_n_u16(range); \
    float32x4_t f0, f1, f2, f3; uint8x8_t i0, i1; \
    float32x4_t vsum1 = vdupq_n_f32(0), vsum2 = vsum1;

#define VCORE(tab) \
    v0 = vreinterpretq_s16_u16(vsubl_u8(i0, i1)); \
    v5 = vreinterpretq_s16_u16(vqsubq_u16(v6, \
            vreinterpretq_u16_s16(vabsq_s16(v0)))); \
    VCORE1(low, f0, f1, tab) VCORE1(high, f2, f3, tab+4) \
    vsum1 = vaddq_f32(vsum1, vaddq_f32(f0, f2)); \
    vsum2 = vaddq_f32(vsum2, vaddq_f32(f1, f3));

#define VCORE1(low, f0, f1, tab) \
    f0 = vcvtq_f32_s32(vmovl_s16(vget_##low##_s16(v5))); \
    f0 = vmulq_f32(f0, f0); \
    f1 = vmulq_f32(f0, vld1q_f32(tab)); \
    f0 = vmulq_f32(f0, vcvtq_f32_s32(vmovl_s16(vget_##low##_s16(v0)))); \
    f0 = vmulq_f32(f0, f1); f1 = vmulq_f32(f1, f1);

#define VFIN { \
    float32x4x2_t p0 = vzipq_f32(vsum1, vsum2); \
    float32x2_t v0; \
    f0 = vaddq_f32(p0.val[0], p0.val[1]); \
    v0 = vadd_f32(vget_low_f32(f0), vget_high_f32(f0)); \
    a2 = vget_lane_f32(v0, 0); \
    a3 = vget_lane_f32(v0, 1); \
}

#define VLDPIX(j, p) i##j = vld1_u8(p);
#define VRIGHT(p) i1 = vext_u8(i0, vld1_dup_u8(p), 1);
#define VCOPY1 i0 = i1;

#elif 1 && defined(USE_AVX2)
#define VINIT \
    __m128i v0, v1, v5, v6 = _mm_set1_epi16(range); \
    __m256 f0, f1, f4, vsum = _mm256_setzero_ps();

#define VCORE(tab) \
    v0 = _mm_sub_epi16(v0, v1); \
    f4 = _mm256_load_ps(tab); \
    v5 = _mm_subs_epu16(v6, _mm_abs_epi16(v0)); \
    f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v5)); \
    f0 = _mm256_mul_ps(f0, f0); \
    f1 = _mm256_mul_ps(f0, f4); \
    f0 = _mm256_mul_ps(f0, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v0))); \
    f0 = _mm256_mul_ps(f0, f1); f1 = _mm256_mul_ps(f1, f1); \
    f0 = _mm256_add_ps(_mm256_permute2f128_ps(f0, f1, 0x20), _mm256_permute2f128_ps(f0, f1, 0x31)); \
    vsum = _mm256_add_ps(vsum, f0);

#define VFIN \
    f0 = vsum; \
    f0 = _mm256_add_ps(f0, _mm256_shuffle_ps(f0, f0, 0xee)); \
    f0 = _mm256_add_ps(f0, _mm256_shuffle_ps(f0, f0, 0x55)); \
    a2 = _mm256_cvtss_f32(f0); \
    a3 = _mm_cvtss_f32(_mm256_extractf128_ps(f0, 1));

#define VLDPIX(i, p) v##i = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)(p)));
#define VRIGHT(p) v1 = _mm_insert_epi16(_mm_bsrli_si128(v0, 2), *(JSAMPLE*)(p), 7);
#define VCOPY1 v0 = v1;

#elif 1 && defined(USE_SSE2)
#define VINIT \
    __m128i v0, v1, v2, v5, v6 = _mm_set1_epi16(range), v7 = _mm_setzero_si128(); \
    __m128 f0, f1, f2, f3; \
    __m128 vsum1 = _mm_setzero_ps(), vsum2 = vsum1;

#define VCORE(tab) \
    v0 = _mm_sub_epi16(v0, v1); \
    v2 = _mm_srai_epi16(v0, 15); \
    v5 = _mm_subs_epu16(v6, _mm_abs_epi16(v0)); \
    VCORE1(lo, f0, f1, tab) VCORE1(hi, f2, f3, tab+4) \
    vsum1 = _mm_add_ps(vsum1, _mm_add_ps(f0, f2)); \
    vsum2 = _mm_add_ps(vsum2, _mm_add_ps(f1, f3));

#define VCORE1(lo, f0, f1, tab) \
    f0 = _mm_cvtepi32_ps(_mm_unpack##lo##_epi16(v5, v7)); \
    f0 = _mm_mul_ps(f0, f0); \
    f1 = _mm_mul_ps(f0, _mm_load_ps(tab)); \
    f0 = _mm_mul_ps(f0, _mm_cvtepi32_ps(_mm_unpack##lo##_epi16(v0, v2))); \
    f0 = _mm_mul_ps(f0, f1); f1 = _mm_mul_ps(f1, f1);

#define VFIN \
    f0 = vsum1; f1 = vsum2; \
    f0 = _mm_add_ps(_mm_unpacklo_ps(f0, f1), _mm_unpackhi_ps(f0, f1)); \
    f0 = _mm_add_ps(f0, _mm_shuffle_ps(f0, f0, 0xee)); \
    a2 = _mm_cvtss_f32(f0); \
    a3 = _mm_cvtss_f32(_mm_shuffle_ps(f0, f0, 0x55));

#define VLDPIX(i, p) v##i = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)(p)));
#define VRIGHT(p) v1 = _mm_insert_epi16(_mm_bsrli_si128(v0, 2), *(JSAMPLE*)(p), 7);
#define VCOPY1 v0 = v1;
#elif 1 // vector code simulation
#define VINIT \
    int i; JSAMPLE *p0, *p1, rborder[8]; \
    float sum[8] = { 0,0,0,0, 0,0,0,0 };

#define VCORE(tab) \
    for (i = 0; i < 4; i++) { \
        float a0, a1, a4, a5, f0; \
        VCORE1(i, a0, a1, tab) VCORE1(i+4, a4, a5, tab) \
        sum[i] += a0 + a4; sum[i+4] += a1 + a5; \
    }

#define VCORE1(i, a0, a1, tab) \
    a0 = p0[i] - p1[i]; a1 = (tab)[i]; \
    f0 = (float)range-fabsf(a0); if (f0 < 0) f0 = 0; f0 *= f0; \
    a0 *= f0; a1 *= f0; a0 *= a1; a1 *= a1;

#define VFIN \
    a2 = (sum[0] + sum[2]) + (sum[1] + sum[3]); \
    a3 = (sum[4] + sum[6]) + (sum[5] + sum[7]);

#define VLDPIX(i, a) p##i = a;
#define VRIGHT(p) p1 = rborder; memcpy(p1, p0 + 1, 7 * sizeof(JSAMPLE)); p1[7] = *(p);
#define VCOPY1 p0 = p1;
#endif

#ifdef VINIT
        {
            JSAMPLE lborder[8];
            VINIT

            for (y = 0; y < DCTSIZE; y++) {
                lborder[y] = buf[y*DCTSIZE];
                VLDPIX(0, buf+y*DCTSIZE)
                VRIGHT(buf+DCTSIZE2+DCTSIZE+y)
                VCORE(tab+64+y*DCTSIZE)
            }

            VLDPIX(0, buf)
            for (y = 0; y < DCTSIZE-1; y++) {
                VLDPIX(1, buf+y*DCTSIZE+DCTSIZE)
                VCORE(tab+64*2+y*DCTSIZE)
                VCOPY1
            }

            VLDPIX(0, buf)
            VLDPIX(1, image-stride)
            VCORE(tab)
            VLDPIX(0, buf+7*DCTSIZE)
            VLDPIX(1, image+8*stride)
            VCORE(tab+7*DCTSIZE)
            VLDPIX(0, lborder)
            VLDPIX(1, buf+DCTSIZE2)
            VCORE(tab+3*DCTSIZE2)

            VFIN
        }
#undef VINIT
#undef VCORE
#ifdef VCORE1
#undef VCORE1
#endif
#undef VFIN
#undef VLDPIX
#undef VRIGHT
#undef VCOPY1
#else
        {
            int p0, p1; float a0, a1;
#define M2(a) { float x = (float)range-fabsf(a); if (x < 0) x = 0; x *= x; a *= x; a1 *= x; }
#define M1(xx, yy, i) \
    p0 = y*DCTSIZE+x; p1 = (yy)*DCTSIZE+(xx); \
    a0 = buf[p0] - buf[p1]; a1 = tab[i*64+p0]; \
    M2(a0) a2 += a0 * a1; a3 += a1 * a1;
            for (y = 0; y < 8; y++)
            for (x = 0; x < 7; x++) { M1(x+1,y,1) }
            for (y = 0; y < 7; y++)
            for (x = 0; x < 8; x++) { M1(x,y+1,2) }
#undef M1
#define M1(xx, yy) \
    p0 = y*DCTSIZE+x; \
    a0 = buf[p0] - image[(yy)*stride+(xx)]; a1 = tab[p0]; \
    M2(a0) a2 += a0 * a1; a3 += a1 * a1;
            y = 0; for (x = 0; x < 8; x++) { M1(x,y-1) }
            y = 7; for (x = 0; x < 8; x++) { M1(x,y+1) }
            x = 0; for (y = 0; y < 8; y++) { M1(x-1,y) }
            x = 7; for (y = 0; y < 8; y++) { M1(x+1,y) }
#undef M1
#undef M2
        }
#endif

        a2 = a2 / a3;

        {
            int div = quantval[i], coef1 = coef[i], add;
            int dh, dl, d0 = (div-1) >> 1, d1 = div >> 1;
            // int a0 = ((coef1 + (div >> 1)) / div) * div - (coef1 < 0 ? div : 0);
            int32_t a0 = coef1, sign = a0 >> 31;
            a0 = (a0 ^ sign) - sign;
            a0 = ((a0 + (div >> 1)) / div) * div;
            a0 = (a0 ^ sign) - sign;

            if (a0 == 0) { dh = d0; dl = -d0; }
            else if (a0 < 0) { dh = d1; dl = -d0; }
            else { dh = d0; dl = -d1; }

            add = coef1 - a0;
            add -= roundf(a2);
            if (add > dh) add = dh;
            if (add < dl) add = dl;
            add += a0;
            flag += add != coef1;
            coef[i] = add;
        }
    }
}

static void quantsmooth_transform(j_decompress_ptr srcinfo, jvirt_barray_ptr *src_coef_arrays, int flags, float gain, int niter) {
    JDIMENSION comp_width, comp_height, blk_y;
    int ci, qtblno, stride, iter;
    jpeg_component_info *compptr;
    JQUANT_TBL *qtbl; JSAMPROW image;

    for (ci = 0; ci < srcinfo->num_components; ci++) {
        compptr = srcinfo->comp_info + ci;
        stride = (srcinfo->max_h_samp_factor * DCTSIZE) / compptr->h_samp_factor;
        comp_width = (srcinfo->image_width + stride - 1) / stride;
        comp_height = compptr->height_in_blocks;

        qtblno = compptr->quant_tbl_no;
        if (qtblno < 0 || qtblno >= NUM_QUANT_TBLS || srcinfo->quant_tbl_ptrs[qtblno] == NULL) continue;
        qtbl = srcinfo->quant_tbl_ptrs[qtblno];

        // skip already processed
        {
            int i, a = 0;
            for (i = 0; i < DCTSIZE2; i++) a |= qtbl->quantval[i];
            if (a <= 1) continue;
        }

        // keeping block pointers aligned
        stride = comp_width * DCTSIZE + 8;
        image = (JSAMPROW)malloc(((comp_height * DCTSIZE + 2) * stride + 8) * sizeof(JSAMPLE));
        if (!image) continue;
        image += 7;

#ifdef WITH_LOG
        if (flags & 4) logfmt("component[%i] : size %ix%i\n", ci, comp_width, comp_height);
#endif
#define IMAGEPTR &image[((blk_y + offset_y) * DCTSIZE + 1) * stride + blk_x * DCTSIZE + 1]

#ifdef USE_JSIMD
        JSAMPROW output_buf[8] = {
            image+stride*0, image+stride*1, image+stride*2, image+stride*3,
            image+stride*4, image+stride*5, image+stride*6, image+stride*7 };
#endif

        for (iter = 0; iter < niter; iter++) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (blk_y = 0; blk_y < comp_height; blk_y += compptr->v_samp_factor) {
                JDIMENSION offset_y, blk_x;
                JDIMENSION height = comp_height - blk_y;
                if (height > compptr->v_samp_factor) height = compptr->v_samp_factor;

                JBLOCKARRAY buffer = (*srcinfo->mem->access_virt_barray)
                    ((j_common_ptr) srcinfo, src_coef_arrays[ci], blk_y,
                    (JDIMENSION) compptr->v_samp_factor, TRUE);

                for (offset_y = 0; offset_y < height; offset_y++) {
                    for (blk_x = 0; blk_x < comp_width; blk_x++) {
                        JCOEFPTR coef = buffer[offset_y][blk_x]; int x;
                        if (!iter)
                            for (x = 0; x < DCTSIZE2; x++) coef[x] *= qtbl->quantval[x];
#ifdef USE_JSIMD
                        int output_col = IMAGEPTR - image;
#endif
                        idct_islow(coef, IMAGEPTR, stride);
                    }
                }
            }

            {
                int y, w = comp_width * DCTSIZE, h = comp_height * DCTSIZE;
                memcpy(image, image + stride, stride);
                for (y = 1; y < h+1; y++) {
                    image[y*stride] = image[y*stride+1];
                    image[y*stride+w+1] = image[y*stride+w];
                }
                memcpy(image + (h+1)*stride, image + h*stride, stride);
            }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (blk_y = 0; blk_y < compptr->height_in_blocks; blk_y += compptr->v_samp_factor) {
                JDIMENSION offset_y, blk_x;
                JDIMENSION height = comp_height - blk_y;
                if (height > compptr->v_samp_factor) height = compptr->v_samp_factor;

                JBLOCKARRAY buffer = (*srcinfo->mem->access_virt_barray)
                    ((j_common_ptr) srcinfo, src_coef_arrays[ci], blk_y,
                    (JDIMENSION) compptr->v_samp_factor, TRUE);

                for (offset_y = 0; offset_y < height; offset_y++) {
                    for (blk_x = 0; blk_x < comp_width; blk_x++) {
                        JCOEFPTR coef = buffer[offset_y][blk_x];
                        quantsmooth_block(coef, qtbl->quantval, IMAGEPTR, stride, gain);
                    }
                }
            }
        } // iter
#undef IMAGEPTR
        free(image - 7);
    }
}

static void do_quantsmooth(j_decompress_ptr srcinfo, jvirt_barray_ptr *coef_arrays, int flags, float border, float gain, int niter) {
    int ci, qtblno, x, y;
    jpeg_component_info *compptr;
    JQUANT_TBL *qtbl, *c_quant;

#ifdef WITH_LOG
    if (flags & 1)
    for (ci = 0, compptr = srcinfo->comp_info; ci < srcinfo->num_components; ci++, compptr++) {
        qtblno = compptr->quant_tbl_no;
        logfmt("component[%i] : table %i, samp %ix%i\n", ci, qtblno, compptr->h_samp_factor, compptr->v_samp_factor);
        /* Make sure specified quantization table is present */
        if (qtblno < 0 || qtblno >= NUM_QUANT_TBLS || srcinfo->quant_tbl_ptrs[qtblno] == NULL) continue;
    }

    if (flags & 2)
    for (qtblno = 0; qtblno < NUM_QUANT_TBLS; qtblno++) {
        qtbl = srcinfo->quant_tbl_ptrs[qtblno];
        if (!qtbl) continue;
        logfmt("quant[%i]:\n", qtblno);

        for (y = 0; y < DCTSIZE; y++) {
            for (x = 0; x < DCTSIZE; x++) {
                logfmt("%04x ", qtbl->quantval[y * DCTSIZE + x]);
            }
            logfmt("\n");
        }
    }
#endif

    {
#ifdef WITH_LOG
        int64_t time = 0;
        if (flags & 8) time = get_time_usec();
#endif
        quantsmooth_init(border);
        quantsmooth_transform(srcinfo, coef_arrays, flags, gain, niter);
#ifdef WITH_LOG
        if (flags & 8) {
            time = get_time_usec() - time;
            logfmt("quantsmooth = %.3fms\n", time * 0.001);
        }
#endif
    }

    for (qtblno = 0; qtblno < NUM_QUANT_TBLS; qtblno++) {
        qtbl = srcinfo->quant_tbl_ptrs[qtblno];
        if (!qtbl) continue;
        for (x = 0; x < DCTSIZE2; x++) qtbl->quantval[x] = 1;
    }

    for (ci = 0, compptr = srcinfo->comp_info; ci < srcinfo->num_components; ci++, compptr++) {
        qtblno = compptr->quant_tbl_no;
        if (qtblno < 0 || qtblno >= NUM_QUANT_TBLS || srcinfo->quant_tbl_ptrs[qtblno] == NULL) continue;
        qtbl = srcinfo->quant_tbl_ptrs[qtblno];
        c_quant = compptr->quant_table;
        if (c_quant) memcpy(c_quant, qtbl, sizeof(qtbl->quantval));
    }
}
