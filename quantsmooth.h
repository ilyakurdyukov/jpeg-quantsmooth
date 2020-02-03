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

#ifdef NO_MATHLIB
#define roundf(x) (float)(int)((x) < 0 ? (x) - 0.5f : (x) + 0.5f)
#define fabsf(x) (float)((x) < 0 ? -(x) : (x))
#else
#include <math.h>
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

#ifndef NO_SIMD
#if defined(__SSE2__)
#define USE_SSE2
#include <emmintrin.h>

#if defined(__SSSE3__)
#include <tmmintrin.h>
#else
static inline __m128i SSE2_mm_abs_epi16(__m128i a) {
	__m128i t = _mm_srai_epi16(a, 15);
	return _mm_xor_si128(_mm_add_epi16(a, t), t);
}
#define _mm_abs_epi16 SSE2_mm_abs_epi16
#endif

#if defined(__SSE4_1__)
#define USE_SSE4
#include <smmintrin.h>
#else
#define _mm_cvtepu8_epi16(a) _mm_unpacklo_epi8(a, _mm_setzero_si128())
// _mm_cmplt_epi16(a, _mm_setzero_si128()) or _mm_srai_epi16(a, 15)
#define _mm_cvtepi16_epi32(a) _mm_unpacklo_epi16(a, _mm_srai_epi16(a, 15))
static inline __m128i SSE2_mm_mullo_epi32(__m128i a, __m128i b) {
	__m128i l = _mm_mul_epu32(a, b);
	__m128i h = _mm_mul_epu32(_mm_bsrli_si128(a, 4), _mm_bsrli_si128(b, 4));
	return _mm_unpacklo_epi64(_mm_unpacklo_epi32(l, h), _mm_unpackhi_epi32(l, h));
}
#define _mm_mullo_epi32 SSE2_mm_mullo_epi32
#endif
#endif // __SSE2__

#ifdef __AVX2__
#define USE_AVX2
#include <immintrin.h>
#endif

#if defined(__ARM_NEON__) || defined(__aarch64__)
#define USE_NEON
#include <arm_neon.h>
// for testing on x86
#elif defined(TEST_NEON) && defined(__SSSE3__)
#define USE_NEON
#define DO_PRAGMA(x) _Pragma(#x)
#define X(x) DO_PRAGMA(GCC diagnostic ignored #x)
X(-Wunused-function) X(-Wdeprecated-declarations)
#pragma GCC diagnostic push
X(-Wsign-compare) X(-Woverflow) X(-Wunused-parameter)
X(-Wsequence-point) X(-Wstrict-aliasing)
#undef X
#include "NEON_2_SSE.h"
#pragma GCC diagnostic pop
#warning NEON test build on x86
#elif defined(__arm__)
#warning compiling for ARM without NEON support
#endif

#endif // NO_SIMD

#define ALIGN(n) __attribute__((aligned(n)))

#include "idct.h"

static float ALIGN(32) quantsmooth_tables[DCTSIZE2][DCTSIZE2 * 2 + DCTSIZE * 4];

static void quantsmooth_init() {
	int i; JCOEF coef[DCTSIZE2];

	range_limit_init();

	for (i = 0; i < DCTSIZE2; i++) {
		float *tab = quantsmooth_tables[i];
		float temp[DCTSIZE2], bcoef = 2.0f;
		int x, y, p, n = DCTSIZE, nn = n * n, n2 = nn + n * 4;
		memset(coef, 0, sizeof(coef)); coef[i] = 1;
		idct_fslow(coef, temp);

#define M1(a, b, j) \
	for (y = 0; y < n - 1 + a; y++) \
	for (x = 0; x < n - 1 + b; x++) { p = y * n + x; \
	tab[p + j] = temp[p] - temp[(y + b) * n + x + a]; }
		M1(1, 0, 0) M1(0, 1, n2)
#undef M1
		for (y = n - 1, x = 0; x < n; x++) {
			tab[x * n + y] = tab[n2 + y * n + x] = 0;
#define M1(a, b, j) tab[nn + n * j + x] = temp[a + b * n] * bcoef;
			M1(x, 0, 0) M1(x, y, 1) M1(0, x, 2) M1(y, x, 3)
#undef M1
		}
	}
}

#if defined(USE_JSIMD) && defined(LIBJPEG_TURBO_VERSION)
#define JSIMD_CONCAT(x) jsimd_idct_islow_##x
#define JSIMD_NAME(x) JSIMD_CONCAT(x)
EXTERN(void) JSIMD_NAME(USE_JSIMD)(void*, JCOEFPTR, JSAMPARRAY, JDIMENSION);
#define idct_islow(coef, buf, st) JSIMD_NAME(USE_JSIMD)(dct_table1, coef, output_buf, output_col)
#define X 1,1,1,1, 1,1,1,1
static int16_t dct_table1[DCTSIZE2] = { X,X,X,X, X,X,X,X };
#undef X
#endif

static const char zigzag_refresh[DCTSIZE2] = {
	1, 0, 1, 0, 1, 0, 1, 0,
	1, 0, 0, 0, 0, 0, 0, 1,
	0, 0, 0, 0, 0, 0, 0, 0,
	1, 0, 0, 0, 0, 0, 0, 1,
	0, 0, 0, 0, 0, 0, 0, 0,
	1, 0, 0, 0, 0, 0, 0, 1,
	0, 0, 0, 0, 0, 0, 0, 0,
	1, 0, 1, 0, 1, 0, 1, 1
};

static void quantsmooth_block(JCOEFPTR coef, UINT16 *quantval, JSAMPROW image, int stride) {
	int k, n = DCTSIZE, x, y, flag = 1;
	JSAMPLE ALIGN(32) buf[DCTSIZE2 + DCTSIZE * 4];
#ifdef USE_JSIMD
	JSAMPROW output_buf[DCTSIZE]; int output_col = 0;
	for (k = 0; k < n; k++) output_buf[k] = buf + k * n;
#endif

#if 1 && defined(USE_NEON)
#define VINIT \
	int16x8_t v0, v5; uint16x8_t v6 = vdupq_n_u16(range); \
	float32x4_t f0, f1, f2, f3; uint8x8_t i0, i1; \
	float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;

#define VCORE(tab) \
	v0 = vreinterpretq_s16_u16(vsubl_u8(i0, i1)); \
	v5 = vreinterpretq_s16_u16(vqsubq_u16(v6, \
			vreinterpretq_u16_s16(vabsq_s16(v0)))); \
	VCORE1(low, f0, f1, tab) VCORE1(high, f2, f3, tab+4) \
	s0 = vaddq_f32(s0, f0); s1 = vaddq_f32(s1, f1); \
	s2 = vaddq_f32(s2, f2); s3 = vaddq_f32(s3, f3); \

#define VCORE1(low, f0, f1, tab) \
	f0 = vcvtq_f32_s32(vmovl_s16(vget_##low##_s16(v5))); \
	f0 = vmulq_f32(f0, f0); f1 = vmulq_f32(f0, vld1q_f32(tab)); \
	f0 = vmulq_f32(f0, vcvtq_f32_s32(vmovl_s16(vget_##low##_s16(v0)))); \
	f0 = vmulq_f32(f0, f1); f1 = vmulq_f32(f1, f1);

#define VFIN { \
	float32x4x2_t p0; float32x2_t v0; \
	p0 = vzipq_f32(vaddq_f32(s0, s2), vaddq_f32(s1, s3)); \
	f0 = vaddq_f32(p0.val[0], p0.val[1]); \
	v0 = vadd_f32(vget_low_f32(f0), vget_high_f32(f0)); \
	a2 = vget_lane_f32(v0, 0); a3 = vget_lane_f32(v0, 1); \
}

#define VLDPIX(j, p) i##j = vld1_u8(p);
#define VRIGHT i1 = vext_u8(i0, i0, 1);
#define VCOPY1 i0 = i1;

#elif 1 && defined(USE_AVX2)
#define VINIT \
	__m128i v0, v1, v5, v6 = _mm_set1_epi16(range); \
	__m256 f0, f1, f4, s0 = _mm256_setzero_ps(), s1 = s0;

#define VCORE(tab) \
	v0 = _mm_sub_epi16(v0, v1); f4 = _mm256_load_ps(tab); \
	v5 = _mm_subs_epu16(v6, _mm_abs_epi16(v0)); \
	f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v5)); \
	f0 = _mm256_mul_ps(f0, f0); f1 = _mm256_mul_ps(f0, f4); \
	f0 = _mm256_mul_ps(f0, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v0))); \
	f0 = _mm256_mul_ps(f0, f1); f1 = _mm256_mul_ps(f1, f1); \
	s0 = _mm256_add_ps(s0, f0); s1 = _mm256_add_ps(s1, f1);

#define VFIN \
	f0 = _mm256_permute2f128_ps(s0, s1, 0x20); \
	f1 = _mm256_permute2f128_ps(s0, s1, 0x31); \
	f0 = _mm256_add_ps(f0, f1); \
	f0 = _mm256_add_ps(f0, _mm256_shuffle_ps(f0, f0, 0xee)); \
	f0 = _mm256_add_ps(f0, _mm256_shuffle_ps(f0, f0, 0x55)); \
	a2 = _mm256_cvtss_f32(f0); \
	a3 = _mm_cvtss_f32(_mm256_extractf128_ps(f0, 1));

#define VLDPIX(i, p) v##i = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)(p)));
#define VRIGHT v1 = _mm_bsrli_si128(v0, 2);
#define VCOPY1 v0 = v1;

#elif 1 && defined(USE_SSE2)
#define VINIT \
	__m128i v0, v1, v2, v5, v6 = _mm_set1_epi16(range), v7 = _mm_setzero_si128(); \
	__m128 f0, f1, f2, f3, s0 = _mm_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;

#define VCORE(tab) \
	v0 = _mm_sub_epi16(v0, v1); v2 = _mm_srai_epi16(v0, 15); \
	v5 = _mm_subs_epu16(v6, _mm_abs_epi16(v0)); \
	VCORE1(lo, f0, f1, tab) VCORE1(hi, f2, f3, tab+4) \
	s0 = _mm_add_ps(s0, f0); s1 = _mm_add_ps(s1, f1); \
	s2 = _mm_add_ps(s2, f2); s3 = _mm_add_ps(s3, f3);

#define VCORE1(lo, f0, f1, tab) \
	f0 = _mm_cvtepi32_ps(_mm_unpack##lo##_epi16(v5, v7)); \
	f0 = _mm_mul_ps(f0, f0); \
	f1 = _mm_mul_ps(f0, _mm_load_ps(tab)); \
	f0 = _mm_mul_ps(f0, _mm_cvtepi32_ps(_mm_unpack##lo##_epi16(v0, v2))); \
	f0 = _mm_mul_ps(f0, f1); f1 = _mm_mul_ps(f1, f1);

#define VFIN \
	f0 = _mm_add_ps(s0, s2); f1 = _mm_add_ps(s1, s3); \
	f0 = _mm_add_ps(_mm_unpacklo_ps(f0, f1), _mm_unpackhi_ps(f0, f1)); \
	f0 = _mm_add_ps(f0, _mm_shuffle_ps(f0, f0, 0xee)); \
	a2 = _mm_cvtss_f32(f0); \
	a3 = _mm_cvtss_f32(_mm_shuffle_ps(f0, f0, 0x55));

#define VLDPIX(i, p) v##i = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)(p)));
#define VRIGHT v1 = _mm_bsrli_si128(v0, 2);
#define VCOPY1 v0 = v1;
#elif 1 // vector code simulation
#define VINIT \
	int j; JSAMPLE *p0, *p1; float a0, a1, f0, sum[DCTSIZE * 2]; \
	for (j = 0; j < n * 2; j++) sum[j] = 0;

#define VCORE(tab) \
	for (j = 0; j < n; j++) { \
		a0 = p0[j] - p1[j]; a1 = (tab)[j]; \
		f0 = (float)range - fabsf(a0); if (f0 < 0) f0 = 0; f0 *= f0; \
		a0 *= f0; a1 *= f0; a0 *= a1; a1 *= a1; \
		sum[j] += a0; sum[j + n] += a1; \
	}

#define VCORE1(sum) \
	((sum[0] + sum[4]) + (sum[1] + sum[5])) + \
	((sum[2] + sum[6]) + (sum[3] + sum[7]));
#define VFIN a2 += VCORE1(sum) a3 += VCORE1((sum+8))

#define VLDPIX(i, a) p##i = a;
#define VRIGHT p1 = p0 + 1;
#define VCOPY1 p0 = p1;
#endif

#ifdef VINIT
	JSAMPLE *border = buf + n * n;
	for (y = 0; y < n; y++) {
		border[y + n * 2] = image[y * stride - 1];
		border[y + n * 3] = image[y * stride + n];
	}
#endif

	(void)x;
	for (k = n * n - 1; k > 0; k--) {
		int i = jpeg_natural_order[k];
		float *tab = quantsmooth_tables[i], a2 = 0, a3 = 0;
		int range = quantval[i] * 2;
		if (flag && zigzag_refresh[i]) {
			idct_islow(coef, buf, n);
			flag = 0;
#ifdef VINIT
			for (y = 0; y < n; y++) {
				border[y] = buf[y * n];
				border[y + n] = buf[y * n + n - 1];
			}
#endif
		}

#ifdef VINIT
		{
			VINIT

			if (i & (n - 1))
			for (y = 0; y < n; y++) {
				VLDPIX(0, buf + y * n)
				VRIGHT
				VCORE(tab + y * n)
			}
			tab += n * n;

			VLDPIX(0, buf)
			VLDPIX(1, image - stride)
			VCORE(tab) tab += n;
			VLDPIX(0, buf + n * n - n)
			VLDPIX(1, image + n * stride)
			VCORE(tab) tab += n;

			VLDPIX(0, border)
			VLDPIX(1, border + n * 2)
			VCORE(tab) tab += n;
			VLDPIX(0, border + n)
			VLDPIX(1, border + n * 3)
			VCORE(tab) tab += n;

			if (i > (n - 1)) {
				VLDPIX(0, buf)
				for (y = 0; y < n - 1; y++) {
					VLDPIX(1, buf + y * n + n)
					VCORE(tab + y * n)
					VCOPY1
				}
			}

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
			int p; float a0, a1, t;
#define CORE t = (float)range - fabsf(a0); \
	if (t < 0) t = 0; t *= t; a0 *= t; a1 *= t; a2 += a0 * a1; a3 += a1 * a1;
#define M1(a, b) \
	for (y = 0; y < n - 1 + a; y++) \
	for (x = 0; x < n - 1 + b; x++) { p = y * n + x; \
	a0 = buf[p] - buf[(y + b) * n + x + a]; a1 = tab[p]; CORE }
#define M2(z, xx, yy) for (z = 0; z < n; z++) { p = y * n + x; \
	a0 = buf[p] - image[(yy) * stride + xx]; a1 = *tab++; CORE }
			if (i & (n - 1)) M1(1, 0) tab += n * n;
			y = 0; M2(x, x, y - 1) y = n - 1; M2(x, x, y + 1)
			x = 0; M2(y, x - 1, y) x = n - 1; M2(y, x + 1, y)
			if (i > (n - 1)) M1(0, 1)
#undef M2
#undef M1
#undef CORE
		}
#endif

		a2 = a2 / a3;

		{
			int div = quantval[i], coef1 = coef[i], add;
			int dh, dl, d0 = (div-1) >> 1, d1 = div >> 1;
			// int a0 = (coef1 + (coef1 < 0 ? -div : div) / 2) / div * div;
			int32_t a0 = coef1, sign = a0 >> 31;
			a0 = (a0 ^ sign) - sign;
			a0 = ((a0 + (div >> 1)) / div) * div;
			a0 = (a0 ^ sign) - sign;

			dh = a0 < 0 ? d1 : d0;
			dl = a0 > 0 ? -d1 : -d0;

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

static void do_quantsmooth(j_decompress_ptr srcinfo, jvirt_barray_ptr *src_coef_arrays, int flags) {
	JDIMENSION comp_width, comp_height, blk_y;
	int i, ci, stride, iter;
	jpeg_component_info *compptr;
	JQUANT_TBL *qtbl; JSAMPROW image;

#ifdef WITH_LOG
	int64_t time = 0;

	if (flags & 1)
	for (ci = 0; ci < srcinfo->num_components; ci++) {
		jpeg_component_info *compptr = srcinfo->comp_info + ci;
		i = compptr->quant_tbl_no;
		logfmt("component[%i] : table %i, samp %ix%i\n", ci, i,
				compptr->h_samp_factor, compptr->v_samp_factor);
	}

	if (flags & 2)
	for (i = 0; i < NUM_QUANT_TBLS; i++) {
		int x, y;
		qtbl = srcinfo->quant_tbl_ptrs[i];
		if (!qtbl) continue;
		logfmt("quant[%i]:\n", i);

		for (y = 0; y < DCTSIZE; y++) {
			for (x = 0; x < DCTSIZE; x++)
				logfmt("%04x ", qtbl->quantval[y * DCTSIZE + x]);
			logfmt("\n");
		}
	}

	if (flags & 8) time = get_time_usec();
#endif

	quantsmooth_init();

	for (ci = 0; ci < srcinfo->num_components; ci++) {
		compptr = srcinfo->comp_info + ci;
		stride = (srcinfo->max_h_samp_factor * DCTSIZE) / compptr->h_samp_factor;
		comp_width = (srcinfo->image_width + stride - 1) / stride;
		comp_height = compptr->height_in_blocks;

		if (!(qtbl = compptr->quant_table)) continue;

		// skip if already processed
		{
			int val = 0;
			for (i = 0; i < DCTSIZE2; i++) val |= qtbl->quantval[i];
			if (val <= 1) continue;
		}

		// keeping block pointers aligned
		stride = comp_width * DCTSIZE + 8;
		image = (JSAMPROW)malloc(((comp_height * DCTSIZE + 2) * stride + 8) * sizeof(JSAMPLE));
		if (!image) continue;
		image += 7;

#ifdef WITH_LOG
		if (flags & 4) logfmt("component[%i] : size %ix%i\n", ci, comp_width, comp_height);
#endif
#define IMAGEPTR &image[(blk_y * DCTSIZE + 1) * stride + blk_x * DCTSIZE + 1]

#ifdef USE_JSIMD
		JSAMPROW output_buf[8] = {
			image+stride*0, image+stride*1, image+stride*2, image+stride*3,
			image+stride*4, image+stride*5, image+stride*6, image+stride*7 };
#endif

		for (iter = 0; iter < 3; iter++) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
			for (blk_y = 0; blk_y < comp_height; blk_y += 1) {
				JDIMENSION blk_x;
				JBLOCKARRAY buffer = (*srcinfo->mem->access_virt_barray)
						((j_common_ptr)srcinfo, src_coef_arrays[ci], blk_y, 1, TRUE);

				for (blk_x = 0; blk_x < comp_width; blk_x++) {
					JCOEFPTR coef = buffer[0][blk_x]; int i;
					if (!iter)
						for (i = 0; i < DCTSIZE2; i++) coef[i] *= qtbl->quantval[i];
#ifdef USE_JSIMD
					int output_col = IMAGEPTR - image;
#endif
					idct_islow(coef, IMAGEPTR, stride);
				}
			}

			{
				int y, w = comp_width * DCTSIZE, h = comp_height * DCTSIZE;
				for (y = 1; y < h+1; y++) {
					image[y*stride] = image[y*stride+1];
					image[y*stride+w+1] = image[y*stride+w];
				}
				memcpy(image, image + stride, stride);
				memcpy(image + (h+1)*stride, image + h*stride, stride);
			}

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
			for (blk_y = 0; blk_y < comp_height; blk_y += 1) {
				JDIMENSION blk_x;
				JBLOCKARRAY buffer = (*srcinfo->mem->access_virt_barray)
						((j_common_ptr)srcinfo, src_coef_arrays[ci], blk_y, 1, TRUE);

				for (blk_x = 0; blk_x < comp_width; blk_x++) {
					JCOEFPTR coef = buffer[0][blk_x];
					quantsmooth_block(coef, qtbl->quantval, IMAGEPTR, stride);
				}
			}
		} // iter
#undef IMAGEPTR
		free(image - 7);
	}

#ifdef WITH_LOG
	if (flags & 8) {
		time = get_time_usec() - time;
		logfmt("quantsmooth = %.3fms\n", time * 0.001);
	}
#endif

	for (ci = 0; ci < NUM_QUANT_TBLS; ci++) {
		qtbl = srcinfo->quant_tbl_ptrs[ci];
		if (qtbl) for (i = 0; i < DCTSIZE2; i++) qtbl->quantval[i] = 1;
	}

	for (ci = 0; ci < srcinfo->num_components; ci++) {
		qtbl = srcinfo->comp_info[ci].quant_table;
		if (qtbl) for (i = 0; i < DCTSIZE2; i++) qtbl->quantval[i] = 1;
	}
}

