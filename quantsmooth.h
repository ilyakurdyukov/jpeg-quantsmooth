/*
 * Copyright (C) 2016-2020 Ilya Kurdyukov
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

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#if !defined(TRANSCODE_ONLY) && !defined(JPEG_INTERNALS)
// declarations needed from jpegint.h

#define DSTATE_SCANNING 205
#define DSTATE_RAW_OK 206

EXTERN(void) jinit_d_main_controller(j_decompress_ptr, boolean);
EXTERN(void) jinit_inverse_dct(j_decompress_ptr);
EXTERN(void) jinit_upsampler(j_decompress_ptr);
EXTERN(void) jinit_color_deconverter(j_decompress_ptr);

struct jpeg_decomp_master {
  void (*prepare_for_output_pass) (j_decompress_ptr);
  void (*finish_output_pass) (j_decompress_ptr);
  boolean is_dummy_pass;
#ifdef LIBJPEG_TURBO_VERSION
  JDIMENSION first_iMCU_col, last_iMCU_col;
  JDIMENSION first_MCU_col[MAX_COMPONENTS];
  JDIMENSION last_MCU_col[MAX_COMPONENTS];
  boolean jinit_upsampler_no_alloc;
#endif
};
#endif

#ifdef WITH_LOG
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifdef INT32
#undef INT32
#endif
// conflict with libjpeg typedef
#define INT32 INT32_WIN
#include <windows.h>
static int64_t get_time_usec() {
	LARGE_INTEGER freq, perf;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&perf);
	return perf.QuadPart * 1000000.0 / freq.QuadPart;
}
#else
#include <time.h>
#include <sys/time.h>
static int64_t get_time_usec() {
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

#ifdef __FMA__
#include <immintrin.h>
#else
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(_mm256_mul_ps(a, b), c)
#define _mm256_fmsub_ps(a, b, c) _mm256_sub_ps(_mm256_mul_ps(a, b), c)
#define _mm256_fnmadd_ps(a, b, c) _mm256_sub_ps(c, _mm256_mul_ps(a, b))
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
#ifndef JPEGQS_ATTR
#define JPEGQS_ATTR static
#endif
#include "libjpegqs.h"

static float** quantsmooth_init(int flags) {
	int i, n = DCTSIZE, nn = n * n, n2 = nn + n * 4;
	float bcoef = flags & JPEGQS_DIAGONALS ? 4.0 : 2.0;
	int size = flags & JPEGQS_DIAGONALS ? nn * 4 + n * (4 - 2) : nn * 2 + n * 4;
	float *ptr, **tables = (float**)malloc(nn * sizeof(float*) + nn * size * sizeof(float) + 31);
	if (!tables) return NULL;
	ptr = (float*)(((intptr_t)&tables[DCTSIZE2] + 31) & -32);
	for (i = nn - 1; i >= 0; i--, ptr += size)
		tables[(int)jpegqs_natural_order[i]] = ptr;

	for (i = 0; i < DCTSIZE2; i++) {
		float *tab = tables[i], temp[DCTSIZE2];
		int x, y, p;
		memset(temp, 0, sizeof(temp)); temp[i] = 1;
		idct_float(temp, temp);

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

		if (flags & JPEGQS_DIAGONALS) {
			tab += nn * 2 + n * 4;
			for (y = 0; y < n - 1; y++, tab += n * 2) {
				for (x = 0; x < n - 1; x++) {
					p = y * n + x;
					tab[x] = temp[p] - temp[p + n + 1];
					tab[x + n] = temp[p + 1] - temp[p + n];
				}
				tab[x] = tab[x + n] = 0;
			}
		}
	}
	return tables;
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

static void fdct_clamp(float *buf, JCOEFPTR coef, UINT16 *quantval) {
	int x, y, n = DCTSIZE;
	(void)x; (void)y;

	fdct_float(buf, buf);
#if 1 && defined(USE_AVX2)
	if (sizeof(quantval[0]) == 2 && sizeof(quantval[0]) == sizeof(coef[0]))
	for (y = 0; y < n; y++) {
		__m128i v0, v1, v2, v3; __m256i v4, v5; __m256 f0, f1;
		v1 = _mm_loadu_si128((void*)&quantval[y * n]);
		v0 = _mm_loadu_si128((void*)&coef[y * n]);
		v2 = _mm_srli_epi16(v1, 1); v3 = _mm_srai_epi16(v0, 15);
		v2 = _mm_xor_si128(_mm_add_epi16(v2, v3), v3);
		v0 = _mm_add_epi16(v0, v2);
		f0 = _mm256_loadu_ps((void*)&buf[y * n]);
		f1 = _mm256_blendv_ps(_mm256_set1_ps(0.5f), _mm256_set1_ps(-0.5f), f0);
		f0 = _mm256_add_ps(f0, f1);
		f1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v0));
		f1 = _mm256_div_ps(f1, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v1)));
		v4 = _mm256_cvtps_epi32(_mm256_round_ps(f0, 3 | 8));
		v5 = _mm256_cvtps_epi32(_mm256_round_ps(f1, 3 | 8));
		v4 = _mm256_permute4x64_epi64(_mm256_packs_epi32(v5, v4), 0xd8);
		v0 = _mm256_castsi256_si128(v4); v2 = _mm256_extracti128_si256(v4, 1);
		v0 = _mm_mullo_epi16(v0, v1); /* v0 = a0, v1 = div, v2 = add */
		v1 = _mm_add_epi16(v1, _mm_set1_epi16(-1));
		v3 = _mm_sub_epi16(v1, _mm_srai_epi16(v0, 15));
		v2 = _mm_min_epi16(v2, _mm_add_epi16(v0, _mm_srai_epi16(v3, 1)));
		v3 = _mm_sub_epi16(v1, _mm_cmpgt_epi16(v0, _mm_setzero_si128()));
		v2 = _mm_max_epi16(v2, _mm_sub_epi16(v0, _mm_srai_epi16(v3, 1)));
		_mm_storeu_si128((void*)&coef[y * n], v2);
	} else
#elif 1 && defined(USE_SSE2)
	if (sizeof(quantval[0]) == 2 && sizeof(quantval[0]) == sizeof(coef[0])) {
	unsigned mxcsr = _mm_getcsr();
	for (y = 0; y < n; y++) {
		__m128i v0, v1, v2, v3; __m128 f0, f1, f2, f3, f4;
		v1 = _mm_loadu_si128((void*)&quantval[y * n]);
		v0 = _mm_loadu_si128((void*)&coef[y * n]);
		v2 = _mm_srli_epi16(v1, 1); v3 = _mm_srai_epi16(v0, 15);
		v2 = _mm_xor_si128(_mm_add_epi16(v2, v3), v3);
		v0 = _mm_add_epi16(v0, v2);
		v2 = _mm_setzero_si128(); v3 = _mm_srai_epi16(v0, 15);
#define M1(lo, f0, f1, x) \
	f0 = _mm_loadu_ps((void*)&buf[y * n + x]); \
	f4 = _mm_cmplt_ps(f0, _mm_setzero_ps()); \
	f4 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(f4), 31)); \
	f0 = _mm_add_ps(f0, _mm_or_ps(f4, _mm_set1_ps(0.5f))); \
	f4 = _mm_cvtepi32_ps(_mm_unpack##lo##_epi16(v1, v2)); \
	f1 = _mm_cvtepi32_ps(_mm_unpack##lo##_epi16(v0, v3)); \
	f1 = _mm_div_ps(f1, f4);
		M1(lo, f0, f1, 0) M1(hi, f2, f3, 4)
#undef M1
		_mm_setcsr(mxcsr | 0x6000);
		v2 = _mm_packs_epi32(_mm_cvtps_epi32(f0), _mm_cvtps_epi32(f2));
		v0 = _mm_packs_epi32(_mm_cvtps_epi32(f1), _mm_cvtps_epi32(f3));
		_mm_setcsr(mxcsr);
		v0 = _mm_mullo_epi16(v0, v1); /* v0 = a0, v1 = div, v2 = add */
		v1 = _mm_add_epi16(v1, _mm_set1_epi16(-1));
		v3 = _mm_sub_epi16(v1, _mm_srai_epi16(v0, 15));
		v2 = _mm_min_epi16(v2, _mm_add_epi16(v0, _mm_srai_epi16(v3, 1)));
		v3 = _mm_sub_epi16(v1, _mm_cmpgt_epi16(v0, _mm_setzero_si128()));
		v2 = _mm_max_epi16(v2, _mm_sub_epi16(v0, _mm_srai_epi16(v3, 1)));
		_mm_storeu_si128((void*)&coef[y * n], v2);
	} } else
#endif
	for (x = 0; x < n * n; x++) {
		int div = quantval[x], coef1 = coef[x], add;
		int dh, dl, d0 = (div - 1) >> 1, d1 = div >> 1;
		int a0 = (coef1 + (coef1 < 0 ? -d1 : d1)) / div * div;
		dh = a0 + (a0 < 0 ? d1 : d0);
		dl = a0 - (a0 > 0 ? d1 : d0);
		add = roundf(buf[x]);
		if (add > dh) add = dh;
		if (add < dl) add = dl;
		coef[x] = add;
	}
}

static void quantsmooth_block(JCOEFPTR coef, UINT16 *quantval,
		JSAMPLE *image, JSAMPLE *image2, int stride, int flags, float **tables) {
	int k, n = DCTSIZE, x, y, need_refresh = 1;
	JSAMPLE ALIGN(32) buf[DCTSIZE2 + DCTSIZE * 6], *border = buf + n * n;
#ifdef USE_JSIMD
	JSAMPROW output_buf[DCTSIZE]; int output_col = 0;
	for (k = 0; k < n; k++) output_buf[k] = buf + k * n;
#endif
	(void)x;

	if (image2) {
		float ALIGN(32) buf[DCTSIZE2];
#if 1 && defined(USE_AVX2)
		for (y = 0; y < n; y++) {
			__m128i v0, v1; __m256i v2, v3, v4, sumA, sumB, sumAA, sumAB;
			__m256 v5, scale, offset;
#define M1(x0, y0, x1, y1) \
	v0 = _mm_loadl_epi64((void*)&image2[(y + y0) * stride + x0]); \
	v1 = _mm_loadl_epi64((void*)&image2[(y + y1) * stride + x1]); \
	v2 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(v0, v1)); \
	v0 = _mm_loadl_epi64((void*)&image[(y + y0) * stride + x0]); \
	v1 = _mm_loadl_epi64((void*)&image[(y + y1) * stride + x1]); \
	v3 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(v0, v1)); \
	sumA = _mm256_add_epi16(sumA, v2); \
	sumB = _mm256_add_epi16(sumB, v3); \
	sumAA = _mm256_add_epi32(sumAA, _mm256_madd_epi16(v2, v2)); \
	sumAB = _mm256_add_epi32(sumAB, _mm256_madd_epi16(v2, v3));
			v0 = _mm_loadl_epi64((void*)&image2[y * stride]);
			v1 = _mm_loadl_epi64((void*)&image[y * stride]);
			sumA = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(v0, v0));
			sumB = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(v1, v1));
			sumAA = _mm256_madd_epi16(sumA, sumA);
			sumAB = _mm256_madd_epi16(sumA, sumB);
			M1(0, -1, -1, 0) M1(1, 0, 0, 1)
			sumA = _mm256_add_epi16(sumA, sumA); sumAA = _mm256_add_epi32(sumAA, sumAA);
			sumB = _mm256_add_epi16(sumB, sumB); sumAB = _mm256_add_epi32(sumAB, sumAB);
			M1(-1, -1, 1, -1) M1(-1, 1, 1, 1)
#undef M1
			v3 = _mm256_set1_epi16(1);
			v2 = _mm256_madd_epi16(sumA, v3); sumAA = _mm256_slli_epi32(sumAA, 4);
			v3 = _mm256_madd_epi16(sumB, v3); sumAB = _mm256_slli_epi32(sumAB, 4);
			sumAA = _mm256_sub_epi32(sumAA, _mm256_mullo_epi32(v2, v2));
			sumAB = _mm256_sub_epi32(sumAB, _mm256_mullo_epi32(v2, v3));
			v4 = _mm256_cmpeq_epi32(sumAA, _mm256_setzero_si256());
			sumAB = _mm256_andnot_si256(v4, sumAB);
			scale = _mm256_cvtepi32_ps(_mm256_or_si256(sumAA, v4));
			scale = _mm256_div_ps(_mm256_cvtepi32_ps(sumAB), scale);
			scale = _mm256_max_ps(scale, _mm256_set1_ps(-16.0f));
			scale = _mm256_min_ps(scale, _mm256_set1_ps(16.0f));
			v5 = _mm256_cvtepi32_ps(v2); offset = _mm256_cvtepi32_ps(v3);
			// offset = _mm256_sub_ps(offset, _mm256_mul_ps(v5, scale));
			offset = _mm256_fnmadd_ps(v5, scale, offset);
			offset = _mm256_mul_ps(offset, _mm256_set1_ps(1.0f / 16));
			v0 = _mm_loadl_epi64((void*)&image2[y * stride]);
			v5 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v0));
			// v5 = _mm256_add_ps(_mm256_mul_ps(v5, scale), offset);
			v5 = _mm256_fmadd_ps(v5, scale, offset);
			v5 = _mm256_max_ps(v5, _mm256_setzero_ps());
			v5 = _mm256_min_ps(v5, _mm256_set1_ps(MAXJSAMPLE));
			v5 = _mm256_sub_ps(v5, _mm256_set1_ps(CENTERJSAMPLE));
			_mm256_storeu_ps(buf + y * n, v5);
		}
#elif 1 && defined(USE_SSE2)
		for (y = 0; y < n; y++) {
			__m128i v0, v1, v2, v3, v4, sumA, sumB, sumAA1, sumAB1, sumAA2, sumAB2;
			__m128 v5, scale, offset;
#define M1(x0, y0, x1, y1) \
	v0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&image2[(y + y0) * stride + x0])); \
	v1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&image2[(y + y1) * stride + x1])); \
	v2 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&image[(y + y0) * stride + x0])); \
	v3 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&image[(y + y1) * stride + x1])); \
	sumA = _mm_add_epi16(_mm_add_epi16(sumA, v0), v1); \
	sumB = _mm_add_epi16(_mm_add_epi16(sumB, v2), v3); \
	v4 = _mm_unpacklo_epi16(v0, v1); sumAA1 = _mm_add_epi32(sumAA1, _mm_madd_epi16(v4, v4)); \
	v1 = _mm_unpackhi_epi16(v0, v1); sumAA2 = _mm_add_epi32(sumAA2, _mm_madd_epi16(v1, v1)); \
	sumAB1 = _mm_add_epi32(sumAB1, _mm_madd_epi16(v4, _mm_unpacklo_epi16(v2, v3))); \
	sumAB2 = _mm_add_epi32(sumAB2, _mm_madd_epi16(v1, _mm_unpackhi_epi16(v2, v3)));
			v0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&image2[y * stride]));
			v1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&image[y * stride]));
			v2 = _mm_unpacklo_epi16(v0, v0); sumAA1 = _mm_madd_epi16(v2, v2);
			v3 = _mm_unpacklo_epi16(v1, v1); sumAB1 = _mm_madd_epi16(v2, v3);
			v2 = _mm_unpackhi_epi16(v0, v0); sumAA2 = _mm_madd_epi16(v2, v2);
			v3 = _mm_unpackhi_epi16(v1, v1); sumAB2 = _mm_madd_epi16(v2, v3);
			sumA = _mm_add_epi16(v0, v0); sumB = _mm_add_epi16(v1, v1);
			M1(0, -1, -1, 0) M1(1, 0, 0, 1)
			sumA = _mm_add_epi16(sumA, sumA); sumB = _mm_add_epi16(sumB, sumB);
			sumAA1 = _mm_add_epi32(sumAA1, sumAA1); sumAA2 = _mm_add_epi32(sumAA2, sumAA2);
			sumAB1 = _mm_add_epi32(sumAB1, sumAB1); sumAB2 = _mm_add_epi32(sumAB2, sumAB2);
			M1(-1, -1, 1, -1) M1(-1, 1, 1, 1)
#undef M1
			v0 = _mm_setzero_si128();
			v1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&image2[y * stride]));
#define M1(lo, sumAA, sumAB, x)  \
	v2 = _mm_unpack##lo##_epi16(sumA, v0); sumAA = _mm_slli_epi32(sumAA, 4); \
	v3 = _mm_unpack##lo##_epi16(sumB, v0); sumAB = _mm_slli_epi32(sumAB, 4); \
	sumAA = _mm_sub_epi32(sumAA, _mm_mullo_epi32(v2, v2)); \
	sumAB = _mm_sub_epi32(sumAB, _mm_mullo_epi32(v2, v3)); \
	v4 = _mm_cmpeq_epi32(sumAA, v0); sumAB = _mm_andnot_si128(v4, sumAB); \
	scale = _mm_cvtepi32_ps(_mm_or_si128(sumAA, v4)); \
	scale = _mm_div_ps(_mm_cvtepi32_ps(sumAB), scale); \
	scale = _mm_max_ps(scale, _mm_set1_ps(-16.0f)); \
	scale = _mm_min_ps(scale, _mm_set1_ps(16.0f)); \
	v5 = _mm_mul_ps(_mm_cvtepi32_ps(v2), scale); \
	offset = _mm_sub_ps(_mm_cvtepi32_ps(v3), v5); \
	offset = _mm_mul_ps(offset, _mm_set1_ps(1.0f / 16)); \
	v5 = _mm_cvtepi32_ps(_mm_unpack##lo##_epi16(v1, v0)); \
	v5 = _mm_add_ps(_mm_mul_ps(v5, scale), offset); \
	v5 = _mm_max_ps(v5, _mm_setzero_ps()); \
	v5 = _mm_min_ps(v5, _mm_set1_ps(MAXJSAMPLE)); \
	v5 = _mm_sub_ps(v5, _mm_set1_ps(CENTERJSAMPLE)); \
	_mm_storeu_ps(buf + y * n + x, v5);
			M1(lo, sumAA1, sumAB1, 0) M1(hi, sumAA2, sumAB2, 4)
#undef M1
		}
#else
		for (y = 0; y < n; y++)
		for (x = 0; x < n; x++) {
			float sumA = 0, sumB = 0, sumAA = 0, sumAB = 0;
			float divN = 1.0f / 16, scale, offset; float a;
#define M1(xx, yy) { \
	float a = image2[(y + yy) * stride + x + xx]; \
	float b = image[(y + yy) * stride + x + xx]; \
	sumA += a; sumAA += a * a; \
	sumB += b; sumAB += a * b; }
#define M2 sumA += sumA; sumB += sumB; \
	sumAA += sumAA; sumAB += sumAB;
			M1(0, 0) M2
			M1(0, -1) M1(-1, 0) M1(1, 0) M1(0, 1) M2
			M1(-1, -1) M1(1, -1) M1(-1, 1) M1(1, 1)
#undef M2
#undef M1
			scale = sumAA - sumA * divN * sumA;
			if (scale != 0.0f) scale = (sumAB - sumA * divN * sumB) / scale;
			scale = scale < -16.0f ? -16.0f : scale;
			scale = scale > 16.0f ? 16.0f : scale;
			offset = (sumB - scale * sumA) * divN;

			a = image2[y * stride + x] * scale + offset;
			a = a < 0 ? 0 : a > MAXJSAMPLE ? MAXJSAMPLE : a;
			buf[y * n + x] = a - CENTERJSAMPLE;
		}
#endif
		fdct_clamp(buf, coef, quantval);
	}

	if (flags & JPEGQS_LOW_QUALITY) {
		float ALIGN(32) buf[DCTSIZE2];
		float range = 0, c0 = 2, c1 = c0 * sqrtf(0.5f);

		if (image2) return;
		{
			int sum = 0;
			for (x = 1; x < n * n; x++) {
				int a = coef[x]; a = a < 0 ? -a : a;
				range += quantval[x] * a; sum += a;
			}
			if (sum) range *= 4.0f / sum;
			if (range > CENTERJSAMPLE) range = CENTERJSAMPLE;
			range = roundf(range);
		}

#if 1 && defined(USE_AVX2)
		for (y = 0; y < n; y++) {
			__m128i v0, v1, v4, v5, v6 = _mm_set1_epi16((int)range);
			__m256 f0, f1, f4, f5, s0 = _mm256_setzero_ps(), s1 = s0;
			f4 = _mm256_set1_ps(c0); f5 = _mm256_set1_ps(c1);
			v0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&image[y * stride]));
#define M1(f4, x, y) \
	v1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&image[(y) * stride + x])); \
	v4 = _mm_sub_epi16(v0, v1); v5 = _mm_subs_epu16(v6, _mm_abs_epi16(v4)); \
	f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v5)); \
	f0 = _mm256_mul_ps(f0, f0); f1 = _mm256_mul_ps(f0, f4); \
	f0 = _mm256_mul_ps(f0, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v4))); \
	s0 = _mm256_fmadd_ps(f0, f1, s0); s1 = _mm256_fmadd_ps(f1, f1, s1);
			M1(f5, -1, y-1) M1(f4, 0, y-1) M1(f5, 1, y-1)
			M1(f4, -1, y)                  M1(f4, 1, y)
			M1(f5, -1, y+1) M1(f4, 0, y+1) M1(f5, 1, y+1)
#undef M1
			f1 = _mm256_cmp_ps(s1, _mm256_setzero_ps(), 0);
			s1 = _mm256_blendv_ps(s1, _mm256_set1_ps(1.0f), f1);
			f0 = _mm256_div_ps(s0, s1);
			f1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v0));
			f0 = _mm256_sub_ps(f1, f0);
			f0 = _mm256_max_ps(f0, _mm256_setzero_ps());
			f0 = _mm256_min_ps(f0, _mm256_set1_ps(MAXJSAMPLE));
			f0 = _mm256_sub_ps(f0, _mm256_set1_ps(CENTERJSAMPLE));
			_mm256_storeu_ps(buf + y * n, f0);
		}
#elif 1 && defined(USE_SSE2)
		for (y = 0; y < n; y++) {
			__m128i v0, v1, v3, v4, v5, v6 = _mm_set1_epi16((int)range), v7 = _mm_setzero_si128();
			__m128 f0, f1, f4, f5, s0 = _mm_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
			f4 = _mm_set1_ps(c0); f5 = _mm_set1_ps(c1);
			v0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&image[y * stride]));
#define M1(f4, x, y) \
	v1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&image[(y) * stride + x])); \
	v4 = _mm_sub_epi16(v0, v1); v3 = _mm_srai_epi16(v4, 15); \
	v5 = _mm_subs_epu16(v6, _mm_abs_epi16(v4)); \
	M2(lo, s0, s1, f4) M2(hi, s2, s3, f4)
#define M2(lo, s0, s1, f4) \
	f0 = _mm_cvtepi32_ps(_mm_unpack##lo##_epi16(v5, v7)); \
	f0 = _mm_mul_ps(f0, f0); f1 = _mm_mul_ps(f0, f4); \
	f0 = _mm_mul_ps(f0, _mm_cvtepi32_ps(_mm_unpack##lo##_epi16(v4, v3))); \
	f0 = _mm_mul_ps(f0, f1); f1 = _mm_mul_ps(f1, f1); \
	s0 = _mm_add_ps(s0, f0); s1 = _mm_add_ps(s1, f1);
			M1(f5, -1, y-1) M1(f4, 0, y-1) M1(f5, 1, y-1)
			M1(f4, -1, y)                  M1(f4, 1, y)
			M1(f5, -1, y+1) M1(f4, 0, y+1) M1(f5, 1, y+1)
#undef M1
#undef M2
#define M1(lo, s0, s1, x) \
	f1 = _mm_cmpeq_ps(s1, _mm_setzero_ps()); \
	f1 = _mm_and_ps(f1, _mm_set1_ps(1.0f)); \
	f0 = _mm_div_ps(s0, _mm_or_ps(s1, f1)); \
	f1 = _mm_cvtepi32_ps(_mm_unpack##lo##_epi16(v0, v7)); \
	f0 = _mm_sub_ps(f1, f0); \
	f0 = _mm_max_ps(f0, _mm_setzero_ps()); \
	f0 = _mm_min_ps(f0, _mm_set1_ps(MAXJSAMPLE)); \
	f0 = _mm_sub_ps(f0, _mm_set1_ps(CENTERJSAMPLE)); \
	_mm_storeu_ps(buf + y * n + x, f0);
			M1(lo, s0, s1, 0) M1(hi, s2, s3, 4)
#undef M1
		}
#else
		for (y = 0; y < n; y++)
		for (x = 0; x < n; x++) {
#define M1(i, x, y) t0 = a - image[(y) * stride + x]; \
	t = range - fabsf(t0); if (t < 0) t = 0; t *= t; aw = c##i * t; \
	a0 += t0 * t * aw; an += aw * aw;
			int a = image[(y)*stride+(x)];
			float a0 = 0, an = 0, aw, t, t0;
			M1(1, x-1, y-1) M1(0, x, y-1) M1(1, x+1, y-1)
			M1(0, x-1, y)                 M1(0, x+1, y)
			M1(1, x-1, y+1) M1(0, x, y+1) M1(1, x+1, y+1)
#undef M1
			if (an > 0.0f) a -= a0 / an;
			a = a < 0 ? 0 : a > MAXJSAMPLE ? MAXJSAMPLE : a;
			buf[y * n + x] = a - CENTERJSAMPLE;
		}
#endif
		fdct_clamp(buf, coef, quantval);
		return;
	}

#if 1 && defined(USE_NEON)
#define VINIT \
	int16x8_t v0, v5; uint16x8_t v6 = vdupq_n_u16(range); \
	float32x4_t f0, f1; uint8x8_t i0, i1, i2; \
	float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;

#define VCORE(tab) \
	v0 = vreinterpretq_s16_u16(vsubl_u8(i0, i1)); \
	v5 = vreinterpretq_s16_u16(vqsubq_u16(v6, \
			vreinterpretq_u16_s16(vabsq_s16(v0)))); \
	VCORE1(low, s0, s1, tab) VCORE1(high, s2, s3, tab+4)

#define VCORE1(low, s0, s1, tab) \
	f0 = vcvtq_f32_s32(vmovl_s16(vget_##low##_s16(v5))); \
	f0 = vmulq_f32(f0, f0); f1 = vmulq_f32(f0, vld1q_f32(tab)); \
	f0 = vmulq_f32(f0, vcvtq_f32_s32(vmovl_s16(vget_##low##_s16(v0)))); \
	s0 = vmlaq_f32(s0, f0, f1); s1 = vmlaq_f32(s1, f1, f1);

#define VFIN { \
	float32x4x2_t p0; float32x2_t v0; \
	p0 = vzipq_f32(vaddq_f32(s0, s2), vaddq_f32(s1, s3)); \
	f0 = vaddq_f32(p0.val[0], p0.val[1]); \
	v0 = vadd_f32(vget_low_f32(f0), vget_high_f32(f0)); \
	a2 = vget_lane_f32(v0, 0); a3 = vget_lane_f32(v0, 1); \
}

#define VLDPIX(j, p) i##j = vld1_u8(p);
#define VRIGHT(a, b) i##a = vext_u8(i##b, i##b, 1);
#define VCOPY(a, b) i##a = i##b;

#elif 1 && defined(USE_AVX2)
#define VINIT \
	__m128i v0, v1, v2, v4, v5, v6 = _mm_set1_epi16(range); \
	__m256 f0, f1, f4, s0 = _mm256_setzero_ps(), s1 = s0;

#define VCORE(tab) \
	v4 = _mm_sub_epi16(v0, v1); f4 = _mm256_load_ps(tab); \
	v5 = _mm_subs_epu16(v6, _mm_abs_epi16(v4)); \
	f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v5)); \
	f0 = _mm256_mul_ps(f0, f0); f1 = _mm256_mul_ps(f0, f4); \
	f0 = _mm256_mul_ps(f0, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v4))); \
	s0 = _mm256_fmadd_ps(f0, f1, s0); s1 = _mm256_fmadd_ps(f1, f1, s1);

#define VFIN \
	f0 = _mm256_permute2f128_ps(s0, s1, 0x20); \
	f1 = _mm256_permute2f128_ps(s0, s1, 0x31); \
	f0 = _mm256_add_ps(f0, f1); \
	f0 = _mm256_add_ps(f0, _mm256_shuffle_ps(f0, f0, 0xee)); \
	f0 = _mm256_add_ps(f0, _mm256_shuffle_ps(f0, f0, 0x55)); \
	a2 = _mm256_cvtss_f32(f0); \
	a3 = _mm_cvtss_f32(_mm256_extractf128_ps(f0, 1));

#define VLDPIX(i, p) v##i = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)(p)));
#define VRIGHT(a, b) v##a = _mm_bsrli_si128(v##b, 2);
#define VCOPY(a, b) v##a = v##b;

#elif 1 && defined(USE_SSE2)
#define VINIT \
	__m128i v0, v1, v2, v3, v4, v5, v6 = _mm_set1_epi16(range), v7 = _mm_setzero_si128(); \
	__m128 f0, f1, s0 = _mm_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;

#define VCORE(tab) \
	v4 = _mm_sub_epi16(v0, v1); v3 = _mm_srai_epi16(v4, 15); \
	v5 = _mm_subs_epu16(v6, _mm_abs_epi16(v4)); \
	VCORE1(lo, s0, s1, tab) VCORE1(hi, s2, s3, tab+4)

#define VCORE1(lo, s0, s1, tab) \
	f0 = _mm_cvtepi32_ps(_mm_unpack##lo##_epi16(v5, v7)); \
	f0 = _mm_mul_ps(f0, f0); \
	f1 = _mm_mul_ps(f0, _mm_load_ps(tab)); \
	f0 = _mm_mul_ps(f0, _mm_cvtepi32_ps(_mm_unpack##lo##_epi16(v4, v3))); \
	f0 = _mm_mul_ps(f0, f1); f1 = _mm_mul_ps(f1, f1); \
	s0 = _mm_add_ps(s0, f0); s1 = _mm_add_ps(s1, f1);

#define VFIN \
	f0 = _mm_add_ps(s0, s2); f1 = _mm_add_ps(s1, s3); \
	f0 = _mm_add_ps(_mm_unpacklo_ps(f0, f1), _mm_unpackhi_ps(f0, f1)); \
	f0 = _mm_add_ps(f0, _mm_shuffle_ps(f0, f0, 0xee)); \
	a2 = _mm_cvtss_f32(f0); \
	a3 = _mm_cvtss_f32(_mm_shuffle_ps(f0, f0, 0x55));

#define VLDPIX(i, p) v##i = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)(p)));
#define VRIGHT(a, b) v##a = _mm_bsrli_si128(v##b, 2);
#define VCOPY(a, b) v##a = v##b;
#elif !defined(NO_SIMD) // vector code simulation
#define VINIT \
	int j; JSAMPLE *p0, *p1, *p2; float a0, a1, f0, sum[DCTSIZE * 2]; \
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
#define VRIGHT(a, b) p##a = p##b + 1;
#define VCOPY(a, b) p##a = p##b;
#endif

	for (y = 0; y < n; y++) {
		border[y + n * 2] = image[y - stride];
		border[y + n * 3] = image[y + stride * n];
		border[y + n * 4] = image[y * stride - 1];
		border[y + n * 5] = image[y * stride + n];
	}

	for (k = n * n - 1; k > 0; k--) {
		int i = jpegqs_natural_order[k];
		float *tab = tables[i], a2 = 0, a3 = 0;
		int range = quantval[i] * 2;
		if (need_refresh && zigzag_refresh[i]) {
			idct_islow(coef, buf, n);
			need_refresh = 0;
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
				VRIGHT(1, 0)
				VCORE(tab + y * n)
			}
			tab += n * n;

			VLDPIX(0, buf)
			VLDPIX(1, border + n * 2)
			VCORE(tab) tab += n;
			VLDPIX(0, buf + n * n - n)
			VLDPIX(1, border + n * 3)
			VCORE(tab) tab += n;

			VLDPIX(0, border)
			VLDPIX(1, border + n * 4)
			VCORE(tab) tab += n;
			VLDPIX(0, border + n)
			VLDPIX(1, border + n * 5)
			VCORE(tab) tab += n;

			if (i > (n - 1)) {
				VLDPIX(0, buf)
				for (y = 0; y < n - 1; y++) {
					VLDPIX(1, buf + y * n + n)
					VCORE(tab + y * n)
					VCOPY(0, 1)
				}
			}

			if (flags & JPEGQS_DIAGONALS) {
				tab += n * n;
				VLDPIX(0, buf)
				for (y = 0; y < n - 1; y++) {
					VLDPIX(2, buf + y * n + n)
					VRIGHT(1, 2)
					VCORE(tab + y * n * 2)
					VRIGHT(0, 0)
					VCOPY(1, 2)
					VCORE(tab + y * n * 2 + n)
					VCOPY(0, 2)
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
#define M2(z, i) for (z = 0; z < n; z++) { p = y * n + x; \
	a0 = buf[p] - border[i * n + z]; a1 = *tab++; CORE }
			if (i & (n - 1)) M1(1, 0) tab += n * n;
			y = 0; M2(x, 2) y = n - 1; M2(x, 3)
			x = 0; M2(y, 4) x = n - 1; M2(y, 5)
			if (i > (n - 1)) M1(0, 1)

			if (flags & JPEGQS_DIAGONALS) {
				tab += n * n;
				for (y = 0; y < n - 1; y++, tab += n * 2)
				for (x = 0; x < n - 1; x++) {
					p = y * n + x;
					a0 = buf[p] - buf[p + n + 1]; a1 = tab[x]; CORE
					a0 = buf[p + 1] - buf[p + n]; a1 = tab[x + n]; CORE
				}
			}
#undef M2
#undef M1
#undef CORE
		}
#endif

		a2 = a2 / a3;
		range = roundf(a2);

		if (range) {
			int div = quantval[i], coef1 = coef[i], add;
			int dh, dl, d0 = (div - 1) >> 1, d1 = div >> 1;
			int a0 = (coef1 + (coef1 < 0 ? -d1 : d1)) / div * div;

			dh = a0 + (a0 < 0 ? d1 : d0);
			dl = a0 - (a0 > 0 ? d1 : d0);

			add = coef1 - range;
			if (add > dh) add = dh;
			if (add < dl) add = dl;
			coef[i] = add;
			need_refresh |= add ^ coef1;
		}
	}
}

//#define PRECISE_PROGRESS

#ifndef QS_NAME
#define QS_NAME do_quantsmooth
#endif
JPEGQS_ATTR int QS_NAME(j_decompress_ptr srcinfo, jvirt_barray_ptr *coef_arrays, jpegqs_control_t *opts) {
	JDIMENSION comp_width, comp_height, blk_y;
	int i, ci, stride, iter, stride1 = 0, need_downsample = 0;
	jpeg_component_info *compptr;
	JQUANT_TBL *qtbl; JSAMPLE *image, *image1 = NULL, *image2 = NULL;
	int num_iter = opts->niter, old_threads = -1;
	int prog_next = 0, prog_max = 0, prog_thr = 0, prog_prec = opts->progprec;
#ifdef PRECISE_PROGRESS
	volatile int stop = 0;
#else
	int stop = 0;
#endif
	jvirt_barray_ptr coef_up[2] = { NULL, NULL };
	float **tables = NULL;

#ifdef WITH_LOG
	int64_t time = 0;

	if (opts->flags & JPEGQS_INFO_COMP1)
	for (ci = 0; ci < srcinfo->num_components; ci++) {
		compptr = srcinfo->comp_info + ci;
		i = compptr->quant_tbl_no;
		logfmt("component[%i] : table %i, samp %ix%i\n", ci, i,
				compptr->h_samp_factor, compptr->v_samp_factor);
	}

	if (opts->flags & JPEGQS_INFO_QUANT)
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

	if (opts->flags & JPEGQS_INFO_TIME) time = get_time_usec();
#endif

	compptr = srcinfo->comp_info;
	if (opts->flags & (JPEGQS_JOINT_YUV | JPEGQS_UPSAMPLE_UV) &&
			srcinfo->jpeg_color_space == JCS_YCbCr &&
			!((compptr[1].h_samp_factor - 1) | (compptr[1].v_samp_factor - 1) |
			(compptr[2].h_samp_factor - 1) | (compptr[2].v_samp_factor - 1))) {
		need_downsample = 1;
	}

	if (num_iter < 0) num_iter = 0;
	if (num_iter > JPEGQS_ITER_MAX) num_iter = JPEGQS_ITER_MAX;

	if (num_iter <= 0 && !(opts->flags & JPEGQS_UPSAMPLE_UV && need_downsample)) return 0;

	range_limit_init();
	if (!(opts->flags & JPEGQS_LOW_QUALITY)) {
		tables = quantsmooth_init(opts->flags);
		if (!tables) return 0;
	}

	(void)old_threads;
#ifdef _OPENMP
	if (opts->threads >= 0) {
		old_threads = omp_get_max_threads();
		omp_set_num_threads(opts->threads ? opts->threads : omp_get_num_procs());
	}
#endif

	if (opts->progress) {
		for (ci = 0; ci < srcinfo->num_components; ci++) {
			compptr = srcinfo->comp_info + ci;
			prog_max += compptr->height_in_blocks * compptr->v_samp_factor * num_iter;
		}
		if (prog_prec == 0) prog_prec = 20;
		if (prog_prec < 0) prog_prec = prog_max;
		prog_thr = (unsigned)(prog_max + prog_prec - 1) / (unsigned)prog_prec;
	}

	for (ci = 0; ci < srcinfo->num_components; ci++) {
		int extra_refresh = 0, num_iter2 = num_iter;
		int prog_cur = prog_next, prog_inc;
		compptr = srcinfo->comp_info + ci;
		comp_width = compptr->width_in_blocks;
		comp_height = compptr->height_in_blocks;
		prog_inc = compptr->v_samp_factor;
		prog_next += comp_height * prog_inc * num_iter;
		if (!(qtbl = compptr->quant_table)) continue;

		if (image1 || (!ci && need_downsample)) extra_refresh = 1;

		// skip if already processed
		{
			int val = 0;
			for (i = 0; i < DCTSIZE2; i++) val |= qtbl->quantval[i];
			if (val <= 1) num_iter2 = 0;
		}

		if (num_iter2 + extra_refresh == 0) continue;
		image = NULL;
		if (!stop) {
			// keeping block pointers aligned
			stride = comp_width * DCTSIZE + 8;
			image = (JSAMPROW)malloc(((comp_height * DCTSIZE + 2) * stride + 8) * sizeof(JSAMPLE));
		}
		if (!image) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
			for (blk_y = 0; blk_y < comp_height; blk_y++) {
				JDIMENSION blk_x;
				JBLOCKARRAY buffer = (*srcinfo->mem->access_virt_barray)
						((j_common_ptr)srcinfo, coef_arrays[ci], blk_y, 1, TRUE);

				for (blk_x = 0; blk_x < comp_width; blk_x++) {
					JCOEFPTR coef = buffer[0][blk_x]; int i;
					for (i = 0; i < DCTSIZE2; i++) coef[i] *= qtbl->quantval[i];
				}
			}
			continue;
		}
		image += 7;

#ifdef WITH_LOG
		if (opts->flags & JPEGQS_INFO_COMP2)
			logfmt("component[%i] : size %ix%i\n", ci, comp_width, comp_height);
#endif
#define IMAGEPTR (blk_y * DCTSIZE + 1) * stride + blk_x * DCTSIZE + 1

#ifdef USE_JSIMD
		JSAMPROW output_buf[DCTSIZE];
		for (i = 0; i < DCTSIZE; i++) output_buf[i] = image + i * stride;
#endif

		for (iter = 0; iter < num_iter2 + extra_refresh; iter++) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
			for (blk_y = 0; blk_y < comp_height; blk_y++) {
				JDIMENSION blk_x;
				JBLOCKARRAY buffer = (*srcinfo->mem->access_virt_barray)
						((j_common_ptr)srcinfo, coef_arrays[ci], blk_y, 1, TRUE);

				for (blk_x = 0; blk_x < comp_width; blk_x++) {
					JCOEFPTR coef = buffer[0][blk_x]; int i;
					if (!iter)
						for (i = 0; i < DCTSIZE2; i++) coef[i] *= qtbl->quantval[i];
#ifdef USE_JSIMD
					int output_col = IMAGEPTR;
#endif
					idct_islow(coef, image + IMAGEPTR, stride);
				}
			}

			{
				int y, w = comp_width * DCTSIZE, h = comp_height * DCTSIZE;
				for (y = 1; y < h + 1; y++) {
					image[y * stride] = image[y * stride + 1];
					image[y * stride + w + 1] = image[y * stride + w];
				}
				memcpy(image, image + stride, stride * sizeof(JSAMPLE));
				memcpy(image + (h + 1) * stride, image + h * stride, stride * sizeof(JSAMPLE));
			}

			if (iter == num_iter2) break;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
			for (blk_y = 0; blk_y < comp_height; blk_y++) {
				JDIMENSION blk_x;
				JBLOCKARRAY buffer = (*srcinfo->mem->access_virt_barray)
						((j_common_ptr)srcinfo, coef_arrays[ci], blk_y, 1, TRUE);

#ifdef PRECISE_PROGRESS
				if (stop) continue;
#endif
				for (blk_x = 0; blk_x < comp_width; blk_x++) {
					JSAMPLE *p2 = image2 && opts->flags & JPEGQS_JOINT_YUV ? image2 + IMAGEPTR : NULL;
					JCOEFPTR coef = buffer[0][blk_x];
					quantsmooth_block(coef, qtbl->quantval, image + IMAGEPTR, p2, stride, opts->flags, tables);
				}
#ifdef PRECISE_PROGRESS
				if (opts->progress) {
					int cur = __sync_add_and_fetch(&prog_cur, prog_inc);
					if (cur >= prog_thr && omp_get_thread_num() == 0) {
						cur = (int64_t)prog_prec * cur / prog_max;
						prog_thr = ((int64_t)(cur + 1) * prog_max + prog_prec - 1) / prog_prec;
						stop = opts->progress(opts->userdata, cur, prog_prec);
					}
				}
#endif
			}

#ifdef PRECISE_PROGRESS
			if (stop) break;
#else
			if (opts->progress) {
				int cur = prog_cur += comp_height * prog_inc;
				if (cur >= prog_thr) {
					cur = (int64_t)prog_prec * cur / prog_max;
					prog_thr = ((int64_t)(cur + 1) * prog_max + prog_prec - 1) / prog_prec;
					stop = opts->progress(opts->userdata, cur, prog_prec);
				}
				if (stop) break;
			}
#endif
		} // iter

		if (!stop && image1) {
			JSAMPLE *mem; int st, w1, h1, ws, hs;
			compptr = srcinfo->comp_info;
			ws = compptr[0].h_samp_factor;
			hs = compptr[0].v_samp_factor;
			w1 = (srcinfo->image_width + ws - 1) / ws;
			h1 = (srcinfo->image_height + hs - 1) / hs;
			comp_width = compptr[0].width_in_blocks;
			comp_height = compptr[0].height_in_blocks;

			coef_up[ci - 1] = (*srcinfo->mem->request_virt_barray)
					((j_common_ptr)srcinfo, JPOOL_IMAGE, FALSE, comp_width, comp_height, 1);
			(*srcinfo->mem->realize_virt_arrays) ((j_common_ptr)srcinfo);

#ifdef _OPENMP
			// need to suppress JERR_BAD_VIRTUAL_ACCESS
			for (blk_y = 0; blk_y < comp_height; blk_y++) {
				(*srcinfo->mem->access_virt_barray)
						((j_common_ptr)srcinfo, coef_up[ci - 1], blk_y, 1, TRUE);
			}
#endif

			st = comp_width * DCTSIZE;
			mem = (JSAMPLE*)malloc(comp_height * DCTSIZE * st * sizeof(JSAMPLE));
			if (mem) {
				int y, ww = comp_width * DCTSIZE, hh = comp_height * DCTSIZE;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
				for (y = 0; y < h1; y++) {
					int x, xx, yy, a, h2 = hh - y * hs;
					h2 = h2 < hs ? h2 : hs;
					for (x = 0; x < w1; x++) {
						JSAMPLE *p = mem + y * hs * st + x * ws;
						JSAMPLE *p1 = image1 + (y * hs + 1) * stride1 + x * ws + 1;
						int w2 = ww - x * ws; 
						w2 = w2 < ws ? w2 : ws;

						float sumA = 0, sumB = 0, sumAA = 0, sumAB = 0;
						float divN = 1.0f / 16, scale, offset; int a;
#define M1(xx, yy) { \
	float a = image2[(y + yy + 1) * stride + x + xx + 1]; \
	float b = image[(y + yy + 1) * stride + x + xx + 1]; \
	sumA += a; sumAA += a * a; \
	sumB += b; sumAB += a * b; }
#define M2 sumA += sumA; sumB += sumB; \
	sumAA += sumAA; sumAB += sumAB;
						M1(0, 0) M2
						M1(0, -1) M1(-1, 0) M1(1, 0) M1(0, 1) M2
						M1(-1, -1) M1(1, -1) M1(-1, 1) M1(1, 1)
#undef M2
#undef M1
						scale = sumAA - sumA * divN * sumA;
						if (scale != 0.0f) scale = (sumAB - sumA * divN * sumB) / scale;
						scale = scale < -16.0f ? -16.0f : scale;
						scale = scale > 16.0f ? 16.0f : scale;
						// offset = (sumB - scale * sumA) * divN;
						a = image2[(y + 1) * stride + x + 1];
						offset = image[(y + 1) * stride + x + 1] - a * scale;

						for (yy = 0; yy < h2; yy++)
						for (xx = 0; xx < w2; xx++) {
							a = p1[yy * stride1 + xx] * scale + offset + 0.5f;
							p[yy * st + xx] = a < 0 ? 0 : a > MAXJSAMPLE ? MAXJSAMPLE : a;
						}
					}
					a = mem[y * st + w1 * ws - 1];
					for (x = w1 * ws; x < ww; x++) mem[y * st + x] = a;
				}
				for (y = h1 * hs; y < hh; y++)
					memcpy(mem + y * st, mem + (h1 * hs - 1) * st, st * sizeof(JSAMPLE));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
				for (blk_y = 0; blk_y < comp_height; blk_y++) {
					JDIMENSION blk_x;
					JBLOCKARRAY buffer = (*srcinfo->mem->access_virt_barray)
							((j_common_ptr)srcinfo, coef_up[ci - 1], blk_y, 1, TRUE);

					for (blk_x = 0; blk_x < comp_width; blk_x++) {
						float ALIGN(32) buf[DCTSIZE2]; int x, y, n = DCTSIZE;
						JSAMPLE *p = mem + blk_y * n * st + blk_x * n;
						JCOEFPTR coef = buffer[0][blk_x];
						for (y = 0; y < n; y++)
						for (x = 0; x < n; x++)
							buf[y * n + x] = p[y * st + x] - CENTERJSAMPLE;
						fdct_float(buf, buf);
						for (x = 0; x < n * n; x++) coef[x] = roundf(buf[x]);
					}
				}
				free(mem);
			}
		} else if (!stop && !ci && need_downsample) do {
			// make downsampled copy of Y component
			int y, w, h, w1, h1, st, ws, hs;

			ws = compptr[0].h_samp_factor;
			hs = compptr[0].v_samp_factor;
			if ((ws - 1) | (hs - 1)) {
				if (opts->flags & JPEGQS_UPSAMPLE_UV) { image1 = image; stride1 = stride; }
			} else { image2 = image; break; }
			w = compptr[1].width_in_blocks * DCTSIZE;
			h = compptr[1].height_in_blocks * DCTSIZE;
			st = w + 8;
			image2 = (JSAMPLE*)malloc(((h + 2) * st + 8) * sizeof(JSAMPLE));
			if (!image2) break;
			image2 += 7;

			w1 = (comp_width * DCTSIZE + ws - 1) / ws;
			h1 = (comp_height * DCTSIZE + hs - 1) / hs;

			// faster case for 4:2:0
			if (1 && !((ws - 2) | (hs - 2))) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
				for (y = 0; y < h1; y++) {
					int x;
					for (x = 0; x < w1; x++) {
						JSAMPLE *p = image + (y * 2 + 1) * stride + x * 2 + 1;
						int a = p[0] + p[1] + p[stride] + p[stride + 1];
						image2[(y + 1) * st + x + 1] = (a + 2) >> 2;
					}
				}
			} else {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
				for (y = 0; y < h1; y++) {
					int x, h2 = comp_height * DCTSIZE - y * hs;
					h2 = h2 < hs ? h2 : hs;
					for (x = 0; x < w1; x++) {
						JSAMPLE *p = image + (y * hs + 1) * stride + x * ws + 1;
						int xx, yy, sum = 0, w2 = comp_width * DCTSIZE - x * ws, div; 
						w2 = w2 < ws ? w2 : ws; div = w2 * h2;

						for (yy = 0; yy < h2; yy++)
						for (xx = 0; xx < w2; xx++) sum += p[yy * stride + xx];
						image2[(y + 1) * st + x + 1] = (sum + div / 2) / div;
					}
				}
			}

			for (y = 1; y < h1 + 1; y++) {
				int x; JSAMPLE a = image2[y * st + w1];
				image2[y * st] = image2[y * st + 1];
				for (x = w1 + 1; x < w + 2; x++)
					image2[y * st + x] = a;
			}
			memcpy(image2, image2 + st, st * sizeof(JSAMPLE));
			for (y = h1 + 1; y < h + 2; y++)
				memcpy(image2 + y * st, image2 + h1 * st, st * sizeof(JSAMPLE));

		} while (0);
#undef IMAGEPTR
		if (image != image1 && image != image2) free(image - 7);
	}

#ifdef WITH_LOG
	if (!stop && opts->flags & JPEGQS_INFO_TIME) {
		time = get_time_usec() - time;
		logfmt("quantsmooth: %.3fms\n", time * 0.001);
	}
#endif

#ifdef _OPENMP
	if (old_threads > 0) omp_set_num_threads(old_threads);
#endif

	if (tables) free(tables);

	if (image2 != image1 && image2) free(image2 - 7);
	if (image1) free(image1 - 7);
	if (stop) image1 = NULL;
	if (image1) {
		srcinfo->max_h_samp_factor = 1;
		srcinfo->max_v_samp_factor = 1;
		compptr = srcinfo->comp_info;
		compptr[0].h_samp_factor = 1;
		compptr[0].v_samp_factor = 1;
		comp_width = compptr[0].width_in_blocks;
		comp_height = compptr[0].height_in_blocks;
#define M1(i) coef_arrays[i] = coef_up[i - 1]; \
	compptr[i].width_in_blocks = comp_width; \
	compptr[i].height_in_blocks = comp_height;
		M1(1) M1(2)
#undef M1
	}

	for (ci = 0; ci < NUM_QUANT_TBLS; ci++) {
		qtbl = srcinfo->quant_tbl_ptrs[ci];
		if (qtbl) for (i = 0; i < DCTSIZE2; i++) qtbl->quantval[i] = 1;
	}

	for (ci = 0; ci < srcinfo->num_components; ci++) {
		qtbl = srcinfo->comp_info[ci].quant_table;
		if (qtbl) for (i = 0; i < DCTSIZE2; i++) qtbl->quantval[i] = 1;
	}

#ifndef TRANSCODE_ONLY
	if (!(opts->flags & JPEGQS_TRANSCODE)) {
		// things needed for jpeg_read_scanlines() to work correctly
		if (image1) {
#ifdef LIBJPEG_TURBO_VERSION
			srcinfo->master->last_MCU_col[1] = srcinfo->master->last_MCU_col[0];
			srcinfo->master->last_MCU_col[2] = srcinfo->master->last_MCU_col[0];
#endif
			jinit_color_deconverter(srcinfo);
			jinit_upsampler(srcinfo);
			jinit_d_main_controller(srcinfo, FALSE);
			srcinfo->input_iMCU_row = (srcinfo->output_height + DCTSIZE - 1) / DCTSIZE;
		}
		jinit_inverse_dct(srcinfo);
	}
#endif
	return stop;
}

#if !defined(TRANSCODE_ONLY) && !defined(NO_HELPERS)
JPEGQS_ATTR
boolean jpegqs_start_decompress(j_decompress_ptr cinfo, jpegqs_control_t *opts) {
	boolean ret; int use_jpeqqs = opts->niter > 0 || opts->flags & JPEGQS_UPSAMPLE_UV;
	if (use_jpeqqs) cinfo->buffered_image = TRUE;
	ret = jpeg_start_decompress(cinfo);
	if (use_jpeqqs) {
		while (!jpeg_input_complete(cinfo)) {
			jpeg_start_output(cinfo, cinfo->input_scan_number);
			jpeg_finish_output(cinfo);
		}
		do_quantsmooth(cinfo, jpeg_read_coefficients(cinfo), opts);
		jpeg_start_output(cinfo, cinfo->input_scan_number);
	}
	return ret;
}

JPEGQS_ATTR
boolean jpegqs_finish_decompress(j_decompress_ptr cinfo) {
	if ((cinfo->global_state == DSTATE_SCANNING ||
			cinfo->global_state == DSTATE_RAW_OK) && cinfo->buffered_image) {
		jpeg_finish_output(cinfo);
	}
	return jpeg_finish_decompress(cinfo);
}
#endif
