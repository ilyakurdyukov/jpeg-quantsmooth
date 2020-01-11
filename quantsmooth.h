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
#include <time.h>
#include <sys/time.h>
static inline int64_t get_time_usec() {
	struct timeval time;
	gettimeofday(&time, NULL);
	return time.tv_sec * (int64_t)1000000 + time.tv_usec;
}
#endif

#if defined(__SSE4_1__)
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

#define ALIGN(n) __attribute__((aligned(n)))

#include "idct.h"

static float ALIGN(32) quantsmooth_tables[DCTSIZE2][3][DCTSIZE2];

static void quantsmooth_init() {
	int i; JCOEF coef[DCTSIZE2];

	range_limit_init();

	for (i = 0; i < DCTSIZE2; i++) {
		float (*tab)[DCTSIZE2] = quantsmooth_tables[i];
		int x, y, p0, p1;
		memset(coef, 0, sizeof(coef)); coef[i] = 1;
		idct_fslow(coef, tab[0]);

#define M1(xx, yy, j) \
	p0 = y*DCTSIZE+x; p1 = (yy)*DCTSIZE+(xx); \
	tab[j][p0] = tab[0][p0] - tab[0][p1];
		for (y = 0; y < DCTSIZE; y++) {
			for (x = 0; x < DCTSIZE-1; x++) { M1(x+1,y,1) }
			tab[1][y*DCTSIZE+x] = 0;
		}
		for (y = 0; y < DCTSIZE-1; y++)
		for (x = 0; x < DCTSIZE; x++) { M1(x,y+1,2) }
#undef M1
		for (x = 0; x < DCTSIZE2; x++) tab[0][x] *= 2.0f;
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

static void quantsmooth_block(JCOEFPTR coef, UINT16 *quantval, JSAMPROW image, int stride) {

	int iter, k, x, y;
	JSAMPLE ALIGN(32) buf[DCTSIZE2];
	JCOEF coef2[DCTSIZE2];
#ifdef USE_JSIMD
	JSAMPROW output_buf[8] = { buf+8*0, buf+8*1, buf+8*2, buf+8*3, buf+8*4, buf+8*5, buf+8*6, buf+8*7 };
	int output_col = 0;
#endif

#if 0
	memcpy(coef2, coef, sizeof(coef2));
#else
	// restore from coef
	for (x = 0; x < 64; x++) {
		int div = quantval[x];
#if 0
		int a0 = ((coef[x] + (div >> 1)) / div) * div - (coef[x] < 0 ? div : 0);
#else
		int32_t a0 = coef[x], sign = a0 >> 31;
		a0 = (a0 ^ sign) - sign;
		a0 = ((a0 + (div >> 1)) / div) * div;
		a0 = (a0 ^ sign) - sign;
#endif
		coef2[x] = a0;
	}
#endif

	for (iter = 0; iter < 1; iter++) {
	int flag = 1;

	for (k = DCTSIZE2-1; k > 0; k--) {
		float *tab, a0, a1, a2, a3, halfx;
		int p0, p1, i = jpeg_natural_order[k];
		halfx = quantval[i] * 0.125f * 16;
		tab = quantsmooth_tables[i][0];
		a2 = a3 = 0;
#if 1
		if (flag != 0 && zigzag_refresh[k]) {
			idct_islow(coef, buf, DCTSIZE);
			flag = 0;
		}
#endif

#define M2(a) { float x = halfx-fabsf(a); if (x < 0) x = 0; x *= x; a *= x; a1 *= x; }

#if 1 && defined(USE_AVX2)
#define M1_INIT \
	__m128i v0, v1, v5; \
	__m256 f0, f1, f4, f7 = _mm256_set1_ps(halfx);

#define M1(tab) \
	f4 = _mm256_load_ps(tab); \
	v5 = _mm_abs_epi16(v0); \
	f0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(v5)); \
	f0 = _mm256_sub_ps(f7, _mm256_min_ps(f0, f7)); \
	f0 = _mm256_mul_ps(f0, f0); \
	f1 = _mm256_mul_ps(f0, f4); \
	f0 = _mm256_mul_ps(f0, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v0))); \
	f0 = _mm256_mul_ps(f0, f1); f1 = _mm256_mul_ps(f1, f1); \
	f0 = _mm256_add_ps(_mm256_permute2f128_ps(f0, f1, 0x20), _mm256_permute2f128_ps(f0, f1, 0x31)); \
	f0 = _mm256_hadd_ps(f0, f0); f0 = _mm256_hadd_ps(f0, f0); \
	a2 += _mm256_cvtss_f32(f0); \
	a3 += _mm_cvtss_f32(_mm256_extractf128_ps(f0, 1));

#elif 1 && (defined(USE_SSE4) || defined(USE_SSE2))

#ifdef USE_SSE4
#define M11 \
	f0 = _mm_hadd_ps(_mm_add_ps(f0, f2), _mm_add_ps(f1, f3)); \
	f0 = _mm_hadd_ps(f0, f0);
#else
#define M11 \
	f0 = _mm_add_ps(f0, f2); f1 = _mm_add_ps(f1, f3); \
	f0 = _mm_add_ps(_mm_unpacklo_ps(f0, f1), _mm_unpackhi_ps(f0, f1)); \
	f0 = _mm_add_ps(f0, _mm_shuffle_ps(f0, f0, 0xee));
#endif

#define M1_INIT \
	__m128i v0, v1, v5, v7 = _mm_setzero_si128(); \
	__m128 f0, f1, f2, f3, f4_lo, f4_hi, f7 = _mm_set1_ps(halfx);
#define M1(tab) \
	f4_lo = _mm_load_ps(tab); f4_hi = _mm_load_ps(tab+4); \
	v5 = _mm_abs_epi16(v0); \
	M10(lo, f0, f1, v0) \
	M10(hi, f2, f3, _mm_bsrli_si128(v0, 8)) \
	M11 \
	a2 += _mm_cvtss_f32(f0); \
	a3 += _mm_cvtss_f32(_mm_shuffle_ps(f0, f0, 0x55));

#define M10(lo, f0, f1, v0) \
	f0 = _mm_cvtepi32_ps(_mm_unpack##lo##_epi16(v5, v7)); \
	f0 = _mm_sub_ps(f7, _mm_min_ps(f0, f7)); \
	f0 = _mm_mul_ps(f0, f0); \
	f1 = _mm_mul_ps(f0, f4_##lo); \
	f0 = _mm_mul_ps(f0, _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v0))); \
	f0 = _mm_mul_ps(f0, f1); f1 = _mm_mul_ps(f1, f1);

#endif

#ifdef M1_INIT
		(void)p1; (void)a1; // suppress unused-variable warnings

		{
			M1_INIT

			for (y = 0; y < DCTSIZE; y++) {
				v0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&buf[y*DCTSIZE]));
				v0 = _mm_sub_epi16(v0, _mm_bsrli_si128(v0, 2));
				M1(tab+64+y*DCTSIZE)
			}

			v0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&buf[0]));
			for (y = 0; y < DCTSIZE-1; y++) {
				v1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&buf[y*DCTSIZE+DCTSIZE]));
				v0 = _mm_sub_epi16(v0, v1);
				M1(tab+64*2+y*DCTSIZE)
				v0 = v1;
			}

#define M5(y, yy) \
	v0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&buf[y*DCTSIZE])); \
	v1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((void*)&image[(yy)*stride])); \
	v0 = _mm_sub_epi16(v0, v1); \
	M1(tab+y*DCTSIZE)
			M5(0, -1) M5(7, 8)
#undef M5

			unsigned short tmp0[16]; float tmp1[16];
#define M5(x, xx, yy, i) p0 = y*DCTSIZE+x; \
	tmp0[i] = buf[p0] - image[(yy)*stride+(xx)]; tmp1[i] = tab[p0];
			for (y = 0; y < DCTSIZE; y++) { M5(0, -1, y, y) M5(7, 8, y, y+8) }
#undef M5
			v0 = _mm_load_si128((void*)&tmp0[0]);
			M1(tmp1)
			v0 = _mm_load_si128((void*)&tmp0[8]);
			M1(tmp1+8)
		}
#undef M1_INIT
#undef M1
#ifdef M10
#undef M10
#undef M11
#endif
#else
#define M1(xx, yy, i) \
	p0 = y*DCTSIZE+x; p1 = (yy)*DCTSIZE+(xx); \
	a0 = buf[p0] - buf[p1]; a1 = tab[i*64+p0]; \
	M2(a0) a2 += a0 * a1; a3 += a1 * a1;
		for (y = 0; y < DCTSIZE; y++)
		for (x = 0; x < DCTSIZE-1; x++) { M1(x+1,y,1) }
		for (y = 0; y < DCTSIZE-1; y++)
		for (x = 0; x < DCTSIZE; x++) { M1(x,y+1,2) }
#undef M1
#define M1(xx, yy) \
	p0 = y*DCTSIZE+x; \
	a0 = buf[p0] - image[(yy)*stride+(xx)]; a1 = tab[p0]; \
	M2(a0) a2 += a0 * a1; a3 += a1 * a1;
		y = 0; for (x = 0; x < DCTSIZE; x++) { M1(x,y-1) }
		y = DCTSIZE-1; for (x = 0; x < DCTSIZE; x++) { M1(x,y+1) }
		x = 0; for (y = 0; y < DCTSIZE; y++) { M1(x-1,y) }
		x = DCTSIZE-1; for (y = 0; y < DCTSIZE; y++) { M1(x+1,y) }
#undef M1
#endif

		a0 = a2 / a3;
#undef M2

		{
			int div = quantval[i];
			int add = coef[i] - coef2[i];
			int dh, dl, d0 = (div-1) >> 1, d1 = div >> 1;
			if (coef2[i] == 0) { dh = d0; dl = -d0; }
			else if (coef2[i] < 0) { dh = d1; dl = -d0; }
			else { dh = d0; dl = -d1; }

			add -= roundf(a0);
			if (add > dh) add = dh;
			if (add < dl) add = dl;
			add = coef2[i] + add;
			flag += add != coef[i];
			coef[i] = add;
		}
	}

	} // iter
}

static void quantsmooth_transform(j_decompress_ptr srcinfo, jvirt_barray_ptr *src_coef_arrays, int flags) {
	JDIMENSION comp_width, comp_height, blk_y;
	int ci, qtblno;
	JSAMPROW image; int stride;
	jpeg_component_info *compptr;
	JQUANT_TBL *qtbl;

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

		stride = comp_width * DCTSIZE + 2;
		image = (JSAMPROW)malloc((comp_height * DCTSIZE + 2) * stride * sizeof(JSAMPLE));
		if (!image) continue;
#ifdef WITH_LOG
		if (flags & 4) logfmt("component[%i] : size %ix%i\n", ci, comp_width, comp_height);
#endif
#define IMAGEPTR &image[((blk_y + offset_y) * DCTSIZE + 1) * stride + blk_x * DCTSIZE + 1]

#ifdef USE_JSIMD
		JSAMPROW output_buf[8] = {
			image+stride*0, image+stride*1, image+stride*2, image+stride*3,
			image+stride*4, image+stride*5, image+stride*6, image+stride*7 };
#endif

		for (int iter = 0; iter < 3; iter++) {
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
						quantsmooth_block(coef, qtbl->quantval, IMAGEPTR, stride);
					}
				}
			}
		} // iter
#undef IMAGEPTR
		free(image);
	}
}

static void do_quantsmooth(j_decompress_ptr srcinfo, jvirt_barray_ptr *coef_arrays, int flags) {
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
		quantsmooth_init();
		quantsmooth_transform(srcinfo, coef_arrays, flags);
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
