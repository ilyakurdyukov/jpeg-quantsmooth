/*
 * Copyright (C) 2016-2020 Ilya Kurdyukov
 *
 * This file is part of jpeg quantsmooth (mini version)
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

/*
 * A minimal JPEG Quant Smooth version for experiments.
 * Without SIMD optimizations and own DCT transforms, but has OpenMP pragmas.
 * Uses DCT transforms from libjpeg.
 *
 * Build with: make SRCNAME=jpegqs-mini.c
 */

#define JPEG_INTERNALS
#ifdef WITH_JPEGSRC
#include "jinclude.h"
#include "jpeglib.h"
#include "jdct.h"
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "jpeglib.h"
#define NO_JPEGTRAN
#ifndef WITH_SIMD
#define DCTELEM int
#else
#define DCTELEM short
#endif
EXTERN(void) jpeg_idct_islow(j_decompress_ptr,
		jpeg_component_info*, JCOEFPTR, JSAMPARRAY, JDIMENSION);
EXTERN(void) jpeg_fdct_islow(DCTELEM*);
#define MEMCOPY memcpy
#endif

#include <math.h>

// X = cos(pi * n / 16) * sqrt(2)
#define A 1.000000000 // 4
#define B 1.387039845 // 1
#define C 1.175875602 // 3
#define D 0.785694958 // 5
#define E 0.275899379 // 7
#define F 1.306562965 // 2
#define G 0.541196100 // 6
static const float idct_fcoef[DCTSIZE2] = {
 A,  A,  A,  A,  A,  A,  A,  A, 
 B,  C,  D,  E, -E, -D, -C, -B, 
 F,  G, -G, -F, -F, -G,  G,  F, 
 C, -E, -B, -D,  D,  B,  E, -C, 
 A, -A, -A,  A,  A, -A, -A,  A, 
 D, -B,  E,  C, -C, -E,  B, -D, 
 G, -F,  F, -G, -G,  F, -F,  G, 
 E, -D,  C, -B,  B, -C,  D, -E };
#undef A
#undef B
#undef C
#undef D
#undef E
#undef F
#undef G

#define IDCT_ISLOW(col) jpeg_idct_islow(srcinfo, compptr, coef, output_buf, col)

static float dct_diff[DCTSIZE2][DCTSIZE2];

static const char zigzag_refresh[DCTSIZE2] = {
	1, 0, 1, 0, 1, 0, 1, 0,
	1, 0, 0, 0, 0, 0, 0, 1,
	0, 0, 0, 0, 0, 0, 0, 0,
	1, 0, 0, 0, 0, 0, 0, 1,
	0, 0, 0, 0, 0, 0, 0, 0,
	1, 0, 0, 0, 0, 0, 0, 1,
	0, 0, 0, 0, 0, 0, 0, 0,
	1, 0, 1, 0, 1, 0, 1, 1 };

#define printf(...) fprintf(stderr, __VA_ARGS__)

#define NUM_ITER 3
#define DIAGONALS
#define JOINT_YUV
#define UPSAMPLE_UV
//#define LOW_QUALITY
//#define NO_REBALANCE
//#define NO_REBALANCE_UV

static void quantsmooth_block(j_decompress_ptr srcinfo, jpeg_component_info *compptr,
		JCOEFPTR coef, UINT16 *quantval, JSAMPLE *image, JSAMPLE *image2, int stride, int luma) {
	int k, x, y, need_refresh = 1, n = DCTSIZE;
	JSAMPLE buf[DCTSIZE2], *output_buf[DCTSIZE];
#ifdef DIAGONALS
	float bcoef = 4.0;
#else
	float bcoef = 2.0;
#endif

	for (k = 0; k < n; k++) output_buf[k] = buf + k * n;

	if (image2) {
#ifdef JOINT_YUV
		DCTELEM buf[DCTSIZE2];
		for (y = 0; y < n; y++)
		for (x = 0; x < n; x++) {
			float sumA = 0, sumB = 0, sumAA = 0, sumAB = 0;
			float divN = 1.0f / 16, scale, offset; int a;
#define M1(xx, yy) { \
	float a = image2[(y + yy) * stride + x + xx]; \
	float b = image[(y + yy) * stride + x + xx]; \
	sumA += a; sumAA += a * a; \
	sumB += b; sumAB += a * b; }
#define M2(n) sumA *= n; sumB *= n; sumAA *= n; sumAB *= n;
			M1(0, 0) M2(2)
			M1(0, -1) M1(-1, 0) M1(1, 0) M1(0, 1) M2(2)
			M1(-1, -1) M1(1, -1) M1(-1, 1) M1(1, 1)
#undef M2
#undef M1
			scale = sumAA - sumA * divN * sumA;
			if (scale != 0.0f) scale = (sumAB - sumA * divN * sumB) / scale;
			scale = fminf(fmaxf(scale, -16.0f), 16.0f);
			offset = (sumB - scale * sumA) * divN;

			a = image2[y * stride + x] * scale + offset + 0.5f;
			a = a < 0 ? 0 : a > MAXJSAMPLE ? MAXJSAMPLE : a;
			buf[y * n + x] = a - CENTERJSAMPLE;
		}
		jpeg_fdct_islow(buf);
		for (x = 0; x < n * n; x++) {
			int div = quantval[x], coef1 = coef[x], add;
			int dh, dl, d0 = (div - 1) >> 1, d1 = div >> 1;
			int a0 = (coef1 + (coef1 < 0 ? -d1 : d1)) / div * div;

			dh = a0 + (a0 < 0 ? d1 : d0);
			dl = a0 - (a0 > 0 ? d1 : d0);

			add = (buf[x] + 4) >> 3;
			if (add > dh) add = dh;
			if (add < dl) add = dl;
			coef[x] = add;
		}
#endif
	}

#ifdef LOW_QUALITY
	(void)bcoef; (void)output_buf; (void)need_refresh;
	(void)srcinfo; (void)compptr; (void)zigzag_refresh;
	if (!image2) {
		DCTELEM buf[DCTSIZE2];
		float range = 0, c0 = 2, c1 = c0 * sqrtf(0.5f);
		{
			int sum = 0;
			for (x = 1; x < n * n; x++) {
				int a = coef[x]; a = a < 0 ? -a : a;
				range += quantval[x] * a; sum += a;
			}
			if (sum) range *= 4.0f / sum;
			if (range > CENTERJSAMPLE) range = CENTERJSAMPLE;
		}

		for (y = 0; y < n; y++)
		for (x = 0; x < n; x++) {
#define CORE(i, x, y) t0 = a - image[(y) * stride + (x)]; \
	t = range - fabsf(t0); if (t < 0) { t = 0; } t *= t; aw = c##i * t; \
	a0 += t0 * t * aw; an += aw * aw;
			int a = image[(y)*stride+(x)];
			float a0 = 0, an = 0, aw, t, t0;
			CORE(1, x-1, y-1) CORE(0, x, y-1) CORE(1, x+1, y-1)
			CORE(0, x-1, y)                   CORE(0, x+1, y)
			CORE(1, x-1, y+1) CORE(0, x, y+1) CORE(1, x+1, y+1)
#undef CORE
			if (an != 0.0f) a -= (int)roundf(a0 / an);
			buf[y * n + x] = a - CENTERJSAMPLE;
		}
		jpeg_fdct_islow(buf);
		for (x = 0; x < n * n; x++) {
			int div = quantval[x], coef1 = coef[x], add;
			int dh, dl, d0 = (div - 1) >> 1, d1 = div >> 1;
			int a0 = (coef1 + (coef1 < 0 ? -d1 : d1)) / div * div;

			dh = a0 + (a0 < 0 ? d1 : d0);
			dl = a0 - (a0 > 0 ? d1 : d0);

			add = (buf[x] + 4) >> 3;
			if (add > dh) add = dh;
			if (add < dl) add = dl;
			coef[x] = add;
		}
	}
#else
	for (k = DCTSIZE2-1; k > 0; k--) {
		int p0, p1, i = jpeg_natural_order[k];
		float *tab = dct_diff[i], a0, a1, t, a2 = 0, a3 = 0;
		int range = quantval[i] * 2;
		if (need_refresh && zigzag_refresh[i]) { IDCT_ISLOW(0); need_refresh = 0; }

#define CORE t = (float)range - fabsf(a0); \
	if (t < 0) { t = 0; } t *= t; a0 *= t; a1 *= t; a2 += a0 * a1; a3 += a1 * a1;
#define M1(a, b) \
	for (y = 0; y < n - 1 + a; y++) \
	for (x = 0; x < n - 1 + b; x++) { \
	p0 = y * n + x; p1 = (y + b) * n + x + a; \
	a0 = buf[p0] - buf[p1]; a1 = tab[p0] - tab[p1]; CORE }
#define M2(i, a, b) for (i = 0; i < n; i++) { p0 = y * n + x; \
	a0 = buf[p0] - image[(b) * stride + a]; a1 = tab[p0] * bcoef; CORE }

		if (i & (n - 1)) M1(1, 0)
		y = 0; M2(x, x, y - 1) y = n - 1; M2(x, x, y + 1)
		x = 0; M2(y, x - 1, y) x = n - 1; M2(y, x + 1, y)
		if (i > (n - 1)) M1(0, 1)

#ifdef DIAGONALS
		for (y = 0; y < 7; y++)
		for (x = 0; x < 7; x++) {
			p0 = y * n + x; p1 = (y + 1) * n + x;
			a0 = buf[p0] - buf[p1+1]; a1 = tab[p0] - tab[p1+1]; CORE
			a0 = buf[p0+1] - buf[p1]; a1 = tab[p0+1] - tab[p1]; CORE
		}
#endif
#undef M2
#undef M1
#undef CORE

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
#endif

	(void)luma;
#ifndef NO_REBALANCE
#ifdef NO_REBALANCE_UV
	if (luma)
#endif
	{
		JCOEF buf[DCTSIZE2];
		int64_t m0 = 0, m1 = 0;
		for (k = 1; k < DCTSIZE2; k++) {
			int div = quantval[k], coef1 = coef[k], d1 = div >> 1;
			int a0 = (coef1 + (coef1 < 0 ? -d1 : d1)) / div * div;
			buf[k] = a0;
			m0 += coef1 * a0; m1 += a0 * a0;
		}
		if (m1 > m0) {
			int mul = ((m1 << 13) + (m0 >> 1)) / m0;
			for (k = 1; k < DCTSIZE2; k++) {
				int div = quantval[k], coef1 = coef[k], add;
				int dh, dl, d0 = (div - 1) >> 1, d1 = div >> 1;
				int a0 = buf[k];

				dh = a0 + (a0 < 0 ? d1 : d0);
				dl = a0 - (a0 > 0 ? d1 : d0);

				add = (coef1 * mul + 0x1000) >> 13;
				if (add > dh) add = dh;
				if (add < dl) add = dl;
				coef[k] = add;
			}
		}
	}
#endif
}

static void do_quantsmooth(j_decompress_ptr srcinfo, jvirt_barray_ptr *src_coef_arrays) {
	JDIMENSION comp_width, comp_height, blk_y;
	int i, ci, stride, iter, stride1 = 0, need_downsample = 0;
	jpeg_component_info *compptr; JQUANT_TBL *qtbl;
	JSAMPLE *image, *image1 = NULL, *image2 = NULL;
	JSAMPLE *output_buf[DCTSIZE], range_limit[11 * CENTERJSAMPLE];
	MULTIPLIER dct_table1[DCTSIZE2];

	(void)stride1;

	for (i = 0; i < DCTSIZE2; i++) dct_table1[i] = 1;

/*
// You need this code before calling jpeg_read_coefficients()
// just to initialize the range limit table for jpeg_idct_islow()

	srcinfo.buffered_image = TRUE;
	jpeg_start_decompress(&srcinfo);
	while (!jpeg_input_complete(&srcinfo)) {
		jpeg_start_output(&srcinfo, srcinfo.input_scan_number);
		jpeg_finish_output(&srcinfo);
	}

// Or you can fill out the table by yourself.
*/

	{
		int c = CENTERJSAMPLE, m = c * 2;
		JSAMPLE *t = range_limit;

		for (i = 0; i < m; i++) t[i] = 0;
		t += m;
		srcinfo->sample_range_limit = t;
		for (i = 0; i < m; i++) t[i] = i;
		for (; i < 2 * m + c; i++) t[i] = m - 1;
		for (; i < 4 * m; i++) t[i] = 0;
		for (i = 0; i < c; i++) t[4 * m + i] = i;
	}

#ifndef LOW_QUALITY
	for (i = 0; i < DCTSIZE2; i++) {
		int x, y; float m, *tab = dct_diff[i];
		for (y = 0; y < DCTSIZE; y++) {
			m = idct_fcoef[i / DCTSIZE * DCTSIZE + y] * (1.0f / DCTSIZE);
			for (x = 0; x < DCTSIZE; x++)
				tab[y * DCTSIZE + x] = m * idct_fcoef[i % DCTSIZE * DCTSIZE + x];
		}
	}
#else
	(void)dct_diff; (void)idct_fcoef;
#endif

#if defined(JOINT_YUV) || defined(UPSAMPLE_UV)
	compptr = srcinfo->comp_info;
	if (srcinfo->jpeg_color_space == JCS_YCbCr &&
			!((compptr[1].h_samp_factor - 1) | (compptr[1].v_samp_factor - 1) |
			(compptr[2].h_samp_factor - 1) | (compptr[2].v_samp_factor - 1))) {
		need_downsample = 1;
	}
#endif

	for (ci = 0; ci < srcinfo->num_components; ci++) {
		int extra_refresh = 0;
		compptr = srcinfo->comp_info + ci;
		comp_width = compptr->width_in_blocks;
		comp_height = compptr->height_in_blocks;

		if (!(qtbl = compptr->quant_table)) continue;

		stride = comp_width * DCTSIZE + 2;
		image = (JSAMPLE*)malloc((comp_height * DCTSIZE + 2) * stride * sizeof(JSAMPLE));
		if (!image) continue;

#define IMAGEPTR (blk_y * DCTSIZE + 1) * stride + blk_x * DCTSIZE + 1
		compptr->dct_table = dct_table1;
		for (i = 0; i < DCTSIZE; i++) output_buf[i] = image + i * stride;

		if (image1 || (!ci && need_downsample)) extra_refresh = 1;

		for (iter = 0; iter < NUM_ITER + extra_refresh; iter++) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
			for (blk_y = 0; blk_y < comp_height; blk_y++) {
				JDIMENSION blk_x;
				JBLOCKARRAY buffer = (*srcinfo->mem->access_virt_barray)
						((j_common_ptr)srcinfo, src_coef_arrays[ci], blk_y, 1, TRUE);

				for (blk_x = 0; blk_x < comp_width; blk_x++) {
					JCOEFPTR coef = buffer[0][blk_x]; int i;
					if (!iter)
						for (i = 0; i < DCTSIZE2; i++) coef[i] *= qtbl->quantval[i];
					IDCT_ISLOW(IMAGEPTR);
				}
			}

			{
				int y, w = comp_width * DCTSIZE, h = comp_height * DCTSIZE;
				for (y = 1; y < h + 1; y++) {
					image[y * stride] = image[y * stride + 1];
					image[y * stride + w + 1] = image[y * stride + w];
				}
				MEMCOPY(image, image + stride, stride * sizeof(JSAMPLE));
				MEMCOPY(image + (h + 1) * stride, image + h * stride, stride * sizeof(JSAMPLE));
			}

			if (iter == NUM_ITER) break;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
			for (blk_y = 0; blk_y < comp_height; blk_y++) {
				JDIMENSION blk_x;
				JBLOCKARRAY buffer = (*srcinfo->mem->access_virt_barray)
						((j_common_ptr)srcinfo, src_coef_arrays[ci], blk_y, 1, TRUE);

				for (blk_x = 0; blk_x < comp_width; blk_x++)
					quantsmooth_block(srcinfo, compptr, buffer[0][blk_x],
							qtbl->quantval, image + IMAGEPTR, image2 ? image2 + IMAGEPTR : NULL,
							stride, !ci || srcinfo->jpeg_color_space != JCS_YCbCr);
			}
		} // iter

#ifdef UPSAMPLE_UV
		if (image1) {
			JSAMPLE *mem; int st, w1, h1, ws, hs;
			compptr = srcinfo->comp_info;
			ws = compptr[0].h_samp_factor;
			hs = compptr[0].v_samp_factor;
			w1 = (srcinfo->image_width + ws - 1) / ws;
			h1 = (srcinfo->image_height + hs - 1) / hs;
			comp_width = compptr[0].width_in_blocks;
			comp_height = compptr[0].height_in_blocks;
			src_coef_arrays[ci] = (*srcinfo->mem->request_virt_barray)
					((j_common_ptr)srcinfo, JPOOL_IMAGE, FALSE, comp_width, comp_height, 1);
			(*srcinfo->mem->realize_virt_arrays) ((j_common_ptr)srcinfo);

#ifdef _OPENMP
			// need to suppress JERR_BAD_VIRTUAL_ACCESS
			for (blk_y = 0; blk_y < comp_height; blk_y++) {
				(*srcinfo->mem->access_virt_barray)
						((j_common_ptr)srcinfo, src_coef_arrays[ci], blk_y, 1, TRUE);
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
#define M2(n) sumA *= n; sumB *= n; sumAA *= n; sumAB *= n;
						M1(0, 0) M2(2)
						M1(0, -1) M1(-1, 0) M1(1, 0) M1(0, 1) M2(2)
						M1(-1, -1) M1(1, -1) M1(-1, 1) M1(1, 1)
#undef M2
#undef M1
						scale = sumAA - sumA * divN * sumA;
						if (scale != 0.0f) scale = (sumAB - sumA * divN * sumB) / scale;
						scale = fminf(fmaxf(scale, -16.0f), 16.0f);
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
					MEMCOPY(mem + y * st, mem + (h1 * hs - 1) * st, st * sizeof(JSAMPLE));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
				for (blk_y = 0; blk_y < comp_height; blk_y++) {
					JDIMENSION blk_x;
					JBLOCKARRAY buffer = (*srcinfo->mem->access_virt_barray)
							((j_common_ptr)srcinfo, src_coef_arrays[ci], blk_y, 1, TRUE);

					for (blk_x = 0; blk_x < comp_width; blk_x++) {
						DCTELEM buf[DCTSIZE2]; int x, y, n = DCTSIZE;
						JSAMPLE *p = mem + blk_y * n * st + blk_x * n;
						JCOEFPTR coef = buffer[0][blk_x];
						for (y = 0; y < n; y++)
						for (x = 0; x < n; x++)
							buf[y * n + x] = p[y * st + x] - CENTERJSAMPLE;
						jpeg_fdct_islow(buf);
						for (x = 0; x < n * n; x++) coef[x] = (buf[x] + 4) >> 3;
					}
				}
				free(mem);
			}
		} else
#endif
#if defined(JOINT_YUV) || defined(UPSAMPLE_UV)
		if (!ci && need_downsample) do {
			// make downsampled copy of Y component
			int y, w, h, w1, h1, st, ws, hs;

			ws = compptr[0].h_samp_factor;
			hs = compptr[0].v_samp_factor;
			if ((ws - 1) | (hs - 1)) {
#ifdef UPSAMPLE_UV
				image1 = image; stride1 = stride;
#endif
			} else { image2 = image; break; }
			w = compptr[1].width_in_blocks * DCTSIZE;
			h = compptr[1].height_in_blocks * DCTSIZE;
			st = w + 2;
			image2 = (JSAMPLE*)malloc((h + 2) * st * sizeof(JSAMPLE));
			if (!image2) break;

			w1 = (comp_width * DCTSIZE + ws - 1) / ws;
			h1 = (comp_height * DCTSIZE + hs - 1) / hs;

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

			for (y = 1; y < h1 + 1; y++) {
				int x; JSAMPLE a = image2[y * st + w1];
				image2[y * st] = image2[y * st + 1];
				for (x = w1 + 1; x < w + 2; x++)
					image2[y * st + x] = a;
			}
			MEMCOPY(image2, image2 + st, st * sizeof(JSAMPLE));
			for (y = h1 + 1; y < h + 2; y++)
				MEMCOPY(image2 + y * st, image2 + h1 * st, st * sizeof(JSAMPLE));

		} while (0);
#endif // JOINT_YUV || UPSAMPLE_UV
#undef IMAGEPTR
		if (image != image1 && image != image2) free(image);
	}

	if (image2 != image1 && image2) free(image2);
	if (image1) {
		srcinfo->max_h_samp_factor = 1;
		srcinfo->max_v_samp_factor = 1;
		srcinfo->comp_info[0].h_samp_factor = 1;
		srcinfo->comp_info[0].v_samp_factor = 1;
		free(image1);
	}

	for (ci = 0; ci < NUM_QUANT_TBLS; ci++) {
		qtbl = srcinfo->quant_tbl_ptrs[ci];
		if (qtbl) for (i = 0; i < DCTSIZE2; i++) qtbl->quantval[i] = 1;
	}

	for (ci = 0; ci < srcinfo->num_components; ci++) {
		qtbl = srcinfo->comp_info[ci].quant_table;
		if (qtbl) for (i = 0; i < DCTSIZE2; i++) qtbl->quantval[i] = 1;
	}
}

#ifndef NO_JPEGTRAN
// Macro for inserting quantsmooth into jpegtran code.
#define jpeg_read_coefficients(srcinfo) \
	jpeg_read_coefficients(srcinfo); \
	do_quantsmooth(srcinfo, src_coef_arrays)
#include "jpegtran.c"
#else

#ifdef _WIN32
#define USE_SETMODE
#endif

#ifdef USE_SETMODE
#include <fcntl.h>
#include <io.h>
#endif

// Usage: ./main [-optimize] < input.jpg > output.jpg
int main(int argc, char **argv) {
	struct jpeg_decompress_struct srcinfo;
	struct jpeg_compress_struct dstinfo;
	struct jpeg_error_mgr jerr;
	jvirt_barray_ptr *src_coef_arrays; 
#ifdef USE_SETMODE
	setmode(fileno(stdin), O_BINARY);
	setmode(fileno(stdout), O_BINARY);
#endif
	srcinfo.err = dstinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&srcinfo);
	jpeg_create_compress(&dstinfo);
	jpeg_stdio_src(&srcinfo, stdin);
	jpeg_read_header(&srcinfo, TRUE);
	src_coef_arrays = jpeg_read_coefficients(&srcinfo);

	do_quantsmooth(&srcinfo, src_coef_arrays);

	jpeg_copy_critical_parameters(&srcinfo, &dstinfo);
	if (argc > 1 && !strcmp(argv[1], "-optimize"))
		dstinfo.optimize_coding = TRUE;
	jpeg_stdio_dest(&dstinfo, stdout);
	jpeg_write_coefficients(&dstinfo, src_coef_arrays);
	jpeg_finish_compress(&dstinfo);
	jpeg_destroy_compress(&dstinfo);
	jpeg_finish_decompress(&srcinfo);
	jpeg_destroy_decompress(&srcinfo);
	return jerr.num_warnings ? 2 : 0;
}
#endif
