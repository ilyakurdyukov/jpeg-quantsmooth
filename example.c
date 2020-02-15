/*
 * Copyright (C) 2016-2020 Kurdyukov Ilya
 *
 * This file is part of jpeg quantsmooth (example)
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
 * This is an example of how to process a JPEG image using Quant Smooth
 * and get the RGB pixel data without saving the coefficients back to JPEG.
 * For those who want to make a library or plugin.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <setjmp.h>
#define JPEG_INTERNALS
#include "jpeglib.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "quantsmooth.h"

typedef struct {
	int width, height, bpp, stride; uint8_t *data;
} bitmap_t;

static bitmap_t *bitmap_create(int width, int height, int bpp) {
	bitmap_t *image; int stride = width * bpp;
	if ((unsigned)((width - 1) | (height - 1)) >= 0x8000) return 0;
	image = (bitmap_t*)malloc(sizeof(bitmap_t) + stride * height);
	if (!image) return 0;
	image->width = width;
	image->height = height;
	image->bpp = bpp;
	// BMP needs 4-byte row alignment
	image->stride = (width * bpp + 3) & -4;
	image->data = (uint8_t*)(image + 1);
	return image;
}

static void bitmap_free(bitmap_t *in) {
	if (in) free(in);
}

typedef struct {
	struct jpeg_error_mgr pub;
	jmp_buf setjmp_buffer;
} bitmap_jpeg_err_ctx;

static void bitmap_jpeg_err(j_common_ptr cinfo) {
	char errorMsg[JMSG_LENGTH_MAX];
	bitmap_jpeg_err_ctx* jerr = (bitmap_jpeg_err_ctx*)cinfo->err;
	(*(cinfo->err->format_message))(cinfo, errorMsg);
	fprintf(stderr, "%s\n", errorMsg);
	longjmp(jerr->setjmp_buffer, 1);
}

bitmap_t* bitmap_read_jpeg(const char *filename, int32_t control) {
	struct jpeg_decompress_struct ci;
	FILE * volatile fp; int volatile ok = 0;
	bitmap_t * volatile bm = NULL;
	void * volatile mem = NULL;

	bitmap_jpeg_err_ctx jerr;
	ci.err = jpeg_std_error(&jerr.pub);
	jerr.pub.error_exit = bitmap_jpeg_err;
	if (!setjmp(jerr.setjmp_buffer)) {
		unsigned x, y, width, height, stride; uint8_t *data;
		bitmap_t *bm1; JSAMPROW *scanline;
		int run_quantsmooth = control & (JPEGQS_ITER_MAX << JPEGQS_ITER_SHIFT);

		jpeg_create_decompress(&ci);
		if (!(fp = fopen(filename, "rb"))) goto err;
		jpeg_stdio_src(&ci, fp);
		jpeg_read_header(&ci, TRUE);
		ci.out_color_space = JCS_RGB;
		if (control & JPEGQS_UPSAMPLE_UV)
			ci.do_fancy_upsampling = FALSE;
		if (run_quantsmooth) ci.buffered_image = TRUE;
		jpeg_start_decompress(&ci);
		if (run_quantsmooth) {
			jvirt_barray_ptr *src_coef_arrays;
			while (!jpeg_input_complete(&ci)) {
				jpeg_start_output(&ci, ci.input_scan_number);
				jpeg_finish_output(&ci);
			}
			src_coef_arrays = jpeg_read_coefficients(&ci);
			do_quantsmooth(&ci, src_coef_arrays, control);

			if (control & JPEGQS_UPSAMPLE_UV) {
#ifdef LIBJPEG_TURBO_VERSION
				ci.master->last_MCU_col[1] = ci.master->last_MCU_col[0];
				ci.master->last_MCU_col[2] = ci.master->last_MCU_col[0];
#endif
				jinit_color_deconverter(&ci);
				jinit_upsampler(&ci);
				jinit_d_main_controller(&ci, FALSE);
				ci.input_iMCU_row = (ci.output_height + DCTSIZE - 1) / DCTSIZE;
			}

			jinit_inverse_dct(&ci);
			jpeg_start_output(&ci, ci.input_scan_number);
		}

		width = ci.output_width;
		height = ci.output_height;
		bm = bm1 = bitmap_create(width, height, 3);
		if (bm1) {
			stride = bm1->stride; data = bm1->data;
			mem = malloc(height * sizeof(JSAMPROW));
			if (mem) {
				scanline = (JSAMPROW*)mem;
				// BMP uses reverse row order
				for (y = 0; y < height; y++)
					scanline[y] = (JSAMPLE*)(data + (height - 1 - y) * stride);

				while ((y = ci.output_scanline) < height) {
					jpeg_read_scanlines(&ci, scanline + y, height - y);
				}

				// need to convert RGB to BGR for BMP
				for (y = 0; y < height; y++) {
					for (x = 0; x < width; x++) {
						JSAMPLE *p = &data[y * stride + x * 3], t;
						t = p[0]; p[0] = p[2]; p[2] = t;
					}
					for (x = width * 3; x < stride; x++) data[y * stride + x] = 0;
				}

				ok = 1;
			}
		}

		if (run_quantsmooth) jpeg_finish_output(&ci);
		jpeg_finish_decompress(&ci);
	}
err:
	if (mem) free(mem);
	if (!ok) { bitmap_free(bm); bm = NULL; }
	jpeg_destroy_decompress(&ci);
	if (fp) fclose(fp);
	return bm;
}

int main(int argc, char **argv) {
	bitmap_t *bm; const char *ifn, *ofn;
	int32_t control = 0; int niter = 3;

	if (argc != 3) {
		printf("Usage: example input.jpg output.bmp\n");
		return 1;
	}

	control |= JPEGQS_DIAGONALS; /* -q4 */
	control |= JPEGQS_JOINT_YUV; /* -q5 */
	control |= JPEGQS_UPSAMPLE_UV; /* -q6 */
	if (niter < 0) niter = 0;
	if (niter > JPEGQS_ITER_MAX) niter = JPEGQS_ITER_MAX;
	control |= niter << JPEGQS_ITER_SHIFT;

	ifn = argv[1]; ofn = argv[2];
	bm = bitmap_read_jpeg(ifn, control);
	if (bm) {
		FILE *f;
		int w = bm->width, h = bm->height, st = bm->stride;
		int32_t n = 54, bmp[] = { 0x4d420000, -1, 0, 54,
			40, -1, -1, (24 << 16) + 1, 0, -1, 2835, 2835, 0, 0 };
		bmp[1] = n + h * st;
		bmp[5] = w; bmp[6] = h;
		bmp[9] = h * st;

		if ((f = fopen(ofn, "wb"))) {
			n = fwrite((uint8_t*)bmp+2, 1, n, f);
			n = fwrite(bm->data, 1, h * st, f);
			fclose(f);
		}
		bitmap_free(bm);
	}
	return bm ? 0 : 1;
}

