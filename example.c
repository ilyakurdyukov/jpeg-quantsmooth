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
#ifdef WITH_JPEGSRC
#define JPEG_INTERNALS
#endif
#include "jpeglib.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// use "libjpegqs.h" for linking with library
#include "quantsmooth.h"

typedef struct {
	int width, height, bpp, stride; uint8_t *data;
} bitmap_t;

static bitmap_t *bitmap_create(int width, int height, int bpp) {
	bitmap_t *bm;
	// BMP needs 4-byte row alignment
	int stride = (width * bpp + 3) & -4;
	uint64_t size = (int64_t)stride * height + sizeof(bitmap_t);
	// check for overflow
	if ((unsigned)((width - 1) | (height - 1)) >= 0x10000 ||
			(uint64_t)(size_t)size != size) return NULL;
	bm = (bitmap_t*)malloc(size);
	if (!bm) return bm;
	bm->width = width;
	bm->height = height;
	bm->bpp = bpp;
	bm->stride = stride;
	bm->data = (uint8_t*)(bm + 1);
	return bm;
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

bitmap_t* bitmap_read_jpeg(const char *filename, jpegqs_control_t *opts) {
	struct jpeg_decompress_struct ci;
	FILE * volatile fp; int volatile ok = 0;
	bitmap_t * volatile bm = NULL;
	void * volatile mem = NULL;

	bitmap_jpeg_err_ctx jerr;
	ci.err = jpeg_std_error(&jerr.pub);
	jerr.pub.error_exit = bitmap_jpeg_err;
	if (!setjmp(jerr.setjmp_buffer)) do {
		unsigned x, y, w, h, st; uint8_t *data;
		bitmap_t *bm1; JSAMPROW *scanline;

		jpeg_create_decompress(&ci);
		if (!(fp = fopen(filename, "rb"))) break;
		jpeg_stdio_src(&ci, fp);
		jpeg_read_header(&ci, TRUE);
		ci.out_color_space = JCS_RGB;
		jpegqs_start_decompress(&ci, opts);

		w = ci.output_width;
		h = ci.output_height;
		bm = bm1 = bitmap_create(w, h, 3);
		if (!bm1) break;
		mem = scanline = (JSAMPROW*)malloc(h * sizeof(JSAMPROW));
		if (!scanline) break;
		st = bm1->stride; data = bm1->data;

		// BMP uses reverse row order
		for (y = 0; y < h; y++)
			scanline[y] = (JSAMPLE*)(data + (h - 1 - y) * st);

		while ((y = ci.output_scanline) < h)
			jpeg_read_scanlines(&ci, scanline + y, h - y);

		// need to convert RGB to BGR for BMP
		for (y = 0; y < h; y++) {
			JSAMPLE *p = data + y * st, t;
			for (x = 0; x < w * 3; x += 3) {
				t = p[x]; p[x] = p[x + 2]; p[x + 2] = t;
			}
			for (; x < st; x++) p[x] = 0;
		}

		ok = 1;
		jpegqs_finish_decompress(&ci);
	} while (0);

	if (mem) free(mem);
	if (!ok) { bitmap_free(bm); bm = NULL; }
	jpeg_destroy_decompress(&ci);
	if (fp) fclose(fp);
	return bm;
}

typedef struct {
	int init;
} progress_data_t;

static int progress(void *data, int cur, int max) {
	progress_data_t *prog = (progress_data_t*)data;
	printf("%s%i%%", prog->init ? ", " : "progress: ", 100 * cur / max);
	fflush(stdout);
	prog->init = 1;
	// return nonzero value to stop processing
	return 0;
}

int main(int argc, char **argv) {
	bitmap_t *bm; const char *ifn, *ofn;
	jpegqs_control_t opts;
	progress_data_t prog;

	if (argc != 3) {
		printf("Usage: example input.jpg output.bmp\n");
		return 1;
	}
	ifn = argv[1]; ofn = argv[2];

	memset(&opts, 0, sizeof(opts));
	opts.niter = 3;
	opts.flags |= JPEGQS_DIAGONALS; /* -q4 */
	opts.flags |= JPEGQS_JOINT_YUV; /* -q5 */
	opts.flags |= JPEGQS_UPSAMPLE_UV; /* -q6 */
	prog.init = 0;
	opts.userdata = &prog;
	opts.progress = progress;

	bm = bitmap_read_jpeg(ifn, &opts);
	if (prog.init) printf("\n");
	if (bm) {
		FILE *f;
		int w = bm->width, h = bm->height, st = bm->stride;
		int32_t n = 54, bmp[] = { 0x4d420000, -1, 0, 54,
			40, -1, -1, (24 << 16) + 1, 0, -1, 2835, 2835, 0, 0 };
		bmp[1] = n + h * st;
		bmp[5] = w; bmp[6] = h;
		bmp[9] = h * st;

		if ((f = fopen(ofn, "wb"))) {
			n = fwrite((uint8_t*)bmp + 2, 1, n, f);
			n = fwrite(bm->data, 1, h * st, f);
			fclose(f);
		}
		bitmap_free(bm);
	}
	return bm ? 0 : 1;
}

