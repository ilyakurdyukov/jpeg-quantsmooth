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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define WASM
#endif

#include "jpeglib.h"

#ifdef WASM
#define logfmt(...) printf(__VA_ARGS__)
#else
#define logfmt(...) fprintf(stderr, __VA_ARGS__)
#endif

#define WITH_LOG
#include "quantsmooth.h"

#ifdef WASM
#define MEM_INPUT
#define MEM_OUTPUT
#endif

#ifdef MEM_INPUT
#ifndef WASM
static uint8_t* loadfile(const char *fn, size_t *num) {
	size_t n, j = 0; uint8_t *buf = 0;
	FILE *fi = fopen(fn, "rb");
	if (fi) {
		fseek(fi, 0, SEEK_END);
		n = ftell(fi);
		fseek(fi, 0, SEEK_SET);
		if (n) {
			buf = (uint8_t*)malloc(n);
			if (buf) j = fread(buf, 1, n, fi);
		}
		fclose(fi);
	}
	if (num) *num = j;
	return buf;
}
#endif

void jpeg_init_source(j_decompress_ptr cinfo) { }
boolean jpeg_fill_input_buffer(j_decompress_ptr cinfo) { return FALSE; }
void jpeg_skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
  struct jpeg_source_mgr *src = cinfo->src;
	if (num_bytes > src->bytes_in_buffer) num_bytes = src->bytes_in_buffer;
	src->next_input_byte += (size_t)num_bytes;
	src->bytes_in_buffer -= (size_t)num_bytes;
}
void jpeg_term_source(j_decompress_ptr cinfo) { }
#endif

#ifdef MEM_OUTPUT
struct jpeg_dest_mem {
  struct jpeg_destination_mgr pub; /* public fields */
  uint8_t *buffer; size_t size, bufsize, maxchunk;
};

static void jpeg_init_destination(j_compress_ptr cinfo) {
	struct jpeg_dest_mem *p = (struct jpeg_dest_mem*)cinfo->dest;
	if (!p->buffer) {
		p->buffer = (uint8_t*)malloc(p->bufsize);
	}
	p->pub.next_output_byte = p->buffer;
	p->pub.free_in_buffer = p->bufsize;
}

static boolean jpeg_empty_output_buffer(j_compress_ptr cinfo) {
	struct jpeg_dest_mem *p = (struct jpeg_dest_mem*)cinfo->dest;
	size_t offset = p->bufsize, next;
	p->size = offset;
	next = p->bufsize;
	if (next > p->maxchunk) next = p->maxchunk;
	p->bufsize += next;
	p->buffer = (uint8_t*)realloc(p->buffer, p->bufsize);
	p->pub.next_output_byte = p->buffer + offset;
	p->pub.free_in_buffer = p->bufsize - offset;
	return TRUE;
}

static void jpeg_term_destination(j_compress_ptr cinfo) {
	struct jpeg_dest_mem *p = (struct jpeg_dest_mem*)cinfo->dest;
	p->size = p->bufsize - p->pub.free_in_buffer;
}
#endif

#ifdef WASM
static char** make_argv(char *str, int *argc_ret) {
	int i = 0, eol = 0, argc = 1; char **argv;
	for (;;) {
		char a = str[i++];
		if (!a) break;
		if (eol) {
			if (a == eol) eol = 0;
		} else {
			if (a != ' ') {
				eol = ' ';
				if (a == '"' || a == '\'') eol = a;
				argc++;
			}
		}
	}
	*argc_ret = argc;
	argv = (char**)malloc(argc * sizeof(char*));
	if (!argv) return argv;
	argv[0] = NULL; eol = 0; argc = 1;
	for (;;) {
		char a = *str++;
		if (!a) break;
		if (eol) {
			if (a == eol) { str[-1] = 0; eol = 0; }
		} else {
			if (a != ' ') {
				eol = ' ';
				if (a == '"' || a == '\'') { eol = a; str++; }
				argv[argc++] = str - 1;
			}
		}
	}
	return argv;
}

EMSCRIPTEN_KEEPALIVE
int web_process(int64_t *params) {
	char *cmdline = (char*)params[0];
#else
int main(int argc, char **argv) {
#endif
	struct jpeg_decompress_struct srcinfo;
	struct jpeg_compress_struct dstinfo;
	struct jpeg_error_mgr jsrcerr, jdsterr;
	jvirt_barray_ptr *src_coef_arrays;

#ifdef MEM_INPUT
	size_t input_size = 0;
	uint8_t *input_mem = NULL;
  struct jpeg_source_mgr src_mgr = { 0 };
#else
	FILE *input_file = stdin;
#endif
#ifdef MEM_OUTPUT
	struct jpeg_dest_mem dest_mgr = { { 0 }, NULL, 0, 0x10000, 0x100000 };
#endif
#ifndef WASM
	FILE *output_file = stdout;
#endif

	int optimize = 0, verbose_level = 0, smooth_info = 0xf;
#ifdef WASM
	int argc = 0;
	char **argv = make_argv(cmdline, &argc);
#else
	const char *progname = "quantsmooth", *fn;
#endif

	while (argc > 1) {
		if (!strcmp(argv[1], "--optimize")) {
			optimize = 1;
			argv++; argc--;
		}
		else if (argc > 2 && !strcmp(argv[1], "--verbose")) {
			verbose_level = atoi(argv[2]);
			argv += 2; argc -= 2;
		}
		else if (argc > 2 && !strcmp(argv[1], "--info")) {
			smooth_info = atoi(argv[2]);
			argv += 2; argc -= 2;
		}
		else if (!strcmp(argv[1], "--")) {
			argv++; argc--;
			break;
		}
		else break;
	}

#ifdef WASM
	free(argv);
	if (argc != 1) {
		logfmt("Unrecognized command line option.\n");
		return 1;
	}
#else
	if (argc != 3) {
		logfmt(
"Usage:\n"
"  %s [options] input.jpg output.jpg\n"
"Options:\n"
"  --optimize        Optimize Huffman table (smaller file, but slow compression)\n"
"  --verbose level   Print libjpeg debug output\n"
"  --info flags      Print quantsmooth debug output\n", progname);
		return 1;
	}
#endif

	if (verbose_level) {
#ifdef LIBJPEG_TURBO_VERSION
#define TOSTR(s) #s
#define TOSTR1(s) TOSTR(s)
		logfmt("using libjpeg-turbo version %s\n", TOSTR1(LIBJPEG_TURBO_VERSION));
#else
		logfmt("using libjpeg version %d\n", JPEG_LIB_VERSION);
#endif
		verbose_level--;
	}

	srcinfo.err = jpeg_std_error(&jsrcerr);
	jpeg_create_decompress(&srcinfo);
	dstinfo.err = jpeg_std_error(&jdsterr);
	jpeg_create_compress(&dstinfo);

	jsrcerr.trace_level = jdsterr.trace_level = verbose_level;
	srcinfo.mem->max_memory_to_use = dstinfo.mem->max_memory_to_use;

#ifdef WASM
  input_mem = (uint8_t*)params[1];
  input_size = params[2];
#else
	fn = argv[1];
#endif
#ifdef MEM_INPUT
#ifndef WASM
	input_mem = loadfile(fn, &input_size);
	if (!input_mem) {
		logfmt("%s: can't open input file \"%s\"\n", progname, fn);
		return 1;
	}
#endif
#if 0 && (JPEG_LIB_VERSION >= 80 || defined(MEM_SRCDST_SUPPORTED))
	jpeg_mem_src(&srcinfo, input_mem, input_size);
#else
	srcinfo.src = &src_mgr;
	src_mgr.init_source = jpeg_init_source;
	src_mgr.fill_input_buffer = jpeg_fill_input_buffer;
	src_mgr.skip_input_data = jpeg_skip_input_data;
	src_mgr.resync_to_restart = jpeg_resync_to_restart; /* use default method */
	src_mgr.term_source = jpeg_term_source;
  src_mgr.next_input_byte = (const JOCTET*)input_mem;
  src_mgr.bytes_in_buffer = input_size;
#endif
#else
	if (strcmp(fn, "-")) {
		if ((input_file = fopen(fn, "rb")) == NULL) {
			logfmt("%s: can't open input file \"%s\"\n", progname, fn);
			return 1;
		}
	}
	jpeg_stdio_src(&srcinfo, input_file);
#endif

	(void) jpeg_read_header(&srcinfo, TRUE);
	src_coef_arrays = jpeg_read_coefficients(&srcinfo);
	do_quantsmooth(&srcinfo, src_coef_arrays, smooth_info);

	jpeg_copy_critical_parameters(&srcinfo, &dstinfo);
	if (optimize) dstinfo.optimize_coding = TRUE;

#ifdef MEM_OUTPUT
	// uint8_t *outbuffer; unsigned long outsize;
	// jpeg_mem_dest(dstinfo, &outbuffer, &outsize);
	dest_mgr.pub.init_destination = jpeg_init_destination;
	dest_mgr.pub.empty_output_buffer = jpeg_empty_output_buffer;
	dest_mgr.pub.term_destination = jpeg_term_destination;
	dstinfo.dest = (struct jpeg_destination_mgr*)&dest_mgr;
#else
	// If output opened after reading coefs, then we can write result to input file
	fn = argv[2];
	if (strcmp(fn, "-")) {
		if ((output_file = fopen(fn, "wb")) == NULL) {
			logfmt("%s: can't open output file \"%s\"\n", progname, fn);
			return 1;
		}
	}
	jpeg_stdio_dest(&dstinfo, output_file);
#endif

	/* Start compressor (note no image data is actually written here) */
	jpeg_write_coefficients(&dstinfo, src_coef_arrays);

	/* Finish compression and release memory */
	jpeg_finish_compress(&dstinfo);
	jpeg_destroy_compress(&dstinfo);
	(void) jpeg_finish_decompress(&srcinfo);
	jpeg_destroy_decompress(&srcinfo);

	/* Close files, if we opened them */
#ifndef MEM_INPUT
	if (input_file != stdin) fclose(input_file);
#endif
#ifdef WASM
	params[3] = (int64_t)dest_mgr.buffer;
	params[4] = dest_mgr.size;
#else
#ifdef MEM_OUTPUT
	fn = argv[2];
	if (strcmp(fn, "-")) {
		if ((output_file = fopen(fn, "wb")) == NULL) {
			logfmt("%s: can't open output file \"%s\"\n", progname, fn);
			return 1;
		}
	}
	if (dest_mgr.buffer) {
		fwrite(dest_mgr.buffer, 1, dest_mgr.size, output_file);
		free(dest_mgr.buffer);
	}
#endif
	if (output_file != stdout) fclose(output_file);
#endif

	return jsrcerr.num_warnings + jdsterr.num_warnings ? 2 : 0;
}
