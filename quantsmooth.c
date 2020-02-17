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

#define STRINGIFY(s) #s
#define TOSTRING(s) STRINGIFY(s)

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#if defined(WASM_MAIN) && !defined(WASM)
#define WASM
#endif

#include "jpeglib.h"
#ifdef WITH_JPEGSRC
#include "jversion.h"
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
// conflict with libjpeg typedef
#define INT32 INT32_WIN
#include <windows.h>
#define USE_SETMODE
#endif

#ifdef USE_SETMODE
#include <fcntl.h>
#include <io.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef WASM
#define logfmt(...) printf(__VA_ARGS__)
#else
#define logfmt(...) fprintf(stderr, __VA_ARGS__)
#endif

#define WITH_LOG

#ifdef WITH_LOG
#ifdef _WIN32
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

#include "quantsmooth.h"

#define CONCAT(a, b) a##b
#ifdef UNICODE
#define S(s) CONCAT(L, s)
#define LS "%S"
#else
#define S(s) s
#define LS "%s"
#endif

#ifndef TCHAR
#ifdef UNICODE
#define TCHAR wchar_t
#else
#define TCHAR char
#endif
#endif

#ifdef WASM
#define MEM_INPUT
#define MEM_OUTPUT
#endif

#ifdef MEM_INPUT
#if !defined(WASM) || defined(WASM_MAIN)
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

void jpeg_init_source(j_decompress_ptr cinfo) { (void)cinfo; }
boolean jpeg_fill_input_buffer(j_decompress_ptr cinfo) { (void)cinfo; return FALSE; }
void jpeg_skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
	struct jpeg_source_mgr *src = cinfo->src;
	if ((size_t)num_bytes > src->bytes_in_buffer) num_bytes = src->bytes_in_buffer;
	src->next_input_byte += (size_t)num_bytes;
	src->bytes_in_buffer -= (size_t)num_bytes;
}
void jpeg_term_source(j_decompress_ptr cinfo) { (void)cinfo; }
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
#ifdef UNICODE
// unicode hacks
#define strcmp(a, b) wcscmp(a, S(b))
#define atoi(a) _wtoi(a)
#define fopen(a, b) _wfopen(a, S(b))
#pragma GCC diagnostic ignored "-Wformat"
int wmain(int argc, wchar_t **argv) {
#else
int main(int argc, char **argv) {
#endif
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

	int optimize = 0, jpeg_verbose = 0, threads = 0;
	int cmd_info = 15, quality = 3, cmd_niter = -1;
	int32_t jpegqs_flags = -1;
#ifdef WASM
	int argc = 0;
	char **argv_ptr = make_argv(cmdline, &argc), **argv = argv_ptr;
#else
#ifndef APPNAME
	const TCHAR *progname = argv[0], *fn;
#else
	const TCHAR *progname = S(TOSTRING(APPNAME)), *fn;
#endif
#endif

	while (argc > 1) {
		const TCHAR *arg1 = argv[1], *arg2 = argc > 2 ? argv[2] : NULL, *arg = arg1; TCHAR c;
		if (arg[0] != '-' || !(c = arg[1])) break;
		if (c != '-') switch (c) {
			case 'o': arg = S("--optimize"); c = 0; break;
			case 'v': arg = S("--verbose"); break;
			case 'i': arg = S("--info"); break;
			case 'n': arg = S("--niter"); break;
			case 'q': arg = S("--quality"); break;
			case 't': arg = S("--threads"); break;
			case 'f': arg = S("--flags"); break;
			default: c = '-';
		}
		if (c != '-' && arg1[2]) {
			if (!c) break;
			arg2 = arg1 + 2; argc++; argv--;
		}

#define CHECKNUM if ((unsigned)(arg2[0] - '0') > 9) break;
		if (!strcmp(arg, "--optimize")) {
			optimize = 1;
			argv++; argc--;
		} else if (argc > 2 && !strcmp(arg, "--verbose")) {
			CHECKNUM
			jpeg_verbose = atoi(arg2);
			argv += 2; argc -= 2;
		} else if (argc > 2 && !strcmp(arg, "--info")) {
			CHECKNUM
			cmd_info = atoi(arg2);
			argv += 2; argc -= 2;
		} else if (argc > 2 && !strcmp(arg, "--niter")) {
			CHECKNUM
			cmd_niter = atoi(arg2);
			argv += 2; argc -= 2;
		} else if (argc > 2 && !strcmp(arg, "--quality")) {
			CHECKNUM
			quality = atoi(arg2);
			argv += 2; argc -= 2;
		} else if (argc > 2 && !strcmp(arg, "--threads")) {
			CHECKNUM
			threads = atoi(arg2);
			argv += 2; argc -= 2;
		} else if (argc > 2 && !strcmp(arg, "--flags")) {
			CHECKNUM
			jpegqs_flags = atoi(arg2);
			argv += 2; argc -= 2;
		} else if (!strcmp(arg, "--")) {
			argv++; argc--;
			break;
		}
		else break;
	}

#ifdef _OPENMP
	if (threads > 0) {
		omp_set_num_threads(threads);
	}
#endif

	{
		int niter = 3, flags = 0;
		if (quality <= 1) niter = 1;
		else if (quality <= 2) niter = 2;
		else if (quality >= 4) flags = JPEGQS_DIAGONALS;
		if (quality >= 5) flags |= JPEGQS_JOINT_YUV;
		if (quality >= 6) flags |= JPEGQS_UPSAMPLE_UV;

		niter = cmd_niter >= 0 ? cmd_niter : niter;
		if (niter > JPEGQS_ITER_MAX) niter = JPEGQS_ITER_MAX;

		if (jpegqs_flags < 0) jpegqs_flags = flags;
		else jpegqs_flags = (jpegqs_flags & JPEGQS_FLAGS_MASK) << JPEGQS_FLAGS_SHIFT;
		jpegqs_flags |= cmd_info & JPEGQS_INFO_MASK;
		jpegqs_flags |= niter << JPEGQS_ITER_SHIFT;
	}

#ifdef WASM
	free(argv_ptr);
	if (argc != 1) {
		logfmt("Unrecognized command line option.\n");
		return 1;
	}
#endif

	srcinfo.err = jpeg_std_error(&jsrcerr);

	if (jpeg_verbose) {
#ifdef LIBJPEG_TURBO_VERSION
		logfmt("Compiled with libjpeg-turbo version %s\n", TOSTRING(LIBJPEG_TURBO_VERSION));
#else
		logfmt("Compiled with libjpeg version %d\n", JPEG_LIB_VERSION);
#endif
#if defined(JVERSION) && defined(JCOPYRIGHT)
		logfmt("Version string: " JVERSION "\n" JCOPYRIGHT "\n\n");
#else
		// Search for libjpeg copyright (to work with static and dynamic linking)
		{
			int i, n = jsrcerr.last_jpeg_message;
			const char *msg, *ver = NULL;
			for (i = 0; i < n; i++) {
				msg = jsrcerr.jpeg_message_table[i];
				if (msg && !memcmp(msg, "Copyright", 9)) break;
			}
			if (i < n) {
				if (i + 1 < n) {
					// version should be next to copyright
					ver = jsrcerr.jpeg_message_table[i + 1];
					// check that it starts with a number
					if (ver && (ver[0] < '0' || ver[0] > '9')) ver = NULL;
				}
				if (!ver) ver = "not found";
				logfmt("Version string: %s\n%s\n\n", ver, msg);
			} else {
				logfmt("Copyright not found\n\n");
			}
		}
#endif
		jpeg_verbose--;
#ifndef WASM
		if (argc == 1) return 1;
#endif
	}

#ifndef WASM
	if (argc != 3) {
		logfmt(
"JPEG Quant Smooth : Copyright (c) 2020 Ilya Kurdyukov\n"
"Build date: " __DATE__ "\n"
"Uses libjpeg, run with \"--verbose 1\" to show its version and copyright\n"
"\n"
"Usage:\n"
"  " LS " [options] input.jpg output.jpg\n"
"\n"
"Options:\n"
"  -q, --quality n   Quality setting (1-6, default is 3)\n"
"  -n, --niter n     Number of iterations (default is 3)\n"
"  -t, --threads n   Set the number of CPU threads to use\n"
"  -o, --optimize    Option for libjpeg to produce smaller output file\n"
"  -v, --verbose n   Print libjpeg debug output\n"
"  -i, --info n      Print quantsmooth debug output:\n"
"                      0 - silent, 8 - processing time, 15 - all (default)\n"
"\n", progname);
		return 1;
	}
#endif

	jpeg_create_decompress(&srcinfo);
	dstinfo.err = jpeg_std_error(&jdsterr);
	jpeg_create_compress(&dstinfo);

	jsrcerr.trace_level = jdsterr.trace_level = jpeg_verbose;
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
		logfmt(LS ": can't open input file \"" LS "\"\n", progname, fn);
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
			logfmt(LS ": can't open input file \"" LS "\"\n", progname, fn);
			return 1;
		}
	} else {
#ifdef USE_SETMODE
	  setmode(fileno(stdin), O_BINARY);
#endif
	}
	jpeg_stdio_src(&srcinfo, input_file);
#endif

	(void) jpeg_read_header(&srcinfo, TRUE);
	src_coef_arrays = jpeg_read_coefficients(&srcinfo);
	do_quantsmooth(&srcinfo, src_coef_arrays, jpegqs_flags);

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
			logfmt(LS ": can't open output file \"" LS "\"\n", progname, fn);
			return 1;
		}
	} else {
#ifdef USE_SETMODE
	  setmode(fileno(stdout), O_BINARY);
#endif
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
	params[3] = (intptr_t)dest_mgr.buffer;
	params[4] = dest_mgr.size;
#else
#ifdef MEM_OUTPUT
	fn = argv[2];
	if (strcmp(fn, "-")) {
		if ((output_file = fopen(fn, "wb")) == NULL) {
			logfmt(LS ": can't open output file \"" LS "\"\n", progname, fn);
			return 1;
		}
	} else {
#ifdef USE_SETMODE
	  setmode(fileno(stdout), O_BINARY);
#endif
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

// for testing purposes
#ifdef WASM_MAIN
int main(int argc, char **argv) {
	int64_t params[5]; int i, n, ret;
	char *cmdline = NULL; size_t cmd_size = 1;
	const char *progname = "quantsmooth", *fn;
	uint8_t *input_mem; size_t input_size;

	if (argc < 3) {
		logfmt("Unrecognized command line.\n");
		return 1;
	}

	for (n = 1; n < argc - 2; n++)
		cmd_size += strlen(argv[n]) + 3;
	cmdline = malloc(cmd_size);
	if (!cmdline) return 1;

	cmd_size = 0;
	for (i = 1; i < argc - 2; i++) {
		const char *str = argv[i];
		int len = strlen(str);
		cmdline[cmd_size++] = '"';
		memcpy(cmdline + cmd_size, str, len);
		cmd_size += len;
		cmdline[cmd_size++] = '"';
		cmdline[cmd_size++] = ' ';
	}
	cmdline[cmd_size] = 0;
	params[0] = (intptr_t)cmdline;

	// printf("cmdline: %s\n", cmdline);

	argv += argc - 3;

	fn = argv[1];
	input_mem = loadfile(fn, &input_size);
	if (!input_mem) {
		logfmt("%s: can't open input file \"%s\"\n", progname, fn);
		return 1;
	}
	params[1] = (intptr_t)input_mem;
	params[2] = input_size;
	params[3] = 0;
	params[4] = 0;
	ret = web_process(params);
	free(input_mem);

	if (params[3]) {
		FILE *output_file;
		fn = argv[2];
		if ((output_file = fopen(fn, "wb")) == NULL) {
			logfmt("%s: can't open output file \"%s\"\n", progname, fn);
			return 1;
		}
		fwrite((void*)params[3], 1, (size_t)params[4], output_file);
		fclose(output_file);
		free((void*)params[3]);
	}

	if (cmdline) free(cmdline);
	return ret;
}
#endif

