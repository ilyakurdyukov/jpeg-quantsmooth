/*
 * Copyright (C) 2020 Ilya Kurdyukov
 *
 * JPEG reader plugin for IrfanView
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <setjmp.h>
#include "jpeglib.h"

#define WIN32_LEAN_AND_MEAN
// conflict with libjpeg typedef
#define INT32 INT32_WIN
#include <windows.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "libjpegqs.h"

typedef BITMAPINFOHEADER bitmap_t;

static bitmap_t *bitmap_create(int width, int height, int bpp) {
	bitmap_t *bm;
	// BMP needs 4-byte row alignment
	int stride = (width * bpp + 3) & -4;
	uint64_t size = (int64_t)stride * height + sizeof(bitmap_t) + (bpp == 1 ? 256 * 4 : 0);
	// check for overflow
	if ((unsigned)((width - 1) | (height - 1)) >= 0x10000 ||
			(uint64_t)(SIZE_T)size != size) return NULL;
	bm = (bitmap_t*)GlobalAlloc(GMEM_FIXED, size);
	if (!bm) return bm;
	memset(bm, 0, sizeof(bitmap_t));
	bm->biSize = sizeof(bitmap_t);
	bm->biWidth = width;
	bm->biHeight = height;
	bm->biPlanes = 1;
	bm->biBitCount = bpp * 8;
	bm->biSizeImage = height * stride;
	if (bpp == 1) {
		int i; int32_t *p = (int32_t*)(bm + 1);
		bm->biClrUsed = 256;
		for (i = 0; i < 256; i++) p[i] = i * 0x010101;
	}
	return bm;
}

static void bitmap_free(bitmap_t *in) {
	if (in) GlobalFree(in);
}

typedef struct {
	struct jpeg_error_mgr pub;
	wchar_t *errbuf;
	jmp_buf setjmp_buffer;
} bitmap_jpeg_err_ctx;

static inline void copyMsg(wchar_t *errbuf, const char *msg) {
	int i = 0; uint8_t a;
	do { errbuf[i] = a = msg[i]; i++; } while (a);
}

static void bitmap_jpeg_err(j_common_ptr cinfo) {
	char errorMsg[JMSG_LENGTH_MAX];
	bitmap_jpeg_err_ctx* jerr = (bitmap_jpeg_err_ctx*)cinfo->err;
	(*(cinfo->err->format_message))(cinfo, errorMsg);
	copyMsg(jerr->errbuf, errorMsg);
	longjmp(jerr->setjmp_buffer, 1);
}

static bitmap_t* bitmap_read_jpeg(const wchar_t *filename,
		jpegqs_control_t *opts, wchar_t *errbuf, int grayscale) {
	struct jpeg_decompress_struct ci;
	FILE * volatile fp; int volatile ok = 0;
	bitmap_t * volatile bm = NULL;
	void * volatile mem = NULL;

	bitmap_jpeg_err_ctx jerr;
	ci.err = jpeg_std_error(&jerr.pub);
	jerr.errbuf = errbuf;
	jerr.pub.error_exit = bitmap_jpeg_err;
	if (!setjmp(jerr.setjmp_buffer)) do {
		unsigned x, y, w, h, st; uint8_t *data;
		bitmap_t *bm1; JSAMPROW *scanline; int bpp;

		jpeg_create_decompress(&ci);
		if (!(fp = _wfopen(filename, L"rb"))) {
			copyMsg(errbuf, "Error opening file for reading");
			break;
		}
		jpeg_stdio_src(&ci, fp);
		jpeg_read_header(&ci, TRUE);
		bpp = grayscale || ci.num_components == 1 ? 1 : 3;
		ci.out_color_space = bpp == 1 ? JCS_GRAYSCALE : JCS_RGB;
		jpegqs_start_decompress(&ci, opts);

		w = ci.output_width;
		h = ci.output_height;
		bm = bm1 = bitmap_create(w, h, bpp);
		if (bm1) mem = scanline = (JSAMPROW*)malloc(h * sizeof(JSAMPROW));
		if (!bm1 || !scanline) {
			copyMsg(errbuf, "Memory allocation failed");
			break;
		}
		st = (w * bpp + 3) & -4; data = (uint8_t*)(bm1 + 1);
		if (bpp == 1) data += 256 * 4;

		// BMP uses reverse row order
		for (y = 0; y < h; y++)
			scanline[y] = (JSAMPLE*)(data + (h - 1 - y) * st);

		while ((y = ci.output_scanline) < h)
			jpeg_read_scanlines(&ci, scanline + y, h - y);

		if (bpp == 1)
		for (y = 0; y < h; y++)
			for (x = w; x < st; x++) data[y * st + x] = 0;
		else
		for (y = 0; y < h; y++) {
			JSAMPLE *p = data + y * st, t;
			// need to convert RGB to BGR for BMP
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

// Do only one processing at a time (need for MiniOMP).
#define NEED_LOCK
// Do processing in a separate thread.
// Need when linking with libgomp from Mingw-w64, otherwise unstable.
//#define SEPARATE_THREAD

static HINSTANCE hInst = NULL;
#ifdef NEED_LOCK
static CRITICAL_SECTION CriticalSection;
#define LOCK EnterCriticalSection(&CriticalSection);
#define UNLOCK LeaveCriticalSection(&CriticalSection);
#else
#define LOCK
#define UNLOCK
#endif

BOOL WINAPI DllMain(HINSTANCE hinstDLL, ULONG fdwReason, LPVOID lpvReserved) {
	(void)lpvReserved;
	if (fdwReason == DLL_PROCESS_ATTACH) {
		hInst = hinstDLL;
#ifdef NEED_LOCK
		InitializeCriticalSection(&CriticalSection);
#endif
	} else if (fdwReason == DLL_PROCESS_DETACH) {
#ifdef NEED_LOCK
		DeleteCriticalSection(&CriticalSection);
#endif
	}
	return TRUE;
}

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static jpegqs_control_t jpegqs_default = {
	JPEGQS_DIAGONALS | JPEGQS_JOINT_YUV | JPEGQS_UPSAMPLE_UV, 3, -1, 0, NULL, NULL
};

// threads < 0 : doesn't changes current number
// threads = 0 : use all available cpu threads

#ifdef SEPARATE_THREAD
#define LoadJPG LoadJPG_orig
static
#else
EXPORT
#endif
HANDLE LoadJPG(const wchar_t *filename, jpegqs_control_t *opts, wchar_t *errbuf, int grayscale, void *dummy) {
	bitmap_t *bm;
	(void)dummy;
	jpegqs_control_t opts2;
	if (!opts) opts = &jpegqs_default;
	memcpy(&opts2, opts, sizeof(opts2));
	opts2.flags &= ~JPEGQS_TRANSCODE;
	if (grayscale) opts2.flags &= ~(JPEGQS_JOINT_YUV | JPEGQS_UPSAMPLE_UV);
#ifndef SEPARATE_THREAD
	LOCK
#endif
	bm = bitmap_read_jpeg(filename, &opts2, errbuf, grayscale);
#ifndef SEPARATE_THREAD
	UNLOCK
#endif
	return (HANDLE)bm;
}

#ifdef SEPARATE_THREAD
#undef LoadJPG

typedef struct {
	const wchar_t *filename; jpegqs_control_t *opts; wchar_t *errbuf; int grayscale; void *dummy;
	HANDLE result;
} LoadJPGParams;

static DWORD WINAPI LoadJPGThread(LPVOID lpParam) {
	LoadJPGParams *p = (LoadJPGParams*)lpParam;
	p->result = LoadJPG_orig(p->filename, p->opts, p->errbuf, p->grayscale, p->dummy);
	return 0;
}

EXPORT
HANDLE LoadJPG(const wchar_t *filename, jpegqs_control_t *opts, wchar_t *errbuf, int grayscale, void *dummy) {
	LoadJPGParams p = { filename, opts, errbuf, grayscale, dummy, (HANDLE)NULL };
	DWORD threadId; HANDLE thread;
	LOCK
	thread = CreateThread(NULL, 0, LoadJPGThread, (LPVOID)&p, 0, &threadId);
	WaitForSingleObject(thread, INFINITE);
	CloseHandle(thread);
	UNLOCK
	return p.result;
}
#endif

EXPORT
int GetPlugInInfo(char *version, char *name) {
  sprintf(version, JPEGQS_VERSION);
  sprintf(name, "JPEG Quant Smooth");
  return 0;
}

