/*
 * Copyright (C) 2020 Ilya Kurdyukov
 *
 * This file is part of jpeg quantsmooth (library)
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

#if !defined(TRANSCODE_ONLY) && defined(WITH_JPEGSRC)
#define JPEG_INTERNALS
#endif
#include "jpeglib.h"

#define logfmt(...) fprintf(stderr, __VA_ARGS__)

#if defined(SIMD_SELECT) && !defined(SIMD_NAME)
#ifndef JPEGQS_ATTR
#define JPEGQS_ATTR
#endif
#include "libjpegqs.h"

#define QS_ARGS (j_decompress_ptr srcinfo, jvirt_barray_ptr *coef_arrays, jpegqs_control_t *opts)

#define M1(name) int do_quantsmooth_##name QS_ARGS;
M1(base) M1(sse2) M1(avx2) M1(avx512)
#undef M1

#ifdef _MSC_VER
#include <intrin.h>
// void __cpuidex(int cpuInfo[4], int function_id, int subfunction_id);
// unsigned __int64 _xgetbv(unsigned int);
#define get_cpuid(a, c, out) __cpuidex(out, a, c)
#define xgetbv(n) _xgetbv(n)
#else
static inline void get_cpuid(int32_t a, int32_t c, int32_t out[4]) {
	__asm__ ("cpuid\n" : "=a"(out[0]), "=b"(out[1]),
			"=c"(out[2]), "=d"(out[3]) : "a"(a), "c"(c));
}

static inline int64_t xgetbv(int32_t n) {
	uint32_t eax, edx;
	__asm__ ("xgetbv\n" : "=a"(eax), "=d"(edx) : "c"(n));
	return ((int64_t)edx << 32) | eax;
}
#endif

JPEGQS_ATTR int do_quantsmooth QS_ARGS {
	int type = 1;
	do {
		int32_t cpuid[4], m, xcr0;
		get_cpuid(0, 0, cpuid); m = cpuid[0];
		if (m < 1) break;
		get_cpuid(1, 0, cpuid);
		if (!(cpuid[3] & (1 << 26))) break; // SSE2
		type = 2;
		// VirtualBox clears FMA, even if AVX2 is set
		// if (!(cpuid[2] & (1 << 12))) break; // FMA
		if (!(cpuid[2] & (1 << 27))) break; // OSXSAVE
		xcr0 = ~xgetbv(0);
		if (m < 7) break;
		get_cpuid(7, 0, cpuid);
		if (!(cpuid[1] & (1 << 5)) && xcr0 & 6) break; // AVX2
		type = 3;
		if (!(cpuid[1] & (1 << 16)) && xcr0 & 0xe6) break; // AVX512F
		type = 4;
	} while (0);

	{
		int x = (opts->flags >> JPEGQS_CPU_SHIFT) & JPEGQS_CPU_MASK;
		if (x) type = x < type ? x : type;
	}

#ifdef WITH_LOG
	if (opts->flags & JPEGQS_INFO_CPU) {
		logfmt("SIMD type: %i\n", type);
	}
#endif

#define M1(name) return do_quantsmooth_##name(srcinfo, coef_arrays, opts);
#ifdef __x86_64__
	if (type >= 4) M1(avx512)
#endif
	if (type >= 3) M1(avx2)
#ifndef __x86_64__
	if (type >= 2) M1(sse2)
#endif
	M1(base)
#undef M1
}
#else
#ifdef SIMD_NAME
#define QS_CONCAT(x) do_quantsmooth_##x
#define QS_NAME1(x) QS_CONCAT(x)
#define QS_NAME QS_NAME1(SIMD_NAME)
#ifndef SIMD_BASE
#define NO_HELPERS
#endif
#endif
#define JPEGQS_ATTR
#if !(defined(SIMD_SSE2) && defined(__x86_64__)) && \
	!(defined(SIMD_AVX512) && !defined(__x86_64__))
#include "quantsmooth.h"
#endif
#endif

