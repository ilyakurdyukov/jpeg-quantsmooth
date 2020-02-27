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

#if defined(SIMD_SELECT) && !defined(SIMD_NAME)
#ifndef JPEGQS_ATTR
#define JPEGQS_ATTR
#endif
#include "libjpegqs.h"

#define QS_ARGS (j_decompress_ptr srcinfo, jvirt_barray_ptr *coef_arrays, jpegqs_control_t *opts)

#define M1(name) int do_quantsmooth_##name QS_ARGS;
M1(avx2) M1(sse2) M1(base)
#undef M1

#ifdef _MSC_VER
#include <intrin.h>
// void __cpuidex(int cpuInfo[4], int function_id, int subfunction_id);
#define get_cpuid(a, c, out) __cpuidex(out, a, c)
#else
static inline void get_cpuid(int32_t a, int32_t c, int32_t out[4]) {
	__asm__ ("cpuid\n" : "=a"(out[0]), "=b"(out[1]),
			"=c"(out[2]), "=d"(out[3]) : "a"(a), "c"(c));
}
#endif

JPEGQS_ATTR int do_quantsmooth QS_ARGS {
	int32_t cpuid[4], m; int type = 0;
	get_cpuid(0, 0, cpuid); m = cpuid[0];
	do {
		if (m < 1) break;
		get_cpuid(1, 0, cpuid);
		if (!(cpuid[3] & (1 << 26))) break; // SSE2
		type = 1;
		if (!(cpuid[2] & (1 << 12))) break; // FMA
		if (m < 7) break;
		get_cpuid(7, 0, cpuid);
		if (!(cpuid[1] & (1 << 5))) break; // AVX2
		type = 2;
	} while (0);

#define M1(name) return do_quantsmooth_##name(srcinfo, coef_arrays, opts);
	if (type == 2) M1(avx2)
#ifndef __x86_64__
	if (type == 1) M1(sse2)
#endif
	M1(base)
#undef M1
}
#else
#define logfmt(...) fprintf(stderr, __VA_ARGS__)
#ifdef SIMD_NAME
#define QS_CONCAT(x) do_quantsmooth_##x
#define QS_NAME1(x) QS_CONCAT(x)
#define QS_NAME QS_NAME1(SIMD_NAME)
#endif
#define JPEGQS_ATTR
#if !defined(SKIP_ON_X64) || !defined(__x86_64__)
#include "quantsmooth.h"
#endif
#endif

