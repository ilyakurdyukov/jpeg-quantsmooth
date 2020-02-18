/*
 * Copyright (C) 2020 Kurdyukov Ilya
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

#include "jpeglib.h"

#if defined(SIMD_SELECT) && !defined(SIMD_NAME)
#ifndef JPEGQS_ATTR
#define JPEGQS_ATTR
#endif
#include "libjpegqs.h"

#define QS_ARGS (j_decompress_ptr srcinfo, jvirt_barray_ptr *src_coef_arrays, int32_t flags)

#define M1(name) void do_quantsmooth_##name QS_ARGS;
M1(avx2)
#ifndef __x86_64__
M1(sse2)
#endif
M1(base)
#undef M1

static inline void get_cpuid(int32_t a, int32_t c, int32_t out[4]) {
	__asm__ ("cpuid\n" : "=a"(out[0]), "=b"(out[1]),
			"=d"(out[2]), "=c"(out[3]) : "a"(a), "c"(c));
}

JPEGQS_ATTR void do_quantsmooth QS_ARGS {
	int32_t cpuid[4], m; int type = 0;
	get_cpuid(0, -1, cpuid); m = cpuid[0];
	get_cpuid(1, -1, cpuid);
	if (cpuid[2] & (1 << 26)) { type = 1;
		if (m >= 7) {
			get_cpuid(7, 0, cpuid);
			if (cpuid[1] & (1 << 5)) type = 2;
		}
	}

#define M1(name) do_quantsmooth_##name(srcinfo, src_coef_arrays, flags); break;
	switch (type) {
		case 2: M1(avx2)
#ifndef __x86_64__
		case 1: M1(sse2)
#endif
		default: M1(base)
	}
#undef M1
}
#else
#define logfmt(...) fprintf(stderr, __VA_ARGS__)
#ifdef SIMD_NAME
#define QS_CONCAT(x) do_quantsmooth_##x
#define QS_NAME(x) QS_CONCAT(x)
#define do_quantsmooth QS_NAME(SIMD_NAME)
#endif
#define JPEGQS_ATTR
#include "quantsmooth.h"
#endif

