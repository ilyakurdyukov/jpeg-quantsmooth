/*
 * Copyright (C) 2020 Kurdyukov Ilya
 *
 * JPEG Quant Smooth API definitions
 */

#ifndef JPEGQS_H
#define JPEGQS_H

enum {
	JPEGQS_ITER_SHIFT = 16,
	JPEGQS_ITER_MAX = 0xff,
	JPEGQS_FLAGS_SHIFT = 8,
	JPEGQS_FLAGS_MASK = 0xff,
	JPEGQS_DIAGONALS = 1 << 8,
	JPEGQS_JOINT_YUV = 2 << 8,
	JPEGQS_UPSAMPLE_UV = 4 << 8,
	JPEGQS_TRANSCODE = 8 << 8,
	JPEGQS_INFO_MASK = 0xff
};

#ifdef JPEGQS_ATTR
JPEGQS_ATTR
#endif
void do_quantsmooth(j_decompress_ptr srcinfo, jvirt_barray_ptr *src_coef_arrays, int32_t flags);

#endif
