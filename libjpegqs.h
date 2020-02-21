/*
 * Copyright (C) 2020 Kurdyukov Ilya
 *
 * JPEG Quant Smooth API definitions
 */

#ifndef JPEGQS_H
#define JPEGQS_H

#ifdef __cplusplus
extern "C" {
#endif

enum {
	JPEGQS_ITER_SHIFT = 16,
	JPEGQS_ITER_MAX = 0xff,

	JPEGQS_FLAGS_SHIFT = 8,
	JPEGQS_FLAGS_MASK = 0xff,
	JPEGQS_DIAGONALS = 1 << JPEGQS_FLAGS_SHIFT,
	JPEGQS_JOINT_YUV = 2 << JPEGQS_FLAGS_SHIFT,
	JPEGQS_UPSAMPLE_UV = 4 << JPEGQS_FLAGS_SHIFT,
	JPEGQS_TRANSCODE = 8 << JPEGQS_FLAGS_SHIFT,

	JPEGQS_INFO_SHIFT = 0,
	JPEGQS_INFO_MASK = 0xff,
	JPEGQS_INFO_COMP1 = 1 << JPEGQS_INFO_SHIFT,
	JPEGQS_INFO_QUANT = 2 << JPEGQS_INFO_SHIFT,
	JPEGQS_INFO_COMP2 = 4 << JPEGQS_INFO_SHIFT,
	JPEGQS_INFO_TIME = 8 << JPEGQS_INFO_SHIFT
};

#ifndef JPEGQS_ATTR
#define JPEGQS_ATTR
#endif

#define JPEGQS_VERSION "2020-02-21"
#define JPEGQS_COPYRIGHT "Copyright (c) 2020 Ilya Kurdyukov"

JPEGQS_ATTR
void do_quantsmooth(j_decompress_ptr srcinfo, jvirt_barray_ptr *src_coef_arrays, int32_t flags);

#ifndef TRANSCODE_ONLY
JPEGQS_ATTR
boolean jpegqs_start_decompress(j_decompress_ptr cinfo, int32_t control);

JPEGQS_ATTR
boolean jpegqs_finish_decompress(j_decompress_ptr cinfo);
#endif

#ifdef __cplusplus
}
#endif
#endif
