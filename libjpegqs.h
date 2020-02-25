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
	JPEGQS_ITER_MAX = 100,
	JPEGQS_DIAGONALS = 1,
	JPEGQS_JOINT_YUV = 2,
	JPEGQS_UPSAMPLE_UV = 4,
	JPEGQS_TRANSCODE = 8,
	JPEGQS_INFO_COMP1 = 1,
	JPEGQS_INFO_QUANT = 2,
	JPEGQS_INFO_COMP2 = 4,
	JPEGQS_INFO_TIME = 8
};

#ifndef JPEGQS_ATTR
#define JPEGQS_ATTR
#endif

#define JPEGQS_VERSION "2020-02-25"
#define JPEGQS_COPYRIGHT "Copyright (c) 2020 Ilya Kurdyukov"

typedef struct {
	int info, flags, niter, threads;
} jpegqs_control_t;

JPEGQS_ATTR
void do_quantsmooth(j_decompress_ptr srcinfo, jvirt_barray_ptr *coef_arrays, jpegqs_control_t *opts);

#ifndef TRANSCODE_ONLY
JPEGQS_ATTR
boolean jpegqs_start_decompress(j_decompress_ptr cinfo, jpegqs_control_t *opts);

JPEGQS_ATTR
boolean jpegqs_finish_decompress(j_decompress_ptr cinfo);
#endif

#ifdef __cplusplus
}
#endif
#endif
