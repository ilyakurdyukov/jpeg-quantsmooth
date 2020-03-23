/*
 * Copyright (C) 2020 Ilya Kurdyukov
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
	JPEGQS_LOW_QUALITY = 8,
	JPEGQS_NO_REBALANCE = 16,
	JPEGQS_NO_REBALANCE_UV = 32,
	JPEGQS_TRANSCODE = 64,
	JPEGQS_FLAGS_MASK = 0x7f,
	JPEGQS_CPU_SHIFT = 12,
	JPEGQS_CPU_MASK = 15,
	JPEGQS_INFO_SHIFT = 16,
	JPEGQS_INFO_COMP1 = 1 << JPEGQS_INFO_SHIFT,
	JPEGQS_INFO_QUANT = 2 << JPEGQS_INFO_SHIFT,
	JPEGQS_INFO_COMP2 = 4 << JPEGQS_INFO_SHIFT,
	JPEGQS_INFO_TIME = 8 << JPEGQS_INFO_SHIFT,
	JPEGQS_INFO_CPU = 16 << JPEGQS_INFO_SHIFT
};

#ifndef JPEGQS_ATTR
#define JPEGQS_ATTR
#endif

#define JPEGQS_VERSION "2020-03-23"
#define JPEGQS_COPYRIGHT "Copyright (C) 2020 Ilya Kurdyukov"

typedef struct {
	int flags, niter, threads, progprec;
	void *userdata;
	int (*progress)(void *data, int cur, int max);
} jpegqs_control_t;

JPEGQS_ATTR
int do_quantsmooth(j_decompress_ptr srcinfo, jvirt_barray_ptr *coef_arrays, jpegqs_control_t *opts);

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
