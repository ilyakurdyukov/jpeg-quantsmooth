/*
 * idct_islow AVX2 intrinsic optimization:
 * Copyright (C) 2016-2020 Kurdyukov Ilya
 *
 * contains modified parts of libjpeg:
 * Copyright (C) 1991-1998, Thomas G. Lane
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

static const char jpeg_natural_order[DCTSIZE2] = {
	 0,  1,  8, 16,  9,  2,  3, 10,
	17, 24, 32, 25, 18, 11,  4,  5,
	12, 19, 26, 33, 40, 48, 41, 34,
	27, 20, 13,  6,  7, 14, 21, 28,
	35, 42, 49, 56, 57, 50, 43, 36,
	29, 22, 15, 23, 30, 37, 44, 51,
	58, 59, 52, 45, 38, 31, 39, 46,
	53, 60, 61, 54, 47, 55, 62, 63
};

static JSAMPLE range_limit_static[5 * (MAXJSAMPLE+1) + CENTERJSAMPLE];

static void range_limit_init() {
	int i;
	JSAMPLE *table = range_limit_static;

	memset(table, 0, (MAXJSAMPLE+1) * sizeof(JSAMPLE));
	table += (MAXJSAMPLE+1);
	for (i = 0; i <= MAXJSAMPLE; i++) table[i] = i;
	table += CENTERJSAMPLE;       /* Point to where post-IDCT table starts */
	/* End of simple table, rest of first half of post-IDCT table */
	for (i = CENTERJSAMPLE; i < 2*(MAXJSAMPLE+1); i++) table[i] = MAXJSAMPLE;
	/* Second half of post-IDCT table */
	memset(table + 2 * (MAXJSAMPLE+1), 0, (2 * (MAXJSAMPLE+1) - CENTERJSAMPLE) * sizeof(JSAMPLE));
	memcpy(table + 4 * (MAXJSAMPLE+1) - CENTERJSAMPLE, table - CENTERJSAMPLE, CENTERJSAMPLE * sizeof(JSAMPLE));
}

#define CONST_BITS  13
#define PASS1_BITS  2

#if CONST_BITS == 13
#define FIX_0_298631336  2446        /* FIX(0.298631336) */
#define FIX_0_390180644  3196        /* FIX(0.390180644) */
#define FIX_0_541196100  4433        /* FIX(0.541196100) */
#define FIX_0_765366865  6270        /* FIX(0.765366865) */
#define FIX_0_899976223  7373        /* FIX(0.899976223) */
#define FIX_1_175875602  9633        /* FIX(1.175875602) */
#define FIX_1_501321110  12299       /* FIX(1.501321110) */
#define FIX_1_847759065  15137       /* FIX(1.847759065) */
#define FIX_1_961570560  16069       /* FIX(1.961570560) */
#define FIX_2_053119869  16819       /* FIX(2.053119869) */
#define FIX_2_562915447  20995       /* FIX(2.562915447) */
#define FIX_3_072711026  25172       /* FIX(3.072711026) */
#else
#define FIX_0_298631336  FIX(0.298631336)
#define FIX_0_390180644  FIX(0.390180644)
#define FIX_0_541196100  FIX(0.541196100)
#define FIX_0_765366865  FIX(0.765366865)
#define FIX_0_899976223  FIX(0.899976223)
#define FIX_1_175875602  FIX(1.175875602)
#define FIX_1_501321110  FIX(1.501321110)
#define FIX_1_847759065  FIX(1.847759065)
#define FIX_1_961570560  FIX(1.961570560)
#define FIX_2_053119869  FIX(2.053119869)
#define FIX_2_562915447  FIX(2.562915447)
#define FIX_3_072711026  FIX(3.072711026)
#endif

#define DESCALE(x,n)  (((x) + (1 << ((n)-1))) >> (n))
#define RANGE_MASK  (MAXJSAMPLE * 4 + 3) /* 2 bits wider than legal samples */

#define MULTIPLY(var,const) ((var) * (const))

static void idct_islow(JCOEFPTR coef_block, JSAMPROW outptr, JDIMENSION stride) {

#define M3 \
	z2 = M1(2); z3 = M1(6); \
	z1 = MUL(ADD(z2, z3), SET1(FIX_0_541196100)); \
	tmp2 = ADD(z1, MUL(z3, SET1(- FIX_1_847759065))); \
	tmp3 = ADD(z1, MUL(z2, SET1(FIX_0_765366865))); \
	z2 = M1(0); z3 = M1(4); \
	tmp0 = SHL(ADD(z2, z3), CONST_BITS); \
	tmp1 = SHL(SUB(z2, z3), CONST_BITS); \
	tmp10 = ADD(tmp0, tmp3); \
	tmp13 = SUB(tmp0, tmp3); \
	tmp11 = ADD(tmp1, tmp2); \
	tmp12 = SUB(tmp1, tmp2); \
\
	tmp0 = M1(7); tmp1 = M1(5); tmp2 = M1(3); tmp3 = M1(1); \
	z1 = ADD(tmp0, tmp3); \
	z2 = ADD(tmp1, tmp2); \
	z3 = ADD(tmp0, tmp2); \
	z4 = ADD(tmp1, tmp3); \
	z5 = MUL(ADD(z3, z4), SET1(FIX_1_175875602)); /* sqrt(2) * c3 */ \
\
	tmp0 = MUL(tmp0, SET1(FIX_0_298631336)); /* sqrt(2) * (-c1+c3+c5-c7) */ \
	tmp1 = MUL(tmp1, SET1(FIX_2_053119869)); /* sqrt(2) * ( c1+c3-c5+c7) */ \
	tmp2 = MUL(tmp2, SET1(FIX_3_072711026)); /* sqrt(2) * ( c1+c3+c5-c7) */ \
	tmp3 = MUL(tmp3, SET1(FIX_1_501321110)); /* sqrt(2) * ( c1+c3-c5-c7) */ \
	z1 = MUL(z1, SET1(- FIX_0_899976223)); /* sqrt(2) * (c7-c3) */ \
	z2 = MUL(z2, SET1(- FIX_2_562915447)); /* sqrt(2) * (-c1-c3) */ \
	z3 = MUL(z3, SET1(- FIX_1_961570560)); /* sqrt(2) * (-c3-c5) */ \
	z4 = MUL(z4, SET1(- FIX_0_390180644)); /* sqrt(2) * (c5-c3) */ \
	z3 = ADD(z3, z5); \
	z4 = ADD(z4, z5); \
	tmp0 = ADD(tmp0, ADD(z1, z3)); \
	tmp1 = ADD(tmp1, ADD(z2, z4)); \
	tmp2 = ADD(tmp2, ADD(z2, z3)); \
	tmp3 = ADD(tmp3, ADD(z1, z4)); \
\
	M2(0, ADD(tmp10, tmp3)) \
	M2(7, SUB(tmp10, tmp3)) \
	M2(1, ADD(tmp11, tmp2)) \
	M2(6, SUB(tmp11, tmp2)) \
	M2(2, ADD(tmp12, tmp1)) \
	M2(5, SUB(tmp12, tmp1)) \
	M2(3, ADD(tmp13, tmp0)) \
	M2(4, SUB(tmp13, tmp0))

#if 1 && defined(USE_AVX2)
	__m256i v0, v1, v2, v3, v4, v5, v6, v7, t0, t1, x0, x1, x2, x3, x4, x5, x6, x7;
	__m256i tmp0, tmp1, tmp2, tmp3;
	__m256i tmp10, tmp11, tmp12, tmp13;
	__m256i z1, z2, z3, z4, z5;

#define ADD _mm256_add_epi32
#define SUB _mm256_sub_epi32
#define MUL _mm256_mullo_epi32
#define SET1 _mm256_set1_epi32
#define SHL _mm256_slli_epi32

#define M1(i) _mm256_cvtepi16_epi32(_mm_loadu_si128((void*)&coef_block[DCTSIZE*i]))
#define M2(i, tmp) x##i = _mm256_srai_epi32(ADD(tmp, t0), CONST_BITS-PASS1_BITS);
	t0 = SET1(1 << (CONST_BITS-PASS1_BITS-1));
	M3
#undef M1
#undef M2

#define M2(v0, v1, v2, v3, k) \
v0 = _mm256_permute2x128_si256(x0, x4, k); \
v1 = _mm256_permute2x128_si256(x1, x5, k); \
v2 = _mm256_permute2x128_si256(x2, x6, k); \
v3 = _mm256_permute2x128_si256(x3, x7, k);
	M2(v0, v1, v2, v3, 0x20)
	M2(v4, v5, v6, v7, 0x31)
#undef M2
#define M4(v0, v1, v2, v3, x0, x1, x2, x3) \
t0 = _mm256_unpacklo_epi32(v0, v2); \
t1 = _mm256_unpacklo_epi32(v1, v3); \
x0 = _mm256_unpacklo_epi32(t0, t1); \
x1 = _mm256_unpackhi_epi32(t0, t1); \
t0 = _mm256_unpackhi_epi32(v0, v2); \
t1 = _mm256_unpackhi_epi32(v1, v3); \
x2 = _mm256_unpacklo_epi32(t0, t1); \
x3 = _mm256_unpackhi_epi32(t0, t1);
	M4(v0, v1, v2, v3, x0, x1, x2, x3)
	M4(v4, v5, v6, v7, x4, x5, x6, x7)

#define M1(i) x##i
#define M2(i, tmp) v##i = _mm256_srai_epi32(ADD(tmp, t0), (CONST_BITS+PASS1_BITS+3));
	t0 = SET1((256+1) << (CONST_BITS+PASS1_BITS+3-1));
	M3
#undef M1
#undef M2

	M4(v0, v1, v2, v3, x0, x1, x2, x3)
	M4(v4, v5, v6, v7, x4, x5, x6, x7)
#undef M4

	x0 = _mm256_packs_epi32(x0, x1);
	x1 = _mm256_packs_epi32(x2, x3);
	x4 = _mm256_packs_epi32(x4, x5);
	x5 = _mm256_packs_epi32(x6, x7);
	x0 = _mm256_packus_epi16(x0, x1);
	x4 = _mm256_packus_epi16(x4, x5);
	v0 = _mm256_unpacklo_epi32(x0, x4);
	v1 = _mm256_unpackhi_epi32(x0, x4);
#define M1(i, v0, j) *(int64_t*)&outptr[stride*i] = _mm256_extract_epi64(v0, j);
	M1(0, v0, 0) M1(1, v0, 1) M1(2, v1, 0) M1(3, v1, 1)
	M1(4, v0, 2) M1(5, v0, 3) M1(6, v1, 2) M1(7, v1, 3)
#undef M1
#else
	int32_t tmp0, tmp1, tmp2, tmp3;
	int32_t tmp10, tmp11, tmp12, tmp13;
	int32_t z1, z2, z3, z4, z5;
	JCOEFPTR inptr;
	JSAMPLE *range_limit = range_limit_static + (MAXJSAMPLE+1) + CENTERJSAMPLE;
	int ctr;
	int32_t *wsptr, workspace[DCTSIZE2];	/* buffers data between passes */

#define ADD(a, b) ((a) + (b))
#define SUB(a, b) ((a) - (b))
#define MUL(a, b) ((a) * (b))
#define SET1(a) (a)
#define SHL(a, b) ((a) << (b))

	/* Pass 1: process columns from input, store into work array. */
	/* Note results are scaled up by sqrt(8) compared to a true IDCT; */
	/* furthermore, we scale the results by 2**PASS1_BITS. */

#define M1(i) inptr[DCTSIZE*i]
#define M2(i, tmp) wsptr[DCTSIZE*i] = DESCALE(tmp, CONST_BITS-PASS1_BITS);
	inptr = coef_block;
	wsptr = workspace;
	for (ctr = DCTSIZE; ctr > 0; ctr--, inptr++, wsptr++) {
		/* Due to quantization, we will usually find that many of the input
		 * coefficients are zero, especially the AC terms.  We can exploit this
		 * by short-circuiting the IDCT calculation for any column in which all
		 * the AC terms are zero.  In that case each output is equal to the
		 * DC coefficient (with scale factor as needed).
		 * With typical images and quantization tables, half or more of the
		 * column DCT calculations can be simplified this way.
		 */

		if (!(inptr[DCTSIZE*1] | inptr[DCTSIZE*2] | inptr[DCTSIZE*3] | inptr[DCTSIZE*4] |
				inptr[DCTSIZE*5] | inptr[DCTSIZE*6] | inptr[DCTSIZE*7])) {
			/* AC terms all zero */
			int dcval = SHL(M1(0), PASS1_BITS);
			wsptr[DCTSIZE*0] = dcval;
			wsptr[DCTSIZE*1] = dcval;
			wsptr[DCTSIZE*2] = dcval;
			wsptr[DCTSIZE*3] = dcval;
			wsptr[DCTSIZE*4] = dcval;
			wsptr[DCTSIZE*5] = dcval;
			wsptr[DCTSIZE*6] = dcval;
			wsptr[DCTSIZE*7] = dcval;
			continue;
		}

		M3
	}
#undef M1
#undef M2

	/* Pass 2: process rows from work array, store into output array. */
	/* Note that we must descale the results by a factor of 8 == 2**3, */
	/* and also undo the PASS1_BITS scaling. */

#define M1(i) wsptr[i]
#define M2(i, tmp) outptr[i] = range_limit[DESCALE(tmp, CONST_BITS+PASS1_BITS+3) & RANGE_MASK];
	wsptr = workspace;
	for (ctr = 0; ctr < DCTSIZE; ctr++, wsptr += DCTSIZE, outptr += stride) {
		/* Rows of zeroes can be exploited in the same way as we did with columns.
		 * However, the column calculation has created many nonzero AC terms, so
		 * the simplification applies less often (typically 5% to 10% of the time).
		 * On machines with very fast multiplication, it's possible that the
		 * test takes more time than it's worth.  In that case this section
		 * may be commented out.
		 */

#ifndef NO_ZERO_ROW_TEST
		if (!(wsptr[1] | wsptr[2] | wsptr[3] | wsptr[4] | wsptr[5] | wsptr[6] | wsptr[7])) {
			/* AC terms all zero */
			JSAMPLE dcval = range_limit[DESCALE(wsptr[0], PASS1_BITS+3) & RANGE_MASK];

			outptr[0] = dcval;
			outptr[1] = dcval;
			outptr[2] = dcval;
			outptr[3] = dcval;
			outptr[4] = dcval;
			outptr[5] = dcval;
			outptr[6] = dcval;
			outptr[7] = dcval;
			continue;
		}
#endif

		M3
	}
#undef M1
#undef M2
#endif

#undef M3
#undef ADD
#undef SUB
#undef MUL
#undef SET1
#undef SHL
}

static void idct_fslow(JCOEFPTR inptr, float *outptr) {
	float tmp0, tmp1, tmp2, tmp3;
	float tmp10, tmp11, tmp12, tmp13;
	float z1, z2, z3, z4, z5;
	float *wsptr, workspace[DCTSIZE2];
	int ctr;

#define M3(inc1, inc2) \
	wsptr = workspace; \
	for (ctr = 0; ctr < DCTSIZE; ctr++, inc1, inc2) { \
		z2 = M1(2); z3 = M1(6); \
		z1 = (z2 + z3) * 0.541196100f; \
		tmp2 = z1 - z3 * 1.847759065f; \
		tmp3 = z1 + z2 * 0.765366865f; \
		z2 = M1(0); z3 = M1(4); \
		tmp0 = z2 + z3; \
		tmp1 = z2 - z3; \
		tmp10 = tmp0 + tmp3; \
		tmp13 = tmp0 - tmp3; \
		tmp11 = tmp1 + tmp2; \
		tmp12 = tmp1 - tmp2; \
		tmp0 = M1(7); tmp1 = M1(5); tmp2 = M1(3); tmp3 = M1(1); \
		z1 = tmp0 + tmp3; \
		z2 = tmp1 + tmp2; \
		z3 = tmp0 + tmp2; \
		z4 = tmp1 + tmp3; \
		z5 = (z3 + z4) * 1.175875602f; /* sqrt(2) * c3 */ \
		tmp0 *= 0.298631336f; /* sqrt(2) * (-c1+c3+c5-c7) */ \
		tmp1 *= 2.053119869f; /* sqrt(2) * ( c1+c3-c5+c7) */ \
		tmp2 *= 3.072711026f; /* sqrt(2) * ( c1+c3+c5-c7) */ \
		tmp3 *= 1.501321110f; /* sqrt(2) * ( c1+c3-c5-c7) */ \
		z1 *= -0.899976223f; /* sqrt(2) * (c7-c3) */ \
		z2 *= -2.562915447f; /* sqrt(2) * (-c1-c3) */ \
		z3 *= -1.961570560f; /* sqrt(2) * (-c3-c5) */ \
		z4 *= -0.390180644f; /* sqrt(2) * (c5-c3) */ \
		z3 += z5; z4 += z5; \
		tmp0 += z1 + z3; \
		tmp1 += z2 + z4; \
		tmp2 += z2 + z3; \
		tmp3 += z1 + z4; \
		M2(0, tmp10 + tmp3) \
		M2(7, tmp10 - tmp3) \
		M2(1, tmp11 + tmp2) \
		M2(6, tmp11 - tmp2) \
		M2(2, tmp12 + tmp1) \
		M2(5, tmp12 - tmp1) \
		M2(3, tmp13 + tmp0) \
		M2(4, tmp13 - tmp0) \
	}
#define M1(i) inptr[DCTSIZE*i]
#define M2(i, tmp) wsptr[DCTSIZE*i] = tmp;
	M3(inptr++, wsptr++)
#undef M1
#undef M2
#define M1(i) wsptr[i]
#define M2(i, tmp) outptr[i] = (tmp) * 0.125f;
	M3(wsptr += DCTSIZE, outptr += DCTSIZE)
#undef M1
#undef M2
#undef M3
}

