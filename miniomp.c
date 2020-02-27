/*
 * Lightweight replacement of libgomp with basic OpenMP functionality.
 * Copyright (C) 2020 Ilya Kurdyukov
 *
 * contains modified parts of libgomp:
 *
 * Copyright (C) 2005-2018 Free Software Foundation, Inc.
 * Contributed by Richard Henderson <rth@redhat.com>.

 * Libgomp is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * Libgomp is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * Under Section 7 of GPL version 3, you are granted additional
 * permissions described in the GCC Runtime Library Exception, version
 * 3.1, as published by the Free Software Foundation.
 *
 * You should have received a copy of the GNU General Public License and
 * a copy of the GCC Runtime Library Exception along with this program;
 * see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#ifdef _WIN32
#define HIDDEN
#else
#define HIDDEN __attribute__((visibility("hidden")))
#endif

#define LOG0(...)
#define LOG(...) { fprintf(stderr, __VA_ARGS__); fflush(stderr); }

#ifndef MAX_THREADS
#define MAX_THREADS 16
#endif

typedef char bool;
#define false 0
#define true 1

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define THREAD_CALLBACK(name) DWORD WINAPI name(LPVOID param)
#define THREAD_RET return 0;

#define THREAD_FIELDS HANDLE handle;
#define THREAD_INIT(t) t.impl.handle = NULL;
#define THREAD_CREATE(t) { DWORD tid; t.impl.handle = CreateThread(NULL, 0, gomp_threadfunc, (void*)&t, 0, &tid); }
#define THREAD_JOIN(t) if (t.impl.handle) { WaitForSingleObject(t.impl.handle, INFINITE); CloseHandle(t.impl.handle); }
#else
#include <pthread.h>
#define THREAD_CALLBACK(name) void* name(void* param)
#define THREAD_RET return NULL;

#define THREAD_FIELDS int err; pthread_t pthread;
#define THREAD_INIT(t) t.impl.err = -1;
#define THREAD_CREATE(t) t.impl.err = pthread_create(&t.impl.pthread, NULL, gomp_threadfunc, (void*)&t);
#define THREAD_JOIN(t) if (!t.impl.err) pthread_join(t.impl.pthread, NULL);
#endif

typedef unsigned long long gomp_ull;

typedef struct {
	void (*fn)(void*);
	void *data;
	unsigned nthreads;
	volatile int wait; // waiting threads
	volatile char lock;
	char mode;
	union { volatile gomp_ull next_ull; volatile long next; };
	union { gomp_ull end_ull; long end; };
	union { gomp_ull chunk_ull; long chunk; };
} gomp_work_t;

typedef struct {
	int team_id;
	gomp_work_t *work_share;
	struct { THREAD_FIELDS } impl;
} gomp_thread_t;

static int nthreads_var = 1, num_procs = 1;

static gomp_work_t gomp_work_default = { NULL, NULL, 1, -1, 0, 0, { 0 }, { 0 }, { 0 } };
static gomp_thread_t gomp_thread_default = { 0, &gomp_work_default, { 0 } };

static inline void gomp_init_num_threads() {
#ifdef _WIN32
	DWORD_PTR procMask, sysMask;
	int count = 0;
	if (GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask)) {
		if (procMask) do count++; while (procMask &= procMask - 1);
	}
#else
	int count = sysconf(_SC_NPROCESSORS_ONLN);
#endif
	if (count < 1) count = 1;
	num_procs = nthreads_var = count;
}

#ifdef _WIN32
typedef CRITICAL_SECTION gomp_mutex_t;
static gomp_mutex_t default_lock;
static gomp_mutex_t barrier_lock;
#define MUTEX_INIT(lock) InitializeCriticalSection(lock);
#define MUTEX_FREE(lock) DeleteCriticalSection(lock);

static void gomp_mutex_lock(gomp_mutex_t *lock) {
	EnterCriticalSection(lock);
}

static void gomp_mutex_unlock(gomp_mutex_t *lock) {
	LeaveCriticalSection(lock);
}
#else
typedef pthread_mutex_t gomp_mutex_t;
static gomp_mutex_t default_lock = PTHREAD_MUTEX_INITIALIZER;
static gomp_mutex_t barrier_lock = PTHREAD_MUTEX_INITIALIZER;
// #define MUTEX_INIT(lock) pthread_mutex_init(lock, NULL)
// #define MUTEX_FREE(lock) pthread_mutex_destroy(lock)

static void gomp_mutex_lock(gomp_mutex_t *lock) {
	pthread_mutex_lock(lock);
}

static void gomp_mutex_unlock(gomp_mutex_t *lock) {
	pthread_mutex_unlock(lock);
}
#endif

#if 1 && defined(_WIN32)
static DWORD omp_tls = TLS_OUT_OF_INDEXES;
#define TLS_INIT omp_tls = TlsAlloc();
#define TLS_FREE if (omp_tls != TLS_OUT_OF_INDEXES) TlsFree(omp_tls);

static inline gomp_thread_t *miniomp_gettls() {
	void *data = NULL;
	if (omp_tls != TLS_OUT_OF_INDEXES) data = TlsGetValue(omp_tls);
	return data ? (gomp_thread_t*)data : &gomp_thread_default;
}

static inline void miniomp_settls(gomp_thread_t *data) {
	if (omp_tls != TLS_OUT_OF_INDEXES) TlsSetValue(omp_tls, data);
}

#define TLS_SET(x) miniomp_settls(x);
#define TLS_GET miniomp_gettls()
#else

static __thread gomp_thread_t *gomp_ptr = &gomp_thread_default;

#define TLS_SET(x) gomp_ptr = x;
#define TLS_GET gomp_ptr
#endif

__attribute__((constructor))
static void miniomp_init() {
	// LOG("miniomp_init()\n");
#ifdef TLS_INIT
	TLS_INIT
#endif
#ifdef MUTEX_INIT
	MUTEX_INIT(&default_lock)
	MUTEX_INIT(&barrier_lock)
#endif
	gomp_init_num_threads();
}

__attribute__((destructor))
static void miniomp_deinit() {
	// LOG("miniomp_deinit()\n");
#ifdef MUTEX_FREE
	MUTEX_FREE(&barrier_lock)
	MUTEX_FREE(&default_lock)
#endif
#ifdef TLS_FREE
	TLS_FREE
#endif
}

static inline gomp_thread_t* gomp_thread() { return TLS_GET; }
HIDDEN int omp_get_thread_num() { return gomp_thread()->team_id; }
HIDDEN int omp_get_num_procs() { return num_procs; }
HIDDEN int omp_get_max_threads() { return nthreads_var; }
HIDDEN int omp_get_num_threads() { return gomp_thread()->work_share->nthreads; }
HIDDEN void omp_set_num_threads(int n) { nthreads_var = n > 0 ? n : 1; }

// need for ULL loops with negative increment ending at zero
// and some other extreme cases
#ifndef OVERFLOW_CHECKS
#define OVERFLOW_CHECKS 1
#endif

#define DYN_NEXT(name, type, next, chunk1, end1) \
static bool gomp_iter_##name(gomp_thread_t *thr, type *pstart, type *pend) { \
	gomp_work_t *ws = thr->work_share; char mode = ws->mode; \
	type end = ws->end1, nend, chunk = ws->chunk1, start; \
	if (!OVERFLOW_CHECKS || mode & 1) { \
		start = __sync_fetch_and_add(&ws->next, chunk); \
		if (!(mode & 2)) { \
			if (start >= end) return false; \
			nend = start + chunk; \
			if (nend > end) nend = end; \
		} else { \
			if (start <= end) return false; \
			nend = start + chunk; \
			if (nend < end) nend = end; \
		} \
	} else { \
		start = ws->next; \
		for (;;) { \
			type left = end - start, tmp; \
			if (!left) return false; \
			if (mode & 2) { if (chunk < left) chunk = left; } \
			else if (chunk > left) chunk = left;  \
			nend = start + chunk; \
			tmp = __sync_val_compare_and_swap(&ws->next, start, nend); \
			if (tmp == start) break; \
			start = tmp; \
		} \
	} \
	*pstart = start; *pend = nend; return true; \
}
DYN_NEXT(dynamic_next, long, next, chunk, end)
DYN_NEXT(ull_dynamic_next, gomp_ull, next_ull, chunk_ull, end_ull)
#undef DYN_NEXT

static THREAD_CALLBACK(gomp_threadfunc) {
	gomp_thread_t *thr = (gomp_thread_t*)param;
	gomp_work_t *ws = thr->work_share;
	TLS_SET(thr)
	ws->fn(ws->data);
	THREAD_RET
}

#if 0
#include <alloca.h>
#define THREADS_DEFINE gomp_thread_t *threads;
#define THREADS_ALLOC threads = (gomp_thread_t*)alloca(nthreads * sizeof(gomp_thread_t));
#else
#define THREADS_DEFINE gomp_thread_t threads[MAX_THREADS];
#define THREADS_ALLOC
#endif

#define TEAM_INIT \
	LOG0("parallel: fn = %p, data = %p, nthreads = %u, flags = %u\n", fn, data, nthreads, flags) \
	gomp_work_t ws; THREADS_DEFINE \
	ws.fn = fn; ws.data = data; \
	if (nthreads <= 0) nthreads = nthreads_var; \
	if (nthreads > MAX_THREADS) nthreads = MAX_THREADS; \
	ws.nthreads = nthreads; ws.wait = -1; \
	THREADS_ALLOC \
	for (unsigned i = 0; i < nthreads; i++) { \
		THREAD_INIT(threads[i]) \
		threads[i].team_id = i; \
		threads[i].work_share = &ws; \
	} \
	TLS_SET(threads)

#define TEAM_START(ws) \
	for (unsigned i = 1; i < nthreads; i++) THREAD_CREATE(threads[i]) \
	fn(data); \
	for (unsigned i = 1; i < nthreads; i++) THREAD_JOIN(threads[i]) \
	TLS_SET(&gomp_thread_default)

#define LOOP_START(INIT) gomp_thread_t *thr = gomp_thread(); \
	gomp_work_t *ws = thr->work_share; unsigned nthreads = ws->nthreads; \
	{ char l; do l = __sync_val_compare_and_swap(&ws->lock, 0, 1); \
	while (l == 1); if (!l) { LOOP_##INIT((*ws)) ws->lock = 2; } }

#define LOOP_INIT(ws) ws.mode = incr >= 0 ? 0 : 2; \
	LOG0("loop_start: start = %li, end = %li, incr = %li, chunk = %li\n", start, end, incr, chunk) \
	end = (incr >= 0 && start > end) || (incr < 0 && start < end) ? start : end; \
	ws.next = start; ws.end = end; ws.chunk = chunk *= incr; \
	if (OVERFLOW_CHECKS) { chunk = (~0UL >> 1) - chunk * (nthreads + 1); \
		ws.mode |= incr >= 0 ? end <= chunk : end >= chunk; }

#define LOOP_ULL_INIT(ws) ws.mode = up ? 0 : 2; \
	LOG0("loop_ull_start: up = %i, start = %llu, end = %llu, incr = %lli, chunk = %llu\n", up, start, end, incr, chunk) \
	end = (up && start > end) || (!up && start < end) ? start : end; \
	ws.next_ull = start; ws.end_ull = end; ws.chunk_ull = chunk *= incr; \
	if (OVERFLOW_CHECKS) { chunk = -1 - chunk * (nthreads + 1); \
		ws.mode |= up ? end <= chunk : end >= chunk; }

// up ? end <= fe00 : end >= 0200-1

#define TEAM_ARGS void (*fn)(void*), void *data, unsigned nthreads
#define LOOP_ARGS(t) t start, t end, t incr, t chunk

HIDDEN void GOMP_parallel(TEAM_ARGS, unsigned flags) {
	(void)flags;
	TEAM_INIT ws.lock = 0;
	TEAM_START(ws)
}

HIDDEN void GOMP_parallel_loop_dynamic(TEAM_ARGS, LOOP_ARGS(long), unsigned flags) {
	(void)flags;
	TEAM_INIT LOOP_INIT(ws) ws.lock = 2;
	TEAM_START(ws)
}

HIDDEN bool GOMP_loop_dynamic_start(LOOP_ARGS(long), long *istart, long *iend) {
	LOOP_START(INIT)
	return gomp_iter_dynamic_next(thr, istart, iend);
}

HIDDEN bool GOMP_loop_ull_dynamic_start(bool up, LOOP_ARGS(gomp_ull), gomp_ull *istart, gomp_ull *iend) {
	LOOP_START(ULL_INIT)
	return gomp_iter_ull_dynamic_next(thr, istart, iend);
}

HIDDEN bool GOMP_loop_dynamic_next(long *istart, long *iend) {
	// LOG("loop_dynamic_next: istart = %p, iend = %p\n", istart, iend);
	return gomp_iter_dynamic_next(gomp_thread(), istart, iend);
}

HIDDEN bool GOMP_loop_ull_dynamic_next(gomp_ull *istart, gomp_ull *iend) {
	// LOG("loop_ull_dynamic_next: istart = %p, iend = %p\n", istart, iend);
	return gomp_iter_ull_dynamic_next(gomp_thread(), istart, iend);
}

HIDDEN void GOMP_loop_end_nowait() {
	// LOG("loop_end_nowait\n");
}

static void miniomp_barrier(int loop_end) {
	gomp_thread_t *thr = gomp_thread();
	gomp_work_t *ws = thr->work_share;
	int i, nthreads = ws->nthreads;
	if (nthreads == 1) {
		if (loop_end) ws->lock = 0;
		return;
	}
	// LOG("barrier: team_id = %i\n", thr->team_id);
	do i = __sync_val_compare_and_swap(&ws->wait, -1, 0); while (!i);
	if (i == -1) gomp_mutex_lock(&barrier_lock);
	i = __sync_add_and_fetch(&ws->wait, 1);
	if (i < nthreads) gomp_mutex_lock(&barrier_lock);
	else {
		if (loop_end) ws->lock = 0;
		ws->wait = -1;
	}
	gomp_mutex_unlock(&barrier_lock);
}

HIDDEN void GOMP_barrier() { miniomp_barrier(0); }
HIDDEN void GOMP_loop_end() { miniomp_barrier(1); }

#define M1(fn, copy) extern __typeof(fn) copy __attribute__((alias(#fn)));
M1(GOMP_parallel_loop_dynamic, GOMP_parallel_loop_guided)
M1(GOMP_parallel_loop_dynamic, GOMP_parallel_loop_nonmonotonic_dynamic)
M1(GOMP_parallel_loop_dynamic, GOMP_parallel_loop_nonmonotonic_guided)
M1(GOMP_loop_dynamic_start, GOMP_loop_guided_start)
M1(GOMP_loop_dynamic_next, GOMP_loop_guided_next)
M1(GOMP_loop_dynamic_start, GOMP_loop_nonmonotonic_dynamic_start)
M1(GOMP_loop_dynamic_next, GOMP_loop_nonmonotonic_dynamic_next)
M1(GOMP_loop_dynamic_start, GOMP_loop_nonmonotonic_guided_start)
M1(GOMP_loop_dynamic_next, GOMP_loop_nonmonotonic_guided_next)
M1(GOMP_loop_ull_dynamic_start, GOMP_loop_ull_guided_start)
M1(GOMP_loop_ull_dynamic_next, GOMP_loop_ull_guided_next)
M1(GOMP_loop_ull_dynamic_start, GOMP_loop_ull_nonmonotonic_dynamic_start)
M1(GOMP_loop_ull_dynamic_next, GOMP_loop_ull_nonmonotonic_dynamic_next)
M1(GOMP_loop_ull_dynamic_start, GOMP_loop_ull_nonmonotonic_guided_start)
M1(GOMP_loop_ull_dynamic_next, GOMP_loop_ull_nonmonotonic_guided_next)
#undef M1

HIDDEN void GOMP_critical_start() { gomp_mutex_lock(&default_lock); }
HIDDEN void GOMP_critical_end() { gomp_mutex_unlock(&default_lock); }

