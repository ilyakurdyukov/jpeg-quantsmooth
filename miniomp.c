/*
 * Copyright (C) 2020 Kurdyukov Ilya
 *
 * Lightweight replacement of libgomp with basic OpenMP functionality.
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

#define LOG(...) fprintf(stderr, __VA_ARGS__)

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
#define THREAD_INIT(t) t.handle = NULL;
#define THREAD_CREATE(t) { DWORD tid; t.handle = CreateThread(NULL, 0, gomp_threadfunc, (void*)&t, 0, &tid); }
#define THREAD_JOIN(t) if (t.handle) { WaitForSingleObject(t.handle, INFINITE); CloseHandle(t.handle); }
#else
#include <pthread.h>
#define THREAD_CALLBACK(name) void* name(void* param)
#define THREAD_RET return NULL;

#define THREAD_FIELDS int err; pthread_t pthread;
#define THREAD_INIT(t) t.err = -1;
#define THREAD_CREATE(t) t.err = pthread_create(&t.pthread, NULL, gomp_threadfunc, (void*)&t);
#define THREAD_JOIN(t) if (!t.err) pthread_join(t.pthread, NULL);
#endif

typedef unsigned long long gomp_ull;

typedef struct {
	void (*fn) (void *);
	void *data;
	unsigned num_threads;
	union { long next; gomp_ull ull_next; };
	union { long end; gomp_ull ull_end; };
	union { long incr; gomp_ull ull_incr; };
	union { long chunk_size; gomp_ull ull_chunk_size; };
} gomp_work_t;

typedef struct {
	struct {
		int team_id;
		gomp_work_t *work_share;
	} ts;
	THREAD_FIELDS
} gomp_thread_t;

static int nthreads_var = MAX_THREADS;
static int num_procs = MAX_THREADS;

static gomp_thread_t gomp_thread_default = { 0 };

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
static CRITICAL_SECTION CriticalSection;
#define MUTEX_INIT InitializeCriticalSection(&CriticalSection);
#define MUTEX_FREE DeleteCriticalSection(&CriticalSection);
#endif

#if 1 && defined(_WIN32)
static DWORD omp_tls = TLS_OUT_OF_INDEXES;
#define TLS_INIT omp_tls = TlsAlloc();
#define TLS_FREE if (omp_tls != TLS_OUT_OF_INDEXES) TlsFree(omp_tls);

static inline gomp_thread_t *miniomp_gettls() {
	void *data = NULL;
	if (omp_tls != TLS_OUT_OF_INDEXES) data = TlsGetValue(omp_tls);
	if (!data) data = &gomp_thread_default;
	return (gomp_thread_t*)data;
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
	MUTEX_INIT
#endif
	gomp_init_num_threads();
}

__attribute__((destructor))
static void miniomp_deinit() {
	// LOG("miniomp_deinit()\n");
#ifdef MUTEX_FREE
	MUTEX_FREE
#endif
#ifdef TLS_FREE
	TLS_FREE
#endif
}

static inline gomp_thread_t* gomp_thread() {
	return TLS_GET;
}

HIDDEN int omp_get_thread_num(void) {
	return gomp_thread()->ts.team_id;
}

HIDDEN int omp_get_num_procs(void) {
	return num_procs;
}

HIDDEN int omp_get_max_threads(void) {
	return nthreads_var;
}

HIDDEN void omp_set_num_threads(int n) {
	nthreads_var = n > 0 ? n : 1;
}

static bool gomp_iter_dynamic_next(long *pstart, long *pend) {
	gomp_thread_t *thr = gomp_thread();
	gomp_work_t *ws = thr->ts.work_share;
	long end = ws->end, nend, chunk = ws->chunk_size, incr = ws->incr;
	long tmp = __sync_fetch_and_add(&ws->next, chunk);
	if (incr > 0) {
		if (tmp >= end) return false;
		nend = tmp + chunk;
		if (nend > end) nend = end;
	} else {
		if (tmp <= end) return false;
		nend = tmp + chunk;
		if (nend < end) nend = end;
	}
	*pstart = tmp;
	*pend = nend;
	return true;
}

static bool gomp_iter_ull_dynamic_next(gomp_ull *pstart, gomp_ull *pend) {
	gomp_thread_t *thr = gomp_thread();
	gomp_work_t *ws = thr->ts.work_share;
	gomp_ull end = ws->ull_end, nend, chunk = ws->ull_chunk_size, incr = ws->ull_incr;
	gomp_ull tmp = __sync_fetch_and_add(&ws->ull_next, chunk);
	if (incr > 0) {
		if (tmp >= end) return false;
		nend = tmp + chunk;
		if (nend > end) nend = end;
	} else {
		if (tmp <= end) return false;
		nend = tmp + chunk;
		if (nend < end) nend = end;
	}
	*pstart = tmp;
	*pend = nend;
	return true;
}

static THREAD_CALLBACK(gomp_threadfunc) {
	gomp_thread_t *thr = (gomp_thread_t*)param;
	gomp_work_t *ws = thr->ts.work_share;

	TLS_SET((gomp_thread_t*)param)

	ws->fn(ws->data);
	THREAD_RET
}

#if 0
#include <alloca.h>
#define THREADS_DEFINE gomp_thread_t *threads;
#define THREADS_ALLOC threads = (gomp_thread_t*)alloca(num_threads * sizeof(gomp_thread_t));
#else
#define THREADS_DEFINE gomp_thread_t threads[MAX_THREADS];
#define THREADS_ALLOC
#endif

#define GOMP_TEAM_INIT \
	gomp_work_t ws; \
	THREADS_DEFINE \
	ws.fn = fn; ws.data = data; \
	if (num_threads <= 0) num_threads = nthreads_var; \
	if (num_threads > MAX_THREADS) num_threads = MAX_THREADS; \
	THREADS_ALLOC \
	for (unsigned i = 0; i < num_threads; i++) { \
		THREAD_INIT(threads[i]) \
		threads[i].ts.team_id = i; \
		threads[i].ts.work_share = &ws; \
	} \
	TLS_SET(threads)

#define GOMP_TEAM_START(ws) \
	ws.num_threads = 0; \
	ws.next = start; ws.end = end; ws.incr = incr; \
	ws.chunk_size = chunk_size * incr; \
	for (unsigned i = 1; i < num_threads; i++) THREAD_CREATE(threads[i])

#define GOMP_TEAM_ULL_START(ws) \
	(void)up; /* TODO */ \
	ws.num_threads = 0; \
	ws.ull_next = start; ws.ull_end = end; ws.ull_incr = incr; \
	ws.ull_chunk_size = chunk_size * incr; \
	for (unsigned i = 1; i < num_threads; i++) THREAD_CREATE(threads[i])

#define GOMP_TEAM_END \
	for (unsigned i = 1; i < num_threads; i++) THREAD_JOIN(threads[i]) \
	TLS_SET(&gomp_thread_default)

HIDDEN void GOMP_parallel_loop_dynamic(void (*fn) (void *), void *data,
			unsigned num_threads, long start, long end,
			long incr, long chunk_size, unsigned flags) {
	(void)flags;
	// LOG("parallel: fn = %p, data = %p, num_threads = %u, flags = %u\n", fn, data, num_threads, flags);
	// LOG("loop_start: start = %li, end = %li, incr = %li, chunk_size = %li\n", start, end, incr, chunk_size);
	GOMP_TEAM_INIT
	GOMP_TEAM_START(ws)
	fn(data);
	GOMP_TEAM_END
}

HIDDEN bool GOMP_loop_dynamic_start(long start, long end, long incr, long chunk_size,
			 long *istart, long *iend) {
	gomp_thread_t *threads = TLS_GET;
	gomp_work_t *ws = threads->ts.work_share;
	unsigned num_threads = ws->num_threads;
	if (num_threads) {
		// LOG("loop_start: start = %li, end = %li, incr = %li, chunk_size = %li\n", start, end, incr, chunk_size);
		GOMP_TEAM_START((*ws))
	}
	return gomp_iter_dynamic_next(istart, iend);
}

HIDDEN bool GOMP_loop_ull_dynamic_start(bool up, gomp_ull start, gomp_ull end, gomp_ull incr, gomp_ull chunk_size,
			 gomp_ull *istart, gomp_ull *iend) {
	gomp_thread_t *threads = TLS_GET;
	gomp_work_t *ws = threads->ts.work_share;
	unsigned num_threads = ws->num_threads;
	if (num_threads) {
		// LOG("loop_ull_start: up = %i, start = %llu, end = %llu, incr = %llu, chunk_size = %llu\n", up, start, end, incr, chunk_size);
		GOMP_TEAM_ULL_START((*ws))
	}
	return gomp_iter_ull_dynamic_next(istart, iend);
}

HIDDEN void GOMP_parallel(void (*fn) (void *), void *data, unsigned num_threads, unsigned flags) {
	(void)flags;
	// LOG("parallel: fn = %p, data = %p, num_threads = %u, flags = %u\n", fn, data, num_threads, flags);
	GOMP_TEAM_INIT
	ws.num_threads = num_threads;
	fn(data);
	GOMP_TEAM_END
}

HIDDEN bool GOMP_loop_dynamic_next(long *istart, long *iend) {
	// LOG("loop_dynamic_next: istart = %p, iend = %p\n", istart, iend);
	return gomp_iter_dynamic_next(istart, iend);
}

HIDDEN bool GOMP_loop_ull_dynamic_next(gomp_ull *istart, gomp_ull *iend) {
	// LOG("loop_ull_dynamic_next: istart = %p, iend = %p\n", istart, iend);
	return gomp_iter_ull_dynamic_next(istart, iend);
}

HIDDEN void GOMP_loop_end_nowait(void) {
	// LOG("loop_end_nowait\n");
}

#define M1(fn, copy) extern __typeof(fn) copy __attribute__((alias(#fn)));
M1(GOMP_parallel_loop_dynamic, GOMP_parallel_loop_guided)
M1(GOMP_loop_dynamic_start, GOMP_loop_guided_start)
M1(GOMP_loop_dynamic_next, GOMP_loop_guided_next)
M1(GOMP_loop_dynamic_start, GOMP_loop_nonmonotonic_dynamic_start)
M1(GOMP_loop_dynamic_next, GOMP_loop_nonmonotonic_dynamic_next)
M1(GOMP_loop_ull_dynamic_start, GOMP_loop_ull_nonmonotonic_dynamic_start)
M1(GOMP_loop_ull_dynamic_next, GOMP_loop_ull_nonmonotonic_dynamic_next)
#undef M1

#ifdef _WIN32
typedef CRITICAL_SECTION gomp_mutex_t;
#define default_lock CriticalSection

static void gomp_mutex_lock(gomp_mutex_t *lock) {
	EnterCriticalSection(lock);
}

static void gomp_mutex_unlock(gomp_mutex_t *lock) {
	LeaveCriticalSection(lock);
}
#else
typedef pthread_mutex_t gomp_mutex_t;
static gomp_mutex_t default_lock = PTHREAD_MUTEX_INITIALIZER;

static void gomp_mutex_lock(gomp_mutex_t *lock) {
	pthread_mutex_lock(lock);
}

static void gomp_mutex_unlock(gomp_mutex_t *lock) {
	pthread_mutex_unlock(lock);
}
#endif

HIDDEN void GOMP_critical_start(void) {
	gomp_mutex_lock(&default_lock);
}

HIDDEN void GOMP_critical_end(void) {
	gomp_mutex_unlock(&default_lock);
}

