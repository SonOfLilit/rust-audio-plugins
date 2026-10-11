/* Linked into this crate's test binary only, on Linux, with the linker's
 * --wrap for each allocator entry point (build.rs): every call the
 * binary's own code makes — Rust's allocator and the plug-ins' C++ alike —
 * comes through here and is counted on the calling thread. */
#include <stddef.h>

void* __real_malloc(size_t size);
void* __real_calloc(size_t count, size_t size);
void* __real_realloc(void* ptr, size_t size);
void* __real__Znwm(size_t size);
void* __real__Znam(size_t size);
void* __real__ZnwmRKSt9nothrow_t(size_t size, const void* tag);
void* __real__ZnamRKSt9nothrow_t(size_t size, const void* tag);

static _Thread_local unsigned long long allocations;

void* __wrap_malloc(size_t size) {
  allocations++;
  return __real_malloc(size);
}

void* __wrap_calloc(size_t count, size_t size) {
  allocations++;
  return __real_calloc(count, size);
}

void* __wrap_realloc(void* ptr, size_t size) {
  allocations++;
  return __real_realloc(ptr, size);
}

void* __wrap__Znwm(size_t size) {
  allocations++;
  return __real__Znwm(size);
}

void* __wrap__Znam(size_t size) {
  allocations++;
  return __real__Znam(size);
}

void* __wrap__ZnwmRKSt9nothrow_t(size_t size, const void* tag) {
  allocations++;
  return __real__ZnwmRKSt9nothrow_t(size, tag);
}

void* __wrap__ZnamRKSt9nothrow_t(size_t size, const void* tag) {
  allocations++;
  return __real__ZnamRKSt9nothrow_t(size, tag);
}

unsigned long long bloop_aw_test_allocations(void) { return allocations; }
