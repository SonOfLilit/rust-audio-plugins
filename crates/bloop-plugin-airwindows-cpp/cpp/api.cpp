// The C ABI bloop drives the Airwindows plug-ins through: one entry per
// plug-in (bloop_aw_entries, written by build.rs), and an instance that
// holds a constructed plug-in in memory of its own.
//
// Only bloop_aw_new and bloop_aw_free allocate or free. Everything else,
// bloop_aw_clear included, runs on the audio thread as well: clearing
// destroys the plug-in and constructs it again in the same memory, which
// none of the plug-ins' constructors allocate for (audioeffectx.h).
#include "audioeffectx.h"

struct BloopEntry {
  const char* name;
  unsigned long size;
  int params;
  AudioEffectX* (*construct)(void* at, BloopHost* host);
};

extern "C" const BloopEntry bloop_aw_entries[];
extern "C" const int bloop_aw_entry_count;

namespace {

// What every instance starts from, and starts from again when cleared:
// the seed its rand() is given.
const uint32_t SEED = 1;

struct Instance {
  BloopHost host;
  const BloopEntry* entry;
  void* memory;
  AudioEffectX* fx;
};

bool valid(const Instance* in, int index) { return index >= 0 && index < in->entry->params; }

}  // namespace

extern "C" {

int bloop_aw_count() { return bloop_aw_entry_count; }

const char* bloop_aw_name(int entry) {
  if (entry < 0 || entry >= bloop_aw_entry_count) return "";
  return bloop_aw_entries[entry].name;
}

int bloop_aw_entry_params(int entry) {
  if (entry < 0 || entry >= bloop_aw_entry_count) return 0;
  return bloop_aw_entries[entry].params;
}

unsigned long bloop_aw_entry_size(int entry) {
  if (entry < 0 || entry >= bloop_aw_entry_count) return 0;
  return bloop_aw_entries[entry].size;
}

// Null for an entry there is not, or memory there is not.
void* bloop_aw_new(int entry, double sample_rate) {
  if (entry < 0 || entry >= bloop_aw_entry_count) return nullptr;
  const BloopEntry* e = &bloop_aw_entries[entry];
  Instance* in = new (std::nothrow) Instance;
  if (!in) return nullptr;
  in->memory = ::operator new(e->size, std::nothrow);
  if (!in->memory) {
    delete in;
    return nullptr;
  }
  in->host.sample_rate = sample_rate;
  in->host.seed = SEED;
  in->entry = e;
  in->fx = e->construct(in->memory, &in->host);
  return in;
}

void bloop_aw_free(void* handle) {
  Instance* in = static_cast<Instance*>(handle);
  if (!in) return;
  in->fx->~AudioEffectX();
  ::operator delete(in->memory);
  delete in;
}

int bloop_aw_param_count(const void* handle) { return static_cast<const Instance*>(handle)->entry->params; }

float bloop_aw_get(void* handle, int index) {
  Instance* in = static_cast<Instance*>(handle);
  return valid(in, index) ? in->fx->getParameter(index) : 0.0f;
}

void bloop_aw_set(void* handle, int index, float value) {
  Instance* in = static_cast<Instance*>(handle);
  if (valid(in, index)) in->fx->setParameter(index, value);
}

// `out` holds at least BLOOP_AW_TEXT bytes (64): every text below is cut
// to kVstMaxParamStrLen, 16, plus its terminator.
void bloop_aw_param_name(void* handle, int index, char* out) {
  Instance* in = static_cast<Instance*>(handle);
  out[0] = 0;
  if (valid(in, index)) in->fx->getParameterName(index, out);
  out[kVstMaxParamStrLen] = 0;
}

void bloop_aw_param_label(void* handle, int index, char* out) {
  Instance* in = static_cast<Instance*>(handle);
  out[0] = 0;
  if (valid(in, index)) in->fx->getParameterLabel(index, out);
  out[kVstMaxParamStrLen] = 0;
}

// The reading at `value`, and the knob put back where it was.
void bloop_aw_param_display_at(void* handle, int index, float value, char* out) {
  Instance* in = static_cast<Instance*>(handle);
  out[0] = 0;
  if (!valid(in, index)) return;
  float was = in->fx->getParameter(index);
  in->fx->setParameter(index, value);
  in->fx->getParameterDisplay(index, out);
  in->fx->setParameter(index, was);
  out[kVstMaxParamStrLen] = 0;
}

// The rate getSampleRate() answers from now on.
void bloop_aw_set_rate(void* handle, double sample_rate) {
  Instance* in = static_cast<Instance*>(handle);
  in->host.sample_rate = sample_rate;
  in->fx->bloop_rate = sample_rate;
}

// Everything the plug-in holds silenced, as it was constructed, with its
// knobs kept. Allocates nothing.
void bloop_aw_clear(void* handle) {
  Instance* in = static_cast<Instance*>(handle);
  const int count = in->entry->params;
  float knobs[64];
  const int kept = count < 64 ? count : 64;
  for (int i = 0; i < kept; i++) knobs[i] = in->fx->getParameter(i);
  in->fx->~AudioEffectX();
  in->host.seed = SEED;
  in->fx = in->entry->construct(in->memory, &in->host);
  for (int i = 0; i < kept; i++) in->fx->setParameter(i, knobs[i]);
}

// A stereo block in place.
void bloop_aw_process(void* handle, float* left, float* right, int frames) {
  Instance* in = static_cast<Instance*>(handle);
  if (frames <= 0) return;
  float* io[2] = {left, right};
  in->fx->processReplacing(io, io, frames);
}

}  // extern "C"
