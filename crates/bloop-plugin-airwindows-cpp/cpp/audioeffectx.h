// A stand-in for the VST2 SDK's audioeffectx.h, of bloop's own: what an
// Airwindows LinuxVST plug-in calls on its base class, and no more. Each
// plug-in is compiled inside a namespace of its own (see build.rs), after
// this header, so the standard headers it includes are already in and its
// `#include <set>` and `<string>` do nothing there.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <new>
#include <set>
#include <string>

#define __audioeffect__

typedef int32_t VstInt32;

// What a plug-in is constructed with: its host. Here the host is two
// numbers, so a plug-in can be constructed again in place (to clear it)
// at the rate and with the seed it had.
struct BloopHost {
  double sample_rate;
  uint32_t seed;
};
typedef BloopHost* audioMasterCallback;

// kVstMaxParamStrLen is 8 in the SDK; a host truncates there. Names and
// readings here get room for twice that, and what a plug-in writes past
// eight characters reaches the knob.
enum {
  kVstMaxProgNameLen = 24,
  kVstMaxParamStrLen = 16,
  kVstMaxProductStrLen = 64,
  kVstMaxVendorStrLen = 64,
  kVstMaxEffectNameLen = 32,
};
enum VstPlugCategory { kPlugCategUnknown = 0, kPlugCategEffect };

inline char* vst_strncpy(char* d, const char* s, int n) {
  strncpy(d, s, (size_t)n);
  d[n] = 0;
  return d;
}

// A knob's value as bloop reads a bare number: two places below ten, one
// below a hundred, none above.
inline void float2string(float v, char* t, int n) {
  double a = fabs((double)v);
  const char* f = a < 9.995 ? "%.2f" : a < 99.95 ? "%.1f" : "%.0f";
  snprintf(t, (size_t)n + 1, f, (double)v);
}
inline void dB2string(float v, char* t, int n) {
  if (v <= 0.0f) {
    vst_strncpy(t, "-inf", n);
  } else {
    float2string((float)(20.0 * log10((double)v)), t, n);
  }
}
inline void int2string(int v, char* t, int n) { snprintf(t, (size_t)n + 1, "%d", v); }

class AudioEffect {
 public:
  virtual ~AudioEffect() {}
};

class AudioEffectX : public AudioEffect {
 public:
  AudioEffectX(audioMasterCallback host, int, int)
      : bloop_rate(host ? host->sample_rate : 44100.0), bloop_seed(host ? host->seed : 1u) {}
  void setNumInputs(int) {}
  void setNumOutputs(int) {}
  void setUniqueID(unsigned long) {}
  void canProcessReplacing() {}
  void canDoubleReplacing() {}
  void programsAreChunks(bool) {}
  double getSampleRate() { return bloop_rate; }

  virtual bool getEffectName(char*) { return false; }
  virtual VstPlugCategory getPlugCategory() { return kPlugCategEffect; }
  virtual bool getProductString(char*) { return false; }
  virtual bool getVendorString(char*) { return false; }
  virtual VstInt32 getVendorVersion() { return 0; }
  virtual void processReplacing(float**, float**, VstInt32) {}
  virtual void processDoubleReplacing(double**, double**, VstInt32) {}
  virtual void getProgramName(char*) {}
  virtual void setProgramName(char*) {}
  virtual VstInt32 getChunk(void**, bool) { return 0; }
  virtual VstInt32 setChunk(void*, VstInt32, bool) { return 0; }
  virtual float getParameter(VstInt32) { return 0.0f; }
  virtual void setParameter(VstInt32, float) {}
  virtual void getParameterLabel(VstInt32, char* t) { t[0] = 0; }
  virtual void getParameterName(VstInt32, char* t) { t[0] = 0; }
  virtual void getParameterDisplay(VstInt32, char* t) { t[0] = 0; }
  virtual VstInt32 canDo(char*) { return 0; }

  double bloop_rate;
  // The plug-in's rand(): glibc's takes a lock, and its sequence is the
  // process's, so a plug-in's dither would depend on every other one
  // built before it. This one is the instance's own (an LCG, glibc's
  // TYPE_0 constants), seeded by the host.
  uint32_t bloop_seed;
  int bloop_rand() {
    bloop_seed = bloop_seed * 1103515245u + 12345u;
    return (int)((bloop_seed >> 1) & 0x7fffffff);
  }
};

// What a plug-in's `std::set<std::string> _canDo` becomes: the host never
// asks canDo, and the real set allocates in the constructor, which would
// stop a plug-in being constructed again in place on the audio thread.
namespace bloop_no_std {
typedef const char* string;
template <class T>
struct set {
  void insert(const char*) {}
  int find(const char*) const { return 0; }
  int end() const { return 0; }
};
}  // namespace bloop_no_std

#define rand() bloop_rand()
