// A stand-in for the VST2 SDK's audioeffectx.h: just enough for an
// Airwindows LinuxVST plug-in to compile and run outside a host.
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#define __audioeffect__
typedef int32_t VstInt32;
typedef void* audioMasterCallback;
enum { kVstMaxProgNameLen = 24, kVstMaxParamStrLen = 8, kVstMaxProductStrLen = 64, kVstMaxVendorStrLen = 64, kVstMaxEffectNameLen = 32 };
enum VstPlugCategory { kPlugCategUnknown = 0, kPlugCategEffect };
inline char* vst_strncpy(char* d, const char* s, int n) { strncpy(d, s, n); d[n] = 0; return d; }
inline void float2string(float v, char* t, int n) { snprintf(t, n + 1, "%f", v); }
inline void dB2string(float v, char* t, int n) { snprintf(t, n + 1, "%f", v); }
inline void int2string(int v, char* t, int n) { snprintf(t, n + 1, "%d", v); }
extern double g_sample_rate;
class AudioEffect {
public:
  virtual ~AudioEffect() {}
};
class AudioEffectX : public AudioEffect {
public:
  AudioEffectX(audioMasterCallback, int, int) {}
  void setNumInputs(int) {}
  void setNumOutputs(int) {}
  void setUniqueID(unsigned long) {}
  void canProcessReplacing() {}
  void canDoubleReplacing() {}
  void programsAreChunks(bool) {}
  double getSampleRate() { return g_sample_rate; }
};
