// MTS-ESP's client, as Wavetable sees it with no tuning master running:
// registering yields no client, and Wavetable skips every call it would
// make with one.
#pragma once

struct MTSClient;

inline MTSClient* MTS_RegisterClient() { return nullptr; }
inline void MTS_DeregisterClient (MTSClient*) {}
inline double MTS_RetuningInSemitones (MTSClient*, char, char) { return 0.0; }
inline void MTS_ParseMIDIDataU (MTSClient*, const unsigned char*, int) {}
