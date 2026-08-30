/*
    Stands in for Surge Synth Team's libMTSClient.

    Just-a-Sample calls into MTS-ESP for microtuning; with no MTS-ESP master
    running, the real library filters no notes and retunes nothing. That is the
    behaviour Bloop's port reproduces, so the harness hard-codes it.
*/

#pragma once

struct MTSClient;

inline bool MTS_ShouldFilterNote (MTSClient*, char, char) { return false; }
inline double MTS_RetuningAsRatio (MTSClient*, char, char) { return 1.0; }
