/*
    Stands in for Just-a-Sample's FX base class and its four effects.

    Bloop routes effects through its own mixer, so the port has no FX chain.
    The reference voice still constructs, initialises and queries the chain on
    every note, so the harness supplies inert effects: with every `*Enabled`
    parameter false — which the gold programs guarantee — the reference never
    calls `process`, and the rendered output is the dry voice.
*/

#pragma once
#include <JuceHeader.h>

#include "../SamplerParameters.h"

class Effect
{
public:
    virtual ~Effect() = default;
    virtual void initialize (int numChannels, int fxSampleRate) = 0;
    virtual void updateParams (const SamplerParameters& sampleSound, bool modulating = false) = 0;
    virtual void process (juce::AudioBuffer<float>& buffer, int numSamples, int startSample = 0) = 0;
};

class InertEffect : public Effect
{
public:
    void initialize (int, int) override {}
    void updateParams (const SamplerParameters&, bool = false) override {}
    void process (juce::AudioBuffer<float>&, int, int = 0) override
    {
        // Unreachable while every effect is disabled; see the file comment.
        std::abort();
    }
};
