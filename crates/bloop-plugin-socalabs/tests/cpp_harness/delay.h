// slPlugins' Delay processor: the constructor's parameters and
// prepareToPlay/reset/processBlock as PluginProcessor.cpp has them, less
// the MIDI learn and the preset files.
#pragma once

#include "params.h"

namespace harness
{

class DelayAudioProcessor : public Processor
{
public:
    DelayAudioProcessor()
    {
        float mxd = float (gin::NoteDuration::getNoteDurations().size()) - 1.0f;

        sync  = addExtParam ("sync",  "Sync",      "", "",   {   0.0f,   1.0f, 1.0f, 1.0f},    0.0f, 0.0f);
        time  = addExtParam ("time",  "Delay",     "", "",   {   0.0f, 120.0f, 0.0f, 0.3f},    1.0f, 0.0f);
        beat  = addExtParam ("beat",  "Delay",     "", "",   {   0.0f,    mxd, 1.0f, 1.0f},   13.0f, 0.0f);
        fb    = addExtParam ("fb",    "Feedback",  "", "dB", {-100.0f,   0.0f, 0.0f, 5.0f},  -10.0f, 0.1f);
        cf    = addExtParam ("cf",    "Crossfeed", "", "dB", {-100.0f,   0.0f, 0.0f, 5.0f}, -100.0f, 0.1f);
        mix   = addExtParam ("mix",   "Mix",       "", "%",  {   0.0f, 100.0f, 0.0f, 1.0f},    0.0f, 0.1f);

        delay = addIntParam ("delay", "Delay",     "", "",   {   0.0f, 120.0f, 0.0f, 1.0f},    1.0f, {0.2f, SmoothingType::eased});

        fb->conversionFunction  = [] (float in) { return juce::Decibels::decibelsToGain (in); };
        cf->conversionFunction  = [] (float in) { return juce::Decibels::decibelsToGain (in); };
        mix->conversionFunction = [] (float in) { return in / 100.0f; };
    }

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        Processor::prepareToPlay (sampleRate, samplesPerBlock);

        stereoDelay.setSampleRate (sampleRate);
    }

    void reset() override
    {
        Processor::reset();

        stereoDelay.reset();
    }

    void updateInternalParams()
    {
        if (sync->isOn())
        {
            auto& duration = gin::NoteDuration::getNoteDurations()[(size_t)beat->getUserValueInt()];
            delay->setUserValue (duration.toSeconds (getPlayHead()));
        }
        else
        {
            delay->setUserValue (time->getUserValue());
        }
    }

    void processBlock (juce::AudioSampleBuffer& buffer) override
    {
        updateInternalParams();

        int numSamples = buffer.getNumSamples();

        if (isSmoothing())
        {
            int pos = 0;
            while (pos < numSamples)
            {
                auto workBuffer = gin::sliceBuffer (buffer, pos, 1);

                stereoDelay.setParams (delay->getProcValue (1), mix->getProcValue (1),
                                       fb->getProcValue (1), cf->getProcValue (1));

                stereoDelay.process (workBuffer);

                pos++;
            }
        }
        else
        {
            stereoDelay.setParams (delay->getProcValue (numSamples), mix->getProcValue (numSamples),
                                   fb->getProcValue (numSamples), cf->getProcValue (numSamples));

            stereoDelay.process (buffer);
        }
    }

    Parameter::Ptr sync, time, beat, fb, cf, mix, delay;
    gin::StereoDelay stereoDelay;
};

}
