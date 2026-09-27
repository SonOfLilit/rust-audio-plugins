// slPlugins' Compressor, Limiter, Gate and Expander processors: the
// constructor's parameters and prepareToPlay/reset/processBlock as
// PluginProcessor.cpp has them, less the MIDI learn, the preset files and
// the scope FIFO, none of which reach the audio.
#pragma once

#include "params.h"

namespace harness
{

class CompressorAudioProcessor : public Processor
{
public:
    CompressorAudioProcessor()
    {
        attack    = addExtParam ("attack",    "Attack",    "", "ms",   { 1.0f,   200.0f, 0.0f, 0.4f},    1.0f, 0.1f);
        release   = addExtParam ("release",   "Release",   "", "ms",   { 1.0f,  2000.0f, 0.0f, 0.4f},    5.0f, 0.1f);
        ratio     = addExtParam ("ratio",     "Ratio",     "", "",     { 1.0f,    30.0f, 0.0f, 0.4f},    5.0f, 0.1f);
        threshold = addExtParam ("threshold", "Threshold", "", "",     { -60.0f,   0.0f, 0.0f, 1.0f},  -30.0f, 0.1f);
        knee      = addExtParam ("knee",      "Knee",      "", "",     { 0.0f,    60.0f, 0.0f, 1.0f},    5.0f, 0.1f);
        input     = addExtParam ("input",     "Input",     "", "",     { -30.0f,  30.0f, 0.0f, 1.0f},    0.0f, 0.1f);
        output    = addExtParam ("output",    "Output",    "", "",     { -30.0f,  30.0f, 0.0f, 1.0f},    0.0f, 0.1f);

        attack->conversionFunction  = [] (float in) { return in / 1000.0; };
        release->conversionFunction = [] (float in) { return in / 1000.0; };
        input->conversionFunction   = [] (float in) { return juce::Decibels::decibelsToGain (in); };
        output->conversionFunction  = [] (float in) { return juce::Decibels::decibelsToGain (in); };
    }

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        Processor::prepareToPlay (sampleRate, samplesPerBlock);

        compressor.setSampleRate (sampleRate);
        compressor.reset();
        compressor.setNumChannels (getTotalNumInputChannels());
    }

    void reset() override
    {
        Processor::reset();
        compressor.reset();
    }

    void processBlock (juce::AudioSampleBuffer& buffer) override
    {
        int numSamples = buffer.getNumSamples();

        gin::ScratchBuffer envData (1, numSamples);

        if (isSmoothing())
        {
            int pos = 0;
            while (pos < numSamples)
            {
                auto workBuffer = gin::sliceBuffer (buffer, pos, 1);
                auto envWorkBuffer = gin::sliceBuffer (envData, pos, 1);

                compressor.setInputGain (input->getProcValue (1));
                compressor.setOutputGain (output->getProcValue (1));
                compressor.setParams (attack->getProcValue (1),
                                      0.0f,
                                      release->getProcValue (1),
                                      threshold->getProcValue (1),
                                      ratio->getProcValue (1),
                                      knee->getProcValue (1));

                compressor.process (workBuffer, &envWorkBuffer);

                pos++;
            }
        }
        else
        {
            compressor.setInputGain (input->getProcValue (numSamples));
            compressor.setOutputGain (output->getProcValue (numSamples));
            compressor.setParams (attack->getProcValue (numSamples),
                                  0.0f,
                                  release->getProcValue (numSamples),
                                  threshold->getProcValue (numSamples),
                                  ratio->getProcValue (numSamples),
                                  knee->getProcValue (numSamples));

            compressor.process (buffer, &envData);
        }
    }

    Parameter::Ptr attack, release, ratio, threshold, knee, input, output;
    gin::Dynamics compressor;
};

class LimiterAudioProcessor : public Processor
{
public:
    LimiterAudioProcessor()
    {
        attack    = addExtParam ("attack",    "Attack",    "", "ms",    { 0.0f,     5.0f, 0.0f, 1.0f},    0.0f, 0.1f);
        release   = addExtParam ("release",   "Release",   "", "ms",    { 1.0f,   500.0f, 0.0f, 0.3f},    5.0f, 0.1f);
        threshold = addExtParam ("threshold", "Threshold", "", "dB",    { -60.0f,   0.0f, 0.0f, 1.0f},  -30.0f, 0.1f);
        input     = addExtParam ("input",     "Input",     "", "dB",    { -30.0f,  30.0f, 0.0f, 1.0f},    0.0f, 0.1f);
        output    = addExtParam ("output",    "Output",    "", "dB",    { -30.0f,  30.0f, 0.0f, 1.0f},    0.0f, 0.1f);

        attack->conversionFunction  = [] (float in) { return in / 1000.0; };
        release->conversionFunction = [] (float in) { return in / 1000.0; };
        input->conversionFunction   = [] (float in) { return juce::Decibels::decibelsToGain (in); };
        output->conversionFunction  = [] (float in) { return juce::Decibels::decibelsToGain (in); };
    }

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        Processor::prepareToPlay (sampleRate, samplesPerBlock);

        limiter.setSampleRate (sampleRate);
        limiter.reset();
        limiter.setMode (gin::Dynamics::limiter);
        limiter.setNumChannels (getTotalNumInputChannels());
    }

    void reset() override
    {
        Processor::reset();
        limiter.reset();
    }

    void processBlock (juce::AudioSampleBuffer& buffer) override
    {
        int numSamples = buffer.getNumSamples();

        gin::ScratchBuffer envData (1, numSamples);

        if (isSmoothing())
        {
            int pos = 0;
            while (pos < numSamples)
            {
                auto workBuffer = gin::sliceBuffer (buffer, pos, 1);
                auto envWorkBuffer = gin::sliceBuffer (envData, pos, 1);

                limiter.setInputGain (input->getProcValue (1));
                limiter.setOutputGain (output->getProcValue (1));
                limiter.setParams (attack->getProcValue (1),
                                   0.0f,
                                   release->getProcValue (1),
                                   threshold->getProcValue (1),
                                   1000,
                                   0);

                limiter.process (workBuffer, &envWorkBuffer);

                pos++;
            }
        }
        else
        {
            limiter.setInputGain (input->getProcValue (numSamples));
            limiter.setOutputGain (output->getProcValue (numSamples));
            limiter.setParams (attack->getProcValue (numSamples),
                               0.0f,
                               release->getProcValue (numSamples),
                               threshold->getProcValue (numSamples),
                               1000,
                               0);

            limiter.process (buffer, &envData);
        }
    }

    Parameter::Ptr attack, release, threshold, input, output;
    gin::Dynamics limiter;
};

class GateAudioProcessor : public Processor
{
public:
    GateAudioProcessor()
    {
        attack    = addExtParam ("attack",    "Attack",    "", "ms",    { 0.0f,     5.0f, 0.0f, 1.0f},    0.0f, 0.1f);
        hold      = addExtParam ("hold",      "Hold",      "", "ms",    { 0.0f,   500.0f, 0.0f, 0.3f},    0.0f, 0.1f);
        release   = addExtParam ("release",   "Release",   "", "ms",    { 1.0f,   500.0f, 0.0f, 0.3f},    5.0f, 0.1f);
        threshold = addExtParam ("threshold", "Threshold", "", "dB",    { -100.0f,  0.0f, 0.0f, 1.0f},  -30.0f, 0.1f);
        knee      = addExtParam ("knee",      "Knee",      "", "dB",    { 0.0f,    60.0f, 0.0f, 1.0f},    0.0f, 0.1f);
        input     = addExtParam ("input",     "Input",     "", "dB",    { -30.0f,  30.0f, 0.0f, 1.0f},    0.0f, 0.1f);
        output    = addExtParam ("output",    "Output",    "", "dB",    { -30.0f,  30.0f, 0.0f, 1.0f},    0.0f, 0.1f);

        attack->conversionFunction  = [] (float in) { return in / 1000.0; };
        hold->conversionFunction    = [] (float in) { return in / 1000.0; };
        release->conversionFunction = [] (float in) { return in / 1000.0; };
        input->conversionFunction   = [] (float in) { return juce::Decibels::decibelsToGain (in); };
        output->conversionFunction  = [] (float in) { return juce::Decibels::decibelsToGain (in); };
    }

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        Processor::prepareToPlay (sampleRate, samplesPerBlock);

        gate.setSampleRate (sampleRate);
        gate.reset();
        gate.setMode (gin::Dynamics::gate);
        gate.setNumChannels (getTotalNumInputChannels());
    }

    void reset() override
    {
        Processor::reset();
        gate.reset();
    }

    void processBlock (juce::AudioSampleBuffer& buffer) override
    {
        int numSamples = buffer.getNumSamples();

        gin::ScratchBuffer envData (1, numSamples);

        if (isSmoothing())
        {
            int pos = 0;
            while (pos < numSamples)
            {
                auto workBuffer = gin::sliceBuffer (buffer, pos, 1);
                auto envWorkBuffer = gin::sliceBuffer (envData, pos, 1);

                gate.setInputGain (input->getProcValue (1));
                gate.setOutputGain (output->getProcValue (1));
                gate.setParams (attack->getProcValue (1),
                                hold->getProcValue (1),
                                release->getProcValue (1),
                                threshold->getProcValue (1),
                                1000,
                                knee->getProcValue (1));

                gate.process (workBuffer, &envWorkBuffer);

                pos++;
            }
        }
        else
        {
            gate.setInputGain (input->getProcValue (numSamples));
            gate.setOutputGain (output->getProcValue (numSamples));
            gate.setParams (attack->getProcValue (numSamples),
                            hold->getProcValue (numSamples),
                            release->getProcValue (numSamples),
                            threshold->getProcValue (numSamples),
                            1000,
                            knee->getProcValue (1));

            gate.process (buffer, &envData);
        }
    }

    Parameter::Ptr attack, hold, release, threshold, knee, input, output;
    gin::Dynamics gate;
};

class ExpanderAudioProcessor : public Processor
{
public:
    ExpanderAudioProcessor()
    {
        attack    = addExtParam ("attack",    "Attack",    "", "ms",   { 1.0f,   200.0f, 0.0f, 0.4f},    1.0f, 0.1f);
        release   = addExtParam ("release",   "Release",   "", "ms",   { 1.0f,  2000.0f, 0.0f, 0.4f},    5.0f, 0.1f);
        ratio     = addExtParam ("ratio",     "Ratio",     "", "",     { 1.0f,     4.0f, 0.0f, 1.0f},    1.0f, 0.1f);
        threshold = addExtParam ("threshold", "Threshold", "", "",     { -60.0f,   0.0f, 0.0f, 1.0f},  -30.0f, 0.1f);
        knee      = addExtParam ("knee",      "Knee",      "", "",     { 0.0f,    60.0f, 0.0f, 1.0f},    5.0f, 0.1f);
        input     = addExtParam ("input",     "Input",     "", "",     { -30.0f,  30.0f, 0.0f, 1.0f},    0.0f, 0.1f);
        output    = addExtParam ("output",    "Output",    "", "",     { -30.0f,  30.0f, 0.0f, 1.0f},    0.0f, 0.1f);

        attack->conversionFunction  = [] (float in) { return in / 1000.0; };
        release->conversionFunction = [] (float in) { return in / 1000.0; };
        input->conversionFunction   = [] (float in) { return juce::Decibels::decibelsToGain (in); };
        output->conversionFunction  = [] (float in) { return juce::Decibels::decibelsToGain (in); };
    }

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        Processor::prepareToPlay (sampleRate, samplesPerBlock);

        expander.setMode (gin::Dynamics::expander);
        expander.setSampleRate (sampleRate);
        expander.reset();
        expander.setNumChannels (getTotalNumInputChannels());
    }

    void reset() override
    {
        Processor::reset();
        expander.reset();
    }

    void processBlock (juce::AudioSampleBuffer& buffer) override
    {
        int numSamples = buffer.getNumSamples();

        gin::ScratchBuffer envData (1, numSamples);

        if (isSmoothing())
        {
            int pos = 0;
            while (pos < numSamples)
            {
                auto workBuffer = gin::sliceBuffer (buffer, pos, 1);
                auto envWorkBuffer = gin::sliceBuffer (envData, pos, 1);

                expander.setInputGain (input->getProcValue (1));
                expander.setOutputGain (output->getProcValue (1));
                expander.setParams (attack->getProcValue (1),
                                    0.0f,
                                    release->getProcValue (1),
                                    threshold->getProcValue (1),
                                    ratio->getProcValue (1),
                                    knee->getProcValue (1));

                expander.process (workBuffer, &envWorkBuffer);

                pos++;
            }
        }
        else
        {
            expander.setInputGain (input->getProcValue (numSamples));
            expander.setOutputGain (output->getProcValue (numSamples));
            expander.setParams (attack->getProcValue (numSamples),
                                0.0f,
                                release->getProcValue (numSamples),
                                threshold->getProcValue (numSamples),
                                ratio->getProcValue (numSamples),
                                knee->getProcValue (numSamples));

            expander.process (buffer, &envData);
        }
    }

    Parameter::Ptr attack, release, ratio, threshold, knee, input, output;
    gin::Dynamics expander;
};

}
