// What Wavetable's sources get from <JuceHeader.h>, so they compile here
// unmodified: the real JUCE non-GUI modules, the real Gin DSP, the real
// gin::ModMatrix and gin::Synthesiser (over JUCE's real MPESynthesiser),
// and stand-ins for the classes that would need JUCE's GUI and plug-in
// modules — juce::AudioProcessor and its editor, and gin::Processor with its
// parameters.
//
// The stand-ins do to a value exactly what the originals do: gin::Parameter
// is transcribed from gin_parameter.{h,cpp} (its value, range, conversion
// and modulation index), gin::Processor from gin_processor.cpp's parameter
// registration (addExtParam and addIntParam, internal or not, in order).
// What only a host or an editor needs — notification, text, programs, the
// GUI — is left out, or is a no-op.
#pragma once

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include <juce_data_structures/juce_data_structures.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>

#include <functional>
#include <memory>
#include <variant>
#include <vector>

namespace juce
{

class AudioProcessorEditor
{
public:
    virtual ~AudioProcessorEditor() = default;
};

class AudioProcessor
{
public:
    struct BusesProperties
    {
        BusesProperties withInput (const String&, const AudioChannelSet&, bool = true) const   { return *this; }
        BusesProperties withOutput (const String&, const AudioChannelSet&, bool = true) const  { return *this; }
    };

    struct BusesLayout
    {
        Array<AudioChannelSet> inputBuses, outputBuses;
    };

    virtual ~AudioProcessor() = default;

    virtual void prepareToPlay (double sampleRate, int samplesPerBlock) = 0;
    virtual void releaseResources() {}
    virtual void processBlock (AudioBuffer<float>& buffer, MidiBuffer& midi) = 0;
    virtual void reset() {}
    virtual bool isBusesLayoutSupported (const BusesLayout&) const { return true; }
    virtual bool supportsMPE() const { return false; }
    virtual AudioProcessorEditor* createEditor() { return nullptr; }
    virtual bool hasEditor() const { return false; }

    void setRateAndBufferSizeDetails (double rate, int block)   { sampleRate = rate; blockSize = block; }
    double getSampleRate() const            { return sampleRate; }
    int getBlockSize() const                { return blockSize; }

    AudioPlayHead* getPlayHead() const      { return playHead; }

    AudioPlayHead* playHead = nullptr;

private:
    double sampleRate = 0.0;
    int blockSize = 0;
};

}

#include "../gin.h"

namespace gin
{

class ScaledPluginEditor : public juce::AudioProcessorEditor
{
public:
    ScaledPluginEditor (juce::AudioProcessorEditor* editor, juce::ValueTree = {})   { delete editor; }
};

}
