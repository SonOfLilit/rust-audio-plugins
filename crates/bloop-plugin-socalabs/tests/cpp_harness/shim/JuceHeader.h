// What slPlugins' PluginProcessor.cpp files get from <JuceHeader.h>, so they
// compile here unmodified: the real JUCE non-GUI modules and Gin DSP
// (gin_dsp.h), and stand-ins for the two classes that would need JUCE's GUI
// modules — juce::AudioProcessor and gin::Processor with its parameters.
//
// The stand-ins do to a value exactly what the originals do. gin::Parameter
// and gin::SmoothedParameter are transcribed from gin_parameter.{h,cpp} and
// gin_smoothedparameter.h over the real juce::NormalisableRange and the
// real gin smoothers; gin::Processor from gin_processor.cpp's createParam,
// prepareToPlay, reset and isSmoothing. Everything a host needs of them
// that never reaches audio — notification, text, state, presets — is left
// out, and whatever a processor calls of it is a no-op.
#pragma once

#include "../gin_dsp.h"

#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
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
    virtual ~AudioProcessor() = default;

    virtual void prepareToPlay (double sampleRate, int samplesPerBlock) = 0;
    virtual void releaseResources() {}
    virtual void processBlock (AudioBuffer<float>& buffer, MidiBuffer& midi) = 0;
    virtual void reset() {}
    virtual void numChannelsChanged() {}
    virtual AudioProcessorEditor* createEditor() { return nullptr; }
    virtual bool hasEditor() const { return false; }

    int getTotalNumInputChannels() const    { return 2; }
    int getTotalNumOutputChannels() const   { return 2; }

    AudioPlayHead* getPlayHead() const      { return playHead; }

    void setLatencySamples (int samples)    { latency = samples; }
    int getLatencySamples() const           { return latency; }

    AudioPlayHead* playHead = nullptr;

private:
    int latency = 0;
};

}

namespace gin
{

class SmoothingType
{
public:
    enum Type { linear, eased };
    SmoothingType (float time_ = 0.1f, Type type_ = linear) : time (time_), type (type_) {}
    float time;
    Type type;
};

class Parameter
{
public:
    using Ptr = Parameter*;
    using ConversionFunction = std::function<float (float)>;

    Parameter (juce::NormalisableRange<float> range_, float defaultValue_)
        : range (range_), value (defaultValue_) {}
    virtual ~Parameter() = default;

    virtual void prepareToPlay (double, int) {}
    virtual void reset() {}

    virtual void setUserValue (float v)
    {
        v = juce::jlimit (range.start, range.end, range.snapToLegalValue (v));
        if (! juce::approximatelyEqual (value, v))
            value = v;
    }

    float getUserValue() const              { return juce::jlimit (range.start, range.end, value); }
    int getUserValueInt() const             { return int (juce::jlimit (range.start, range.end, value)); }
    bool isOn()                             { return ! juce::exactlyEqual (range.start, getUserValue()); }

    float getProcValue()
    {
        if (conversionFunction != nullptr)
            return conversionFunction (getUserValue());
        return getUserValue();
    }

    virtual float getProcValue (int /*stepSize*/)
    {
        if (conversionFunction != nullptr)
            return conversionFunction (getUserValue());
        return getUserValue();
    }

    virtual bool isSmoothingActive()        { return false; }

    ConversionFunction conversionFunction;

protected:
    juce::NormalisableRange<float> range;
    float value;
};

template <class T>
class SmoothedParameter : public Parameter
{
public:
    SmoothedParameter (juce::NormalisableRange<float> range_, float defaultValue_)
        : Parameter (range_, defaultValue_)
    {
        smoother.setValue (range.convertTo0to1 (value));
        smoother.snapToValue();
    }

    void prepareToPlay (double sampleRate, int) override    { smoother.setSampleRate (sampleRate); }
    void reset() override                                   { smoother.snapToValue(); }

    void setUserValue (float v) override
    {
        v = juce::jlimit (range.start, range.end, range.snapToLegalValue (v));
        if (! juce::approximatelyEqual (value, v))
        {
            value = v;
            smoother.setValue (range.convertTo0to1 (value));
        }
    }

    float getProcValue (int stepSize) override
    {
        if (smoother.isSmoothing())
        {
            auto v = range.convertFrom0to1 (smoother.getCurrentValue());
            smoother.process (stepSize);
            if (conversionFunction != nullptr)
                return conversionFunction (v);
            return v;
        }
        return Parameter::getProcValue (stepSize);
    }

    void setSmoothingTime (float t)         { smoother.setTime (t); }
    bool isSmoothingActive() override       { return smoother.isSmoothing(); }

protected:
    T smoother;
};

class ProcessorOptions
{
public:
    ProcessorOptions withMidiLearn() const                                  { return *this; }
    ProcessorOptions withAdditionalCredits (const juce::StringArray&) const  { return *this; }
};

class MidiLearn
{
public:
    void processBlock (juce::MidiBuffer&, int) {}
};

class Processor : public juce::AudioProcessor
{
public:
    Processor (bool = false, ProcessorOptions = {}) {}

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        for (auto& p : allParameters)
            p->prepareToPlay (sampleRate, samplesPerBlock);
    }

    void reset() override
    {
        for (auto& p : allParameters)
            p->reset();
    }

    bool isSmoothing()
    {
        for (auto& p : allParameters)
            if (p->isSmoothingActive())
                return true;
        return false;
    }

    virtual juce::File getProgramDirectory()                        { return {}; }
    virtual juce::Array<juce::File> getFactoryProgramDirectories()  { return {}; }

    void init() {}

    // Always null: a MIDI learn map only ever moves parameters, and the
    // gold programs move them directly.
    std::unique_ptr<MidiLearn> midiLearn;

    // External parameters in the order they are added: what the gold
    // programs' `set <index>` addresses.
    std::vector<Parameter*> extParameters;

protected:
    Parameter::Ptr addExtParam (juce::String, juce::String, juce::String, juce::String,
                                juce::NormalisableRange<float> range, float defaultValue,
                                SmoothingType st)
    {
        auto p = createParam (range, defaultValue, st);
        extParameters.push_back (p);
        return p;
    }

    template <class TextFunction>
    Parameter::Ptr addExtParam (juce::String uid, juce::String name, juce::String shortName, juce::String label,
                                juce::NormalisableRange<float> range, float defaultValue,
                                SmoothingType st, TextFunction&&)
    {
        return addExtParam (uid, name, shortName, label, range, defaultValue, st);
    }

    Parameter::Ptr addIntParam (juce::String, juce::String, juce::String, juce::String,
                                juce::NormalisableRange<float> range, float defaultValue,
                                SmoothingType st)
    {
        return createParam (range, defaultValue, st);
    }

    template <class TextFunction>
    Parameter::Ptr addIntParam (juce::String uid, juce::String name, juce::String shortName, juce::String label,
                                juce::NormalisableRange<float> range, float defaultValue,
                                SmoothingType st, TextFunction&&)
    {
        return addIntParam (uid, name, shortName, label, range, defaultValue, st);
    }

private:
    Parameter::Ptr createParam (juce::NormalisableRange<float> range, float defaultValue, SmoothingType st)
    {
        std::unique_ptr<Parameter> p;
        if (st.time > 0.0f)
        {
            if (st.type == SmoothingType::linear)
            {
                auto sp = new SmoothedParameter<ValueSmoother<float>> (range, defaultValue);
                sp->setSmoothingTime (st.time);
                p = std::unique_ptr<Parameter> (sp);
            }
            else if (st.type == SmoothingType::eased)
            {
                auto sp = new SmoothedParameter<EasedValueSmoother<float>> (range, defaultValue);
                sp->setSmoothingTime (st.time);
                p = std::unique_ptr<Parameter> (sp);
            }
        }
        else
        {
            p = std::make_unique<Parameter> (range, defaultValue);
        }
        auto raw = p.get();
        allParameters.push_back (std::move (p));
        return raw;
    }

    std::vector<std::unique_ptr<Parameter>> allParameters;
};

}
