// A stand-in for gin::Parameter and gin::SmoothedParameter, which cannot be
// compiled here: they derive from juce::AudioProcessorParameter, and
// juce_audio_processors needs JUCE's GUI modules. What they do to a value
// is transcribed from gin_parameter.h, gin_smoothedparameter.h and
// Processor::createParam line for line, over the real
// juce::NormalisableRange and the real gin smoothers, so the harness
// processors below can keep the originals' processBlock bodies as written.
#pragma once

#include "gin_dsp.h"

#include <functional>
#include <memory>
#include <vector>

namespace harness
{

struct SmoothingType
{
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

// gin::Processor, as far as its parameters go.
class Processor
{
public:
    virtual ~Processor() = default;

    virtual void prepareToPlay (double sampleRate, int samplesPerBlock)
    {
        for (auto& p : allParameters)
            p->prepareToPlay (sampleRate, samplesPerBlock);
    }

    virtual void reset()
    {
        for (auto& p : allParameters)
            p->reset();
    }

    virtual void processBlock (juce::AudioSampleBuffer& buffer) = 0;

    bool isSmoothing()
    {
        for (auto& p : allParameters)
            if (p->isSmoothingActive())
                return true;
        return false;
    }

    int getTotalNumInputChannels() const    { return 2; }

    // The host's play head: 4/4 at whatever tempo the program last set.
    struct PlayHead : juce::AudioPlayHead
    {
        double bpm = 120.0;

        juce::Optional<PositionInfo> getPosition() const override
        {
            PositionInfo info;
            info.setBpm (bpm);
            info.setTimeSignature (juce::AudioPlayHead::TimeSignature { 4, 4 });
            return info;
        }
    };

    PlayHead playHead;
    juce::AudioPlayHead* getPlayHead()      { return &playHead; }

    // The order parameters are added is the order the gold tests index them.
    std::vector<std::unique_ptr<Parameter>> allParameters;
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

    Parameter::Ptr addIntParam (juce::String, juce::String, juce::String, juce::String,
                                juce::NormalisableRange<float> range, float defaultValue,
                                SmoothingType st)
    {
        return createParam (range, defaultValue, st);
    }

private:
    Parameter::Ptr createParam (juce::NormalisableRange<float> range, float defaultValue, SmoothingType st)
    {
        std::unique_ptr<Parameter> p;
        if (st.time > 0.0f)
        {
            if (st.type == SmoothingType::linear)
            {
                auto sp = new SmoothedParameter<gin::ValueSmoother<float>> (range, defaultValue);
                sp->setSmoothingTime (st.time);
                p = std::unique_ptr<Parameter> (sp);
            }
            else
            {
                auto sp = new SmoothedParameter<gin::EasedValueSmoother<float>> (range, defaultValue);
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
};

}
