// The part of Gin Wavetable is built from: its utilities, its SIMD maths
// (mipp, so the wavetable oscillator takes the SIMD path a desktop build
// takes), gin_dsp's classes, and from gin_plugin the modulation matrix and
// note durations — included as Gin's module headers include them — plus
// the stand-in gin::Parameter and gin::Processor (see shim/JuceHeader.h).
#pragma once

#include <map>
#include <random>

JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wfloat-equal")
#include <gin_dsp/3rdparty/AudioFilter/src/AudioFilterTypes.h>
#include <gin_dsp/3rdparty/AudioFilter/src/FilterInstance.h>
#include <gin_dsp/3rdparty/AudioFilter/src/ParametricCreator.h>
#include <gin_dsp/3rdparty/AudioFilter/src/Response.h>
JUCE_END_IGNORE_WARNINGS_GCC_LIKE

#if JUCE_INTEL
 #ifndef __SSE__
  #define __SSE__ 1
 #endif
 #ifndef __SSE2__
  #define __SSE2__ 1
 #endif
#endif
#define GIN_HAS_SIMD 1
#include <gin_simd/mipp/mipp.h>

namespace gin
{
#include <gin/utilities/gin_point.h>
#include <gin/utilities/gin_util.h>
#include <gin/utilities/gin_easing.h>
#include <gin/utilities/gin_lagrange.h>
#include <gin/utilities/gin_riffparser.h>

#include <gin_simd/math/gin_math.h>

#include <gin_dsp/dsp/gin_audioutil.h>
#include <gin_dsp/dsp/gin_scratchbuffer.h>
#include <gin_dsp/dsp/gin_adsr.h>
#include <gin_dsp/dsp/gin_analogadsr.h>
#include <gin_dsp/dsp/gin_audiofifo.h>
#include <gin_dsp/dsp/gin_bandlimitedlookuptable.h>
#include <gin_dsp/dsp/gin_wavetable.h>
#include <gin_dsp/dsp/gin_distortion.h>
#include <gin_dsp/dsp/gin_delayline.h>
#include <gin_dsp/dsp/gin_easedvaluesmoother.h>
#include <gin_dsp/dsp/gin_fastmath.h>
#include <gin_dsp/dsp/gin_filter.h>
#include <gin_dsp/dsp/gin_gateeffect.h>
#include <gin_dsp/dsp/gin_lfo.h>
#include <gin_dsp/dsp/gin_modulation.h>
#include <gin_dsp/dsp/gin_noise.h>
#include <gin_dsp/dsp/gin_oscillators.h>
#include <gin_dsp/dsp/gin_platereverb.h>
#include <gin_dsp/dsp/gin_steplfo.h>
#include <gin_dsp/dsp/gin_stereodelay.h>
#include <gin_dsp/dsp/gin_synthesiser.h>
#include <gin_dsp/dsp/gin_valuesmoother.h>
#include <gin_dsp/dsp/gin_dcblocker.h>
#include <gin_dsp/dsp/gin_wtoscillators.h>

class Parameter;
class Processor;
class MidiLearn;

#include <gin_plugin/plugin/gin_noteduration.h>
#include <gin_plugin/plugin/gin_imodmatrix.h>

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
    using ConversionFunction = std::function<std::variant<float, juce::String> (const Parameter&, const std::variant<float, juce::String>&)>;

    Parameter (juce::String uid_, juce::String name_, juce::NormalisableRange<float> range_, float defaultValue_, bool internal_)
        : range (range_), value (defaultValue_), defaultValue (defaultValue_), uid (uid_), name (name_), internal (internal_) {}

    juce::String getUid() const             { return uid; }
    juce::String getName (int) const        { return name; }
    bool isInternal() const                 { return internal; }
    void setModIndex (int i)                { modIndex = i; }
    int getModIndex() const                 { return modIndex; }
    void setModMatrix (IModMatrix* m)       { modMatrix = m; }

    float getProcValue()
    {
        if (conversionFunction != nullptr)
            return conversionFunction (getUserValue());
        return getUserValue();
    }

    float getUserValue() const              { return juce::jlimit (range.start, range.end, value); }
    int getUserValueInt() const             { return int (juce::jlimit (range.start, range.end, value)); }
    float getUserDefaultValue() const       { return defaultValue; }
    bool isOn()                             { return ! juce::exactlyEqual (range.start, getUserValue()); }
    float getValue() const                  { return juce::jlimit (0.0f, 1.0f, range.convertTo0to1 (value)); }
    bool getBoolValue() const               { return getValue() != 0.0f; }
    juce::NormalisableRange<float> getUserRange() const { return range; }

    void setUserValue (float v)
    {
        v = juce::jlimit (range.start, range.end, range.snapToLegalValue (v));
        if (! juce::approximatelyEqual (value, v))
            value = v;
    }

    void setUserValueNotifingHost (float v)     { setUserValue (v); }

    std::function<float (float)> conversionFunction;

private:
    const juce::NormalisableRange<float> range;
    float value;
    const float defaultValue;
    const juce::String uid, name;
    const bool internal;
    IModMatrix* modMatrix = nullptr;
    int modIndex = -1;
};

class MidiLearn
{
public:
    void processBlock (juce::MidiBuffer&, int) {}
};

class ProcessorOptions
{
public:
    ProcessorOptions withMidiLearn() const  { return *this; }
};

class Processor : public juce::AudioProcessor
{
public:
    Processor (const BusesProperties&, bool = false, ProcessorOptions = {}) {}

    void prepareToPlay (double, int) override   {}
    void reset() override                       {}

    virtual void init()                         {}
    virtual juce::PropertiesFile* getSettings() { return nullptr; }
    virtual bool isParamLocked (Parameter*)     { return false; }
    virtual juce::File getProgramDirectory()    { return {}; }
    virtual juce::Array<juce::File> getFactoryProgramDirectories() { return {}; }
    virtual void stateUpdated()                 {}
    virtual void updateState()                  {}

    const juce::Array<Parameter*>& getPluginParameters()   { return parameters; }

    Parameter* getParameter (const juce::String& uid)
    {
        for (auto p : parameters)
            if (p->getUid() == uid)
                return p;
        return nullptr;
    }

    using TextFunction = std::function<juce::String (const Parameter&, float)>;

    Parameter::Ptr addExtParam (juce::String uid, juce::String name, juce::String, juce::String,
                                juce::NormalisableRange<float> range, float defaultValue,
                                SmoothingType, TextFunction = nullptr)
    {
        return add (uid, name, range, defaultValue, false);
    }

    Parameter::Ptr addExtParam (juce::String uid, juce::String name, juce::String, juce::String,
                                juce::NormalisableRange<float> range, float defaultValue,
                                SmoothingType, Parameter::ConversionFunction)
    {
        return add (uid, name, range, defaultValue, false);
    }

    Parameter::Ptr addIntParam (juce::String uid, juce::String name, juce::String, juce::String,
                                juce::NormalisableRange<float> range, float defaultValue,
                                SmoothingType, TextFunction = nullptr)
    {
        return add (uid, name, range, defaultValue, true);
    }

    Parameter::Ptr addIntParam (juce::String uid, juce::String name, juce::String, juce::String,
                                juce::NormalisableRange<float> range, float defaultValue,
                                SmoothingType, Parameter::ConversionFunction)
    {
        return add (uid, name, range, defaultValue, true);
    }

    juce::ValueTree state { "state" };
    int versionHint = 1;
    std::unique_ptr<MidiLearn> midiLearn;

private:
    Parameter::Ptr add (juce::String uid, juce::String name, juce::NormalisableRange<float> range, float defaultValue, bool internal)
    {
        owned.push_back (std::make_unique<Parameter> (uid, name, range, defaultValue, internal));
        parameters.add (owned.back().get());
        return owned.back().get();
    }

    std::vector<std::unique_ptr<Parameter>> owned;
    juce::Array<Parameter*> parameters;
};

#include <gin_plugin/plugin/gin_modmatrix.h>
}
