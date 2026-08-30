/*
    A minimal JUCE stand-in, just large enough to compile Just-a-Sample's
    `CustomSamplerVoice` for the gold tests.

    Every class here is either trivial plumbing (buffers, parameter holders) or
    a faithful transcription of the JUCE 8 original where the maths reaches the
    audio output — `IIRCoefficients`, `IIRFilterBase` and `Decibels` are copied
    from `modules/juce_audio_basics`. If a transcription drifts, the gold tests
    fail, because the Rust port under test implements the same maths.

    This file is test-only: it is never linked into Bloop.
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define DBG(x)
#define jassert(x)
#define ignoreUnused(x) (void) (x)

namespace juce
{

using String = std::string;

struct StringArray
{
    std::vector<String> strings;

    StringArray() = default;
    StringArray (std::initializer_list<const char*> items)
    {
        for (auto* item : items)
            strings.emplace_back (item);
    }

    int size() const { return int (strings.size()); }
    const String& operator[] (int i) const { return strings[size_t (i)]; }
};

template <typename T>
struct MathConstants
{
    static constexpr T pi = static_cast<T> (3.141592653589793238L);
    static constexpr T sqrt2 = static_cast<T> (1.4142135623730950488L);
};

template <typename T> T jmin (T a, T b)              { return a < b ? a : b; }
template <typename T> T jmax (T a, T b)              { return a < b ? b : a; }
template <typename T> T jlimit (T low, T high, T v)  { return v < low ? low : (high < v ? high : v); }

template <typename T>
T jmap (T sourceValue, T sourceMin, T sourceMax, T targetMin, T targetMax)
{
    return targetMin + ((targetMax - targetMin) * (sourceValue - sourceMin)) / (sourceMax - sourceMin);
}

template <typename T>
struct Tolerance
{
    Tolerance withAbsolute (T t) const { auto copy = *this; copy.absolute = t; return copy; }
    T absolute { };
};

template <typename T>
bool approximatelyEqual (T a, T b, Tolerance<T> tolerance)
{
    return a == b || std::abs (a - b) <= tolerance.absolute;
}

template <typename T>
struct Range
{
    constexpr Range() = default;
    constexpr Range (T s, T e) : start (s), end (e) {}
    T start { }, end { };
};

//==============================================================================
template <typename SampleType>
class AudioBuffer
{
public:
    AudioBuffer() = default;
    AudioBuffer (int numChannels, int numSamples) { setSize (numChannels, numSamples); }

    void setSize (int newNumChannels, int newNumSamples,
                  bool keepExistingContent = false, bool clearExtraSpace = false,
                  bool avoidReallocating = false)
    {
        ignoreUnused (avoidReallocating);
        std::vector<std::vector<SampleType>> replacement (size_t (std::max (0, newNumChannels)),
                                                     std::vector<SampleType> (size_t (std::max (0, newNumSamples)), SampleType()));

        if (keepExistingContent)
            for (size_t ch = 0; ch < std::min (replacement.size(), channels.size()); ++ch)
                for (size_t i = 0; i < std::min (replacement[ch].size(), channels[ch].size()); ++i)
                    replacement[ch][i] = channels[ch][i];

        ignoreUnused (clearExtraSpace);
        channels = std::move (replacement);
    }

    void clear()
    {
        for (auto& channel : channels)
            std::fill (channel.begin(), channel.end(), SampleType());
    }

    int getNumChannels() const { return int (channels.size()); }
    int getNumSamples()  const { return channels.empty() ? 0 : int (channels[0].size()); }

    SampleType getSample (int channel, int index) const { return channels[size_t (channel)][size_t (index)]; }
    void       setSample (int channel, int index, SampleType value) { channels[size_t (channel)][size_t (index)] = value; }
    void       addSample (int channel, int index, SampleType value) { channels[size_t (channel)][size_t (index)] += value; }

    const SampleType* getReadPointer  (int channel, int startIndex = 0) const { return channels[size_t (channel)].data() + startIndex; }
    SampleType*       getWritePointer (int channel, int startIndex = 0)       { return channels[size_t (channel)].data() + startIndex; }

    void copyFrom (int destChannel, int destStartSample, const AudioBuffer<SampleType>& source,
                   int sourceChannel, int sourceStartSample, int numSamples)
    {
        for (int i = 0; i < numSamples; ++i)
            setSample (destChannel, destStartSample + i, source.getSample (sourceChannel, sourceStartSample + i));
    }

    void addFrom (int destChannel, int destStartSample, const SampleType* source, int numSamples, SampleType gain = SampleType (1))
    {
        for (int i = 0; i < numSamples; ++i)
            addSample (destChannel, destStartSample + i, source[i] * gain);
    }

    SampleType getRMSLevel (int channel, int startSample, int numSamples) const
    {
        if (numSamples <= 0)
            return 0.0f;

        double sum = 0.0;
        for (int i = 0; i < numSamples; ++i)
        {
            auto value = double (getSample (channel, startSample + i));
            sum += value * value;
        }
        return SampleType (std::sqrt (sum / numSamples));
    }

private:
    std::vector<std::vector<SampleType>> channels;
};

//==============================================================================
struct FloatVectorOperations
{
    static void multiply (float* dest, const float* src1, const float* src2, int num)
    {
        for (int i = 0; i < num; ++i)
            dest[i] = src1[i] * src2[i];
    }
};

//==============================================================================
/** Transcribed from juce_Decibels.h. */
struct Decibels
{
    template <typename Type>
    static Type decibelsToGain (Type decibels, Type minusInfinityDb = Type (-100.0))
    {
        return decibels > minusInfinityDb ? std::pow (Type (10.0), decibels * Type (0.05))
                                          : Type();
    }
};

//==============================================================================
/** Transcribed from juce_IIRFilter.cpp / .h. */
class IIRCoefficients
{
public:
    IIRCoefficients() = default;

    IIRCoefficients (double c1, double c2, double c3, double c4, double c5, double c6) noexcept
    {
        const auto a = 1.0 / c4;

        coefficients[0] = (float) (c1 * a);
        coefficients[1] = (float) (c2 * a);
        coefficients[2] = (float) (c3 * a);
        coefficients[3] = (float) (c5 * a);
        coefficients[4] = (float) (c6 * a);
    }

    static IIRCoefficients makeLowPass (double sampleRate, double frequency, double Q) noexcept
    {
        const auto n = 1.0 / std::tan (MathConstants<double>::pi * frequency / sampleRate);
        const auto nSquared = n * n;
        const auto c1 = 1.0 / (1.0 + 1.0 / Q * n + nSquared);

        return IIRCoefficients (c1,
                                c1 * 2.0,
                                c1,
                                1.0,
                                c1 * 2.0 * (1.0 - nSquared),
                                c1 * (1.0 - 1.0 / Q * n + nSquared));
    }

    float coefficients[5] { };
};

/** JUCE only compiles JUCE_SNAP_TO_ZERO in on Intel; Bloop's port applies it
    everywhere so its output does not change between desktop and phone, so the
    reference build here does the same. */
class SingleThreadedIIRFilter
{
public:
    void setCoefficients (const IIRCoefficients& newCoefficients) noexcept { coefficients = newCoefficients; }

    void reset() noexcept { v1 = v2 = 0.0f; }

    float processSingleSampleRaw (float in) noexcept
    {
        auto out = coefficients.coefficients[0] * in + v1;

        if (! (out < -1.0e-8f || out > 1.0e-8f)) out = 0;

        v1 = coefficients.coefficients[1] * in - coefficients.coefficients[3] * out + v2;
        v2 = coefficients.coefficients[2] * in - coefficients.coefficients[4] * out;

        return out;
    }

private:
    IIRCoefficients coefficients;
    float v1 = 0, v2 = 0;
};

//==============================================================================
struct MidiMessage
{
    static String getMidiNoteName (int, bool, bool, int) { return {}; }
};

struct MemoryBlock
{
    std::vector<uint8_t> bytes;
    void append (const void* data, size_t numBytes)
    {
        auto* start = static_cast<const uint8_t*> (data);
        bytes.insert (bytes.end(), start, start + numBytes);
    }
};

struct MD5
{
    explicit MD5 (const MemoryBlock&) {}
    String toHexString() const { return {}; }
};

//==============================================================================
class RangedAudioParameter
{
public:
    virtual ~RangedAudioParameter() = default;
};

class AudioParameterFloat final : public RangedAudioParameter
{
public:
    explicit AudioParameterFloat (float initial) : value (initial) {}
    float get() const { return value; }
    void set (float v) { value = v; }

private:
    float value { };
};

class AudioParameterInt final : public RangedAudioParameter
{
public:
    explicit AudioParameterInt (int initial) : value (initial) {}
    int get() const { return value; }
    void set (int v) { value = v; }

private:
    int value { };
};

class AudioParameterBool final : public RangedAudioParameter
{
public:
    explicit AudioParameterBool (bool initial) : value (initial) {}
    bool get() const { return value; }
    void set (bool v) { value = v; }

private:
    bool value { };
};

class AudioParameterChoice final : public RangedAudioParameter
{
public:
    explicit AudioParameterChoice (int initial) : index (initial) {}
    int getIndex() const { return index; }
    void set (int v) { index = v; }

private:
    int index { };
};

//==============================================================================
class SynthesiserSound
{
public:
    virtual ~SynthesiserSound() = default;
    virtual bool appliesToNote (int) = 0;
    virtual bool appliesToChannel (int) = 0;
};

class SynthesiserVoice
{
public:
    virtual ~SynthesiserVoice() = default;

    void setCurrentPlaybackSampleRate (double newRate) { sampleRate = newRate; }
    double getSampleRate() const { return sampleRate; }

    SynthesiserSound* getCurrentlyPlayingSound() const { return currentSound; }
    int getCurrentlyPlayingNote() const { return currentNote; }

    void clearCurrentNote() { currentSound = nullptr; currentNote = -1; }

    /** The harness stands in for juce::Synthesiser, which normally assigns
        these before calling startNote. */
    void setCurrentlyPlaying (SynthesiserSound* sound, int note) { currentSound = sound; currentNote = note; }

    virtual void stopNote (float velocity, bool allowTailOff) = 0;

private:
    virtual bool canPlaySound (SynthesiserSound*) = 0;
    virtual void startNote (int midiNoteNumber, float velocity, SynthesiserSound* sound, int currentPitchWheelPosition) = 0;
    virtual void pitchWheelMoved (int newPitchWheelValue) = 0;
    virtual void controllerMoved (int, int) = 0;
    virtual void renderNextBlock (AudioBuffer<float>& outputBuffer, int startSample, int numSamples) = 0;

    friend struct VoiceDriver;

    double sampleRate { 0.0 };
    SynthesiserSound* currentSound { nullptr };
    int currentNote { -1 };
};

}  // namespace juce
