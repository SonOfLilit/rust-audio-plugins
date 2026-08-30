/*
    Stands in for Just-a-Sample's `Stretcher.h`.

    The real header wraps the Bungee time stretcher, which drives the
    reference's ADVANCED playback mode. Bloop ports BASIC mode only, so the
    gold tests never reach these calls; the stub exists purely so the reference
    voice compiles without vendoring Bungee. Every method traps.
*/

#pragma once
#include <JuceHeader.h>

#include <stdexcept>

class BungeeStretcher
{
public:
    BungeeStretcher (const juce::AudioBuffer<float>& sampleBuffer, int sampleRate)
        : buffer (&sampleBuffer), bufferSampleRate (sampleRate) {}

    void preallocateStretcher (int) {}
    void initialize (long double, float = 1, float = 1) { unreachable(); }
    void preroll (double) { unreachable(); }
    float nextSample (int, bool = true) { unreachable(); }
    void setPitchAndSpeed (float, float) {}
    float getPositionSpeed() const { return 0.f; }

private:
    [[noreturn]] static float unreachable()
    {
        throw std::logic_error ("ADVANCED playback mode is out of scope for the gold tests");
    }

    const juce::AudioBuffer<float>* buffer { nullptr };
    int bufferSampleRate { 0 };
};
