/*
    Stands in for Just-a-Sample's `Source/Sampler/SamplerParameters.h`.

    The real class reads its members out of an `AudioProcessorValueTreeState`
    and out of `ListenableValue`s, both of which reach into JUCE's GUI module.
    The member names, types and accessors are identical, so the reference voice
    compiles against this unchanged — the harness just owns the parameter
    objects directly and sets them from the test program.
*/

#pragma once
#include <JuceHeader.h>

#include "../PluginParameters.h"

class SamplerParameters final
{
public:
    SamplerParameters (const juce::AudioBuffer<float>& sample, int sampleRate)
        : sample (sample), sampleRate (sampleRate)
    {
    }

    void sampleChanged (int newSampleRate) { sampleRate = newSampleRate; }

    PluginParameters::PLAYBACK_MODES getPlaybackMode() const
    {
        return static_cast<PluginParameters::PLAYBACK_MODES> (playbackModeIndex);
    }

    std::array<PluginParameters::FxTypes, 4> getFxOrder() const
    {
        return { PluginParameters::DISTORTION, PluginParameters::CHORUS,
                 PluginParameters::REVERB, PluginParameters::EQ };
    }

    const juce::AudioBuffer<float>& sample;
    int sampleRate;

    // Storage for every parameter the voice reads. The pointer members below
    // alias these, matching the reference's `AudioParameter*` members.
    juce::AudioParameterFloat gainValue { 0.f }, speedFactorValue { 1.f }, octaveSpeedFactorValue { 0.f },
        attackValue { 1.f }, releaseValue { 1.f }, attackShapeValue { 0.f }, releaseShapeValue { 2.f },
        a4FreqValue { 440.f }, pitchWheelRangeValue { 1.f }, wideTuningValue { 0.f };
    juce::AudioParameterInt semitoneTuningValue { 0 }, centTuningValue { 0 }, waveformSemitoneTuningValue { 0 },
        waveformCentTuningValue { 0 }, crossfadeSamplesValue { 1000 };
    juce::AudioParameterBool monoOutputValue { false }, disableVelocityValue { false }, skipAntialiasingValue { false },
        applyFXPreValue { false }, playUntilEndValue { false }, disableWavetableModeValue { false },
        isLoopingValue { false }, loopingHasStartValue { false }, loopingHasEndValue { false };
    juce::AudioParameterInt midiStartValue { 0 }, midiEndValue { 127 }, midiRootValue { 69 };
    juce::AudioParameterBool followMidiPitchValue { true };
    juce::AudioParameterBool reverbEnabledValue { false }, distortionEnabledValue { false },
        eqEnabledValue { false }, chorusEnabledValue { false };
    juce::AudioParameterFloat reverbPredelayValue { 0.f };

    int playbackModeIndex { PluginParameters::BASIC };

    juce::AudioParameterFloat* gain { &gainValue };
    juce::AudioParameterFloat* speedFactor { &speedFactorValue };
    juce::AudioParameterFloat* octaveSpeedFactor { &octaveSpeedFactorValue };
    juce::AudioParameterFloat* attack { &attackValue };
    juce::AudioParameterFloat* release { &releaseValue };
    juce::AudioParameterFloat* attackShape { &attackShapeValue };
    juce::AudioParameterFloat* releaseShape { &releaseShapeValue };
    juce::AudioParameterFloat* a4_freq { &a4FreqValue };
    juce::AudioParameterFloat* pitchWheelRange { &pitchWheelRangeValue };
    juce::AudioParameterFloat* wideTuningControl { &wideTuningValue };

    juce::AudioParameterInt* semitoneTuning { &semitoneTuningValue };
    juce::AudioParameterInt* centTuning { &centTuningValue };
    juce::AudioParameterInt* waveformSemitoneTuning { &waveformSemitoneTuningValue };
    juce::AudioParameterInt* waveformCentTuning { &waveformCentTuningValue };
    juce::AudioParameterInt* crossfadeSamples { &crossfadeSamplesValue };

    juce::AudioParameterBool* monoOutput { &monoOutputValue };
    juce::AudioParameterBool* disableVelocity { &disableVelocityValue };
    juce::AudioParameterBool* skipAntialiasing { &skipAntialiasingValue };
    juce::AudioParameterBool* applyFXPre { &applyFXPreValue };
    juce::AudioParameterBool* playUntilEnd { &playUntilEndValue };
    juce::AudioParameterBool* disableWavetableMode { &disableWavetableModeValue };
    juce::AudioParameterBool* isLooping { &isLoopingValue };
    juce::AudioParameterBool* loopingHasStart { &loopingHasStartValue };
    juce::AudioParameterBool* loopingHasEnd { &loopingHasEndValue };

    // The reference stores the bounds as `ListenableAtomic<int>&`, which the
    // voice reads through an implicit conversion to int. Plain ints behave the
    // same for the voice's purposes.
    int sampleStart { 0 }, sampleEnd { 0 }, loopStart { 0 }, loopEnd { 0 };

    juce::AudioParameterInt* midiStart { &midiStartValue };
    juce::AudioParameterInt* midiEnd { &midiEndValue };
    juce::AudioParameterInt* midiRoot { &midiRootValue };
    juce::AudioParameterBool* followMidiPitch { &followMidiPitchValue };

    juce::AudioParameterBool* reverbEnabled { &reverbEnabledValue };
    juce::AudioParameterBool* distortionEnabled { &distortionEnabledValue };
    juce::AudioParameterBool* eqEnabled { &eqEnabledValue };
    juce::AudioParameterBool* chorusEnabled { &chorusEnabledValue };
    juce::AudioParameterFloat* reverbPredelay { &reverbPredelayValue };
};
