/*
    Stands in for Just-a-Sample's `PluginParameters.h`.

    Only the constants and enums `CustomSamplerVoice` reads are kept; the real
    header also builds the plugin's APVTS layout, which drags in JUCE's GUI
    module. Values here are copied from the pinned reference.
*/

#pragma once
#include <JuceHeader.h>

#include <array>

namespace PluginParameters
{

enum PLAYBACK_MODES : std::uint8_t
{
    BASIC,
    BUNGEE,
};

enum FxTypes : std::uint8_t
{
    DISTORTION,
    CHORUS,
    REVERB,
    EQ
};

inline static constexpr float WAVETABLE_CUTOFF_HZ { 20 };
inline static constexpr float FX_TAIL_OFF_MAX { 0.0001f };

}  // namespace PluginParameters
