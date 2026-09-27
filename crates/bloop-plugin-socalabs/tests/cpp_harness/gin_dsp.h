// The part of Gin the ported effects are built from: gin_dsp's DSP classes
// and the few gin utilities they use, included exactly as gin_dsp.h
// includes them, without the UI components or the rest of the module.
#pragma once

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include <juce_data_structures/juce_data_structures.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>

#include <map>
#include <random>

JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wfloat-equal")
#include <gin_dsp/3rdparty/AudioFilter/src/AudioFilterTypes.h>
#include <gin_dsp/3rdparty/AudioFilter/src/FilterInstance.h>
#include <gin_dsp/3rdparty/AudioFilter/src/ParametricCreator.h>
#include <gin_dsp/3rdparty/AudioFilter/src/Response.h>
JUCE_END_IGNORE_WARNINGS_GCC_LIKE

namespace gin
{
#include <gin/utilities/gin_point.h>
#include <gin/utilities/gin_util.h>
#include <gin/utilities/gin_easing.h>
#include <gin/utilities/gin_lagrange.h>

#include <gin_dsp/dsp/gin_audioutil.h>
#include <gin_dsp/dsp/gin_leveltracker.h>
#include <gin_dsp/dsp/gin_scratchbuffer.h>
#include <gin_dsp/dsp/gin_analogadsr.h>
#include <gin_dsp/dsp/gin_biquad.h>
#include <gin_dsp/dsp/gin_distortion.h>
#include <gin_dsp/dsp/gin_delayline.h>
#include <gin_dsp/dsp/gin_dynamics.h>
#include <gin_dsp/dsp/gin_easedvaluesmoother.h>
#include <gin_dsp/dsp/gin_fastmath.h>
#include <gin_dsp/dsp/gin_filter.h>
#include <gin_dsp/dsp/gin_eq.h>
#include <gin_dsp/dsp/gin_graphiceq.h>
#include <gin_dsp/dsp/gin_platereverb.h>
#include <gin_dsp/dsp/gin_simpleverb.h>
#include <gin_dsp/dsp/gin_stereodelay.h>
#include <gin_dsp/dsp/gin_stereoenhancer.h>
#include <gin_dsp/dsp/gin_stereoprocessor.h>
#include <gin_dsp/dsp/gin_valuesmoother.h>
#include <gin_dsp/dsp/gin_dcblocker.h>
#include <gin_dsp/dsp/gin_bitcrusher.h>
#include <gin_dsp/dsp/gin_exciter.h>
#include <gin_dsp/dsp/gin_maximizer.h>
#include <gin_dsp/dsp/gin_softclipper.h>
#include <gin_dsp/dsp/gin_transientshaper.h>

#include <gin_plugin/plugin/gin_noteduration.h>
}
