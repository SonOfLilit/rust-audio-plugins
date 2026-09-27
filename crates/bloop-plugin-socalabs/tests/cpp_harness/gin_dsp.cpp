// The out-of-line half of the Gin classes gin_dsp.h declares, as
// gin_dsp.cpp compiles them.
#include "gin_dsp.h"

JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wfloat-equal")
#include <gin_dsp/3rdparty/AudioFilter/src/ButterworthCreator.cpp>
#include <gin_dsp/3rdparty/AudioFilter/src/ParametricCreator.cpp>
#include <gin_dsp/3rdparty/AudioFilter/src/Response.cpp>
JUCE_END_IGNORE_WARNINGS_GCC_LIKE

namespace gin
{
#include <gin_dsp/dsp/gin_analogadsr.cpp>
#include <gin_dsp/dsp/gin_audioutil.cpp>
#include <gin_dsp/dsp/gin_dynamics.cpp>
#include <gin_dsp/dsp/gin_filter.cpp>
#include <gin_dsp/dsp/gin_leveltracker.cpp>
#include <gin_dsp/dsp/gin_scratchbuffer.cpp>
#include <gin_dsp/dsp/gin_simpleverb.cpp>
#include <gin_dsp/dsp/gin_stereoenhancer.cpp>
#include <gin_dsp/dsp/gin_stereoprocessor.cpp>

#include <gin_plugin/plugin/gin_noteduration.cpp>
}
