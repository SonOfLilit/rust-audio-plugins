// The out-of-line half of the Gin classes gin.h declares, as Gin's
// module units compile them.
#include <JuceHeader.h>

JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wfloat-equal")
#include <gin_dsp/3rdparty/AudioFilter/src/ButterworthCreator.cpp>
#include <gin_dsp/3rdparty/AudioFilter/src/ParametricCreator.cpp>
#include <gin_dsp/3rdparty/AudioFilter/src/Response.cpp>
JUCE_END_IGNORE_WARNINGS_GCC_LIKE

namespace gin
{
#include <gin/utilities/gin_riffparser.cpp>
#include <gin/utilities/gin_util.cpp>

#include <gin_dsp/dsp/gin_adsr.cpp>
#include <gin_dsp/dsp/gin_analogadsr.cpp>
#include <gin_dsp/dsp/gin_audioutil.cpp>
#include <gin_dsp/dsp/gin_bandlimitedlookuptable.cpp>
#include <gin_dsp/dsp/gin_wavetable.cpp>
#include <gin_dsp/dsp/gin_filter.cpp>
#include <gin_dsp/dsp/gin_lfo.cpp>
#include <gin_dsp/dsp/gin_modulation.cpp>
#include <gin_dsp/dsp/gin_oscillators.cpp>
#include <gin_dsp/dsp/gin_scratchbuffer.cpp>
#include <gin_dsp/dsp/gin_wtoscillators.cpp>

#include <gin_plugin/plugin/gin_noteduration.cpp>
#include <gin_plugin/plugin/gin_modmatrix.cpp>
}
