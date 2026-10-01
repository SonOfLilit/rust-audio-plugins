// gin::Modulation as SocaLabs Wavetable drives it for its chorus
// (Wavetable rev 5f932ff87c453d32cebee233af1fde946f53ac89,
// plugin/Source/PluginProcessor.cpp: ChorusParams::setup, prepareToPlay,
// reset, updateParams), behind Wavetable's five knobs. Its ranges and
// defaults are Wavetable's; Width and Mix are percentages where Wavetable
// has 0 to 1, and Delay and Depth are converted from milliseconds where
// the knobs are read, once a block, as Wavetable's conversionFunction does.
// They smooth over a tenth of a second, as every other extra's knobs do.
//
// One difference from Wavetable: the class sizes its delay line from the
// depth set last, so prepareToPlay puts the class's own defaults back
// before sizing it. The line is then the size Wavetable's first
// prepareToPlay makes, whatever the knobs did since, and a host that
// prepares the effect again at the same rate does not rebuild it.
#include <JuceHeader.h>

namespace
{
class ChorusProcessor : public gin::Processor
{
public:
    ChorusProcessor()
    {
        delay = addExtParam ("delay", "Delay", "", "ms", { 0.1f,  30.0f, 0.0f, 1.0f },  1.0f, 0.1f);
        depth = addExtParam ("depth", "Depth", "", "ms", { 0.1f,  20.0f, 0.0f, 1.0f },  1.0f, 0.1f);
        speed = addExtParam ("speed", "Speed", "", "Hz", { 0.1f,  10.0f, 0.0f, 1.0f },  3.0f, 0.1f);
        width = addExtParam ("width", "Width", "", "%",  { 0.0f, 100.0f, 0.0f, 1.0f }, 50.0f, 0.1f);
        mix   = addExtParam ("mix",   "Mix",   "", "%",  { 0.0f, 100.0f, 0.0f, 1.0f }, 50.0f, 0.1f);
        init();
    }

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        gin::Processor::prepareToPlay (sampleRate, samplesPerBlock);
        chorus.setParams (0.02f, 1.0f, 0.03f, 0.5f, 0.0f);
        chorus.setSampleRate (sampleRate);
    }

    void reset() override
    {
        gin::Processor::reset();
        chorus.reset();
    }

    void processBlock (juce::AudioSampleBuffer& buffer, juce::MidiBuffer&) override
    {
        int numSamples = buffer.getNumSamples();
        float d = delay->getProcValue (numSamples);
        float dp = depth->getProcValue (numSamples);
        float sp = speed->getProcValue (numSamples);
        float w = width->getProcValue (numSamples);
        float m = mix->getProcValue (numSamples);
        chorus.setParams (d / 1000.0f, sp, dp / 1000.0f, w / 100.0f, m / 100.0f);
        chorus.process (buffer);
    }

    gin::Parameter::Ptr delay, depth, speed, width, mix;
    gin::Modulation chorus { 0.5f };
};
}

juce::AudioProcessor* create_Chorus() { return new ChorusProcessor(); }
