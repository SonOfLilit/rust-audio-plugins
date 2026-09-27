// gin::Maximizer behind three knobs, at its default 50 ms longest cycle.
#include <JuceHeader.h>

namespace
{
class MaximizerProcessor : public gin::Processor
{
public:
    MaximizerProcessor()
    {
        threshold = addExtParam ("threshold", "Threshold", "", "dB", { -24.0f,    0.0f, 0.0f, 1.0f },  -6.0f, 0.1f);
        ceiling   = addExtParam ("ceiling",   "Ceiling",   "", "dB", { -24.0f,    0.0f, 0.0f, 1.0f },  -0.3f, 0.1f);
        release   = addExtParam ("release",   "Release",   "", "ms", {   0.0f, 1000.0f, 0.0f, 0.5f }, 100.0f, 0.1f);
        init();
        maximizer.setNumChannels (2);
    }

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        gin::Processor::prepareToPlay (sampleRate, samplesPerBlock);
        maximizer.setSampleRate (sampleRate);
    }

    void reset() override
    {
        gin::Processor::reset();
        maximizer.reset();
    }

    void processBlock (juce::AudioSampleBuffer& buffer, juce::MidiBuffer&) override
    {
        int numSamples = buffer.getNumSamples();
        maximizer.setThreshold (threshold->getProcValue (numSamples));
        maximizer.setCeiling (ceiling->getProcValue (numSamples));
        maximizer.setRelease (release->getProcValue (numSamples) / 1000.0f);
        maximizer.process (buffer);
    }

    gin::Parameter::Ptr threshold, ceiling, release;
    gin::Maximizer maximizer;
};
}

juce::AudioProcessor* create_Maximizer() { return new MaximizerProcessor(); }
