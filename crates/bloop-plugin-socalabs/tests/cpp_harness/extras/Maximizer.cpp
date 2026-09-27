// gin::Maximizer behind three knobs, at its default 50 ms longest cycle.
// What comes out is held to the ceiling: a half cycle longer than the
// longest (DC, anything under 10 Hz) is never scaled, and would otherwise
// get the whole makeup gain.
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
        auto ceilingDb = ceiling->getProcValue (numSamples);
        maximizer.setThreshold (threshold->getProcValue (numSamples));
        maximizer.setCeiling (ceilingDb);
        maximizer.setRelease (release->getProcValue (numSamples) / 1000.0f);
        maximizer.process (buffer);

        auto limit = juce::Decibels::decibelsToGain (ceilingDb);
        for (int ch = 0; ch < buffer.getNumChannels(); ch++)
        {
            auto* d = buffer.getWritePointer (ch);
            for (int i = 0; i < numSamples; i++)
                d[i] = juce::jlimit (-limit, limit, d[i]);
        }
    }

    gin::Parameter::Ptr threshold, ceiling, release;
    gin::Maximizer maximizer;
};
}

juce::AudioProcessor* create_Maximizer() { return new MaximizerProcessor(); }
