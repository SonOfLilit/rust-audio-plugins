// gin::AirWindowsDistortion behind four knobs. Built with a 64-bit long
// double (tests/it/gold.rs), which is what the port computes in: the class
// works in long double, and that is 80 bits here, 128 on arm64 Linux and
// 64 on arm64 Apple, so there is no one original to match but the 64.
#include <JuceHeader.h>

namespace
{
class DistortionProcessor : public gin::Processor
{
public:
    DistortionProcessor()
    {
        density  = addExtParam ("density",  "Density",  "", "",  { -1.0f,   4.0f, 0.0f, 1.0f },   0.0f, 0.1f);
        highpass = addExtParam ("highpass", "Highpass", "", "%", {  0.0f, 100.0f, 0.0f, 1.0f },   0.0f, 0.1f);
        output   = addExtParam ("output",   "Output",   "", "%", {  0.0f, 100.0f, 0.0f, 1.0f }, 100.0f, 0.1f);
        mix      = addExtParam ("mix",      "Mix",      "", "%", {  0.0f, 100.0f, 0.0f, 1.0f }, 100.0f, 0.1f);
        init();
    }

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        gin::Processor::prepareToPlay (sampleRate, samplesPerBlock);
        distortion.setSampleRate (sampleRate);
    }

    void reset() override
    {
        gin::Processor::reset();
        distortion.reset();
    }

    void processBlock (juce::AudioSampleBuffer& buffer, juce::MidiBuffer&) override
    {
        int numSamples = buffer.getNumSamples();
        distortion.setParams ((density->getProcValue (numSamples) + 1.0f) / 5.0f,
                              highpass->getProcValue (numSamples) / 100.0f,
                              output->getProcValue (numSamples) / 100.0f,
                              mix->getProcValue (numSamples) / 100.0f);
        distortion.process (buffer);
    }

    gin::Parameter::Ptr density, highpass, output, mix;
    gin::AirWindowsDistortion distortion;
};
}

juce::AudioProcessor* create_Distortion() { return new DistortionProcessor(); }
