// gin::SoftClipper behind three knobs.
#include <JuceHeader.h>

namespace
{
class SoftClipperProcessor : public gin::Processor
{
public:
    SoftClipperProcessor()
    {
        drive   = addExtParam ("drive",   "Drive",   "", "dB", {   0.0f,  24.0f, 0.0f, 1.0f },  0.0f, 0.1f);
        knee    = addExtParam ("knee",    "Knee",    "", "%",  {   0.0f, 100.0f, 0.0f, 1.0f }, 50.0f, 0.1f);
        ceiling = addExtParam ("ceiling", "Ceiling", "", "dB", { -24.0f,   0.0f, 0.0f, 1.0f },  0.0f, 0.1f);
        init();
    }

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        gin::Processor::prepareToPlay (sampleRate, samplesPerBlock);
    }

    void processBlock (juce::AudioSampleBuffer& buffer, juce::MidiBuffer&) override
    {
        int numSamples = buffer.getNumSamples();
        clipper.setDrive (drive->getProcValue (numSamples));
        clipper.setKnee (knee->getProcValue (numSamples) / 100.0f);
        clipper.setCeiling (ceiling->getProcValue (numSamples));
        clipper.process (buffer);
    }

    gin::Parameter::Ptr drive, knee, ceiling;
    gin::SoftClipper clipper;
};
}

juce::AudioProcessor* create_SoftClipper() { return new SoftClipperProcessor(); }
