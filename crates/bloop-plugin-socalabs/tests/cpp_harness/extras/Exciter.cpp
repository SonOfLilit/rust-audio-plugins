// gin::Exciter behind four knobs.
#include <JuceHeader.h>

namespace
{
class ExciterProcessor : public gin::Processor
{
public:
    ExciterProcessor()
    {
        frequency = addExtParam ("frequency", "Frequency", "", "Hz", { 1000.0f, 16000.0f, 0.0f, 0.5f }, 7000.0f, 0.1f);
        drive     = addExtParam ("drive",     "Drive",     "", "",   {    1.0f,     8.0f, 0.0f, 1.0f },    1.0f, 0.1f);
        harmonics = addExtParam ("harmonics", "Harmonics", "", "%",  {    0.0f,   100.0f, 0.0f, 1.0f },    0.0f, 0.1f);
        mix       = addExtParam ("mix",       "Mix",       "", "%",  {    0.0f,   200.0f, 0.0f, 1.0f },   50.0f, 0.1f);
        init();
    }

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        gin::Processor::prepareToPlay (sampleRate, samplesPerBlock);
        exciter.setSampleRate (sampleRate);
    }

    void reset() override
    {
        gin::Processor::reset();
        exciter.reset();
    }

    void processBlock (juce::AudioSampleBuffer& buffer, juce::MidiBuffer&) override
    {
        int numSamples = buffer.getNumSamples();
        exciter.setFrequency (frequency->getProcValue (numSamples));
        exciter.setDrive (drive->getProcValue (numSamples));
        exciter.setHarmonics (harmonics->getProcValue (numSamples) / 100.0f);
        exciter.setMix (mix->getProcValue (numSamples) / 100.0f);
        exciter.process (buffer);
    }

    gin::Parameter::Ptr frequency, drive, harmonics, mix;
    gin::Exciter exciter;
};
}

juce::AudioProcessor* create_Exciter() { return new ExciterProcessor(); }
