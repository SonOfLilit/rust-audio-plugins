// gin::PlateReverb behind six knobs. Percentages and milliseconds are
// converted where the knobs are read, once a block.
#include <JuceHeader.h>

namespace
{
class PlateReverbProcessor : public gin::Processor
{
public:
    PlateReverbProcessor()
    {
        mix      = addExtParam ("mix",      "Mix",      "", "%",  {  0.0f,   100.0f, 0.0f, 1.0f },    25.0f, 0.1f);
        predelay = addExtParam ("predelay", "Predelay", "", "ms", {  0.0f,   100.0f, 0.0f, 1.0f },     0.0f, 0.0f);
        lowpass  = addExtParam ("lowpass",  "Lowpass",  "", "Hz", { 16.0f, 20000.0f, 0.0f, 0.3f }, 10000.0f, 0.1f);
        decay    = addExtParam ("decay",    "Decay",    "", "%",  {  0.0f,   100.0f, 0.0f, 1.0f },    50.0f, 0.1f);
        size     = addExtParam ("size",     "Size",     "", "%",  {  0.0f,   200.0f, 0.0f, 1.0f },   100.0f, 0.0f);
        damping  = addExtParam ("damping",  "Damping",  "", "Hz", { 16.0f, 20000.0f, 0.0f, 0.3f }, 10000.0f, 0.1f);
        init();
    }

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        gin::Processor::prepareToPlay (sampleRate, samplesPerBlock);
        reverb.setSampleRate (float (sampleRate));
    }

    void reset() override
    {
        gin::Processor::reset();
        reverb.reset();
    }

    void processBlock (juce::AudioSampleBuffer& buffer, juce::MidiBuffer&) override
    {
        int numSamples = buffer.getNumSamples();
        reverb.setMix (mix->getProcValue (numSamples) / 100.0f);
        reverb.setPredelay (predelay->getProcValue (numSamples) / 1000.0f);
        reverb.setLowpass (lowpass->getProcValue (numSamples));
        reverb.setDecay (decay->getProcValue (numSamples) / 100.0f);
        reverb.setSize (size->getProcValue (numSamples) / 100.0f);
        reverb.setDamping (damping->getProcValue (numSamples));

        auto* l = buffer.getWritePointer (0);
        auto* r = buffer.getWritePointer (1);
        for (int i = 0; i < numSamples; i++)
            reverb.process (l[i], r[i], &l[i], &r[i]);
    }

    gin::Parameter::Ptr mix, predelay, lowpass, decay, size, damping;
    gin::PlateReverb<float, int> reverb;
};
}

juce::AudioProcessor* create_PlateReverb() { return new PlateReverbProcessor(); }
