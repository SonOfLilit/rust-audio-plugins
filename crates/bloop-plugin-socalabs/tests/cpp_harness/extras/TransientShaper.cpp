// gin::TransientShaper behind eight knobs. A cut at 0 Hz is off.
#include <JuceHeader.h>

namespace
{
class TransientShaperProcessor : public gin::Processor
{
public:
    TransientShaperProcessor()
    {
        attack    = addExtParam ("attack",    "Attack",    "", "",   { -4.0f,     4.0f, 0.0f, 1.0f },  0.0f, 0.1f);
        sustain   = addExtParam ("sustain",   "Sustain",   "", "",   { -8.0f,     8.0f, 0.0f, 1.0f },  0.0f, 0.1f);
        smoothing = addExtParam ("smoothing", "Smoothing", "", "%",  {  0.0f,   100.0f, 0.0f, 1.0f }, 50.0f, 0.1f);
        lowCut    = addExtParam ("lowcut",    "Low Cut",   "", "Hz", {  0.0f,  2000.0f, 0.0f, 0.5f },  0.0f, 0.0f);
        highCut   = addExtParam ("highcut",   "High Cut",  "", "Hz", {  0.0f, 20000.0f, 0.0f, 0.3f },  0.0f, 0.0f);
        detector  = addExtParam ("detector",  "Detector",  "", "%",  {  0.0f,   100.0f, 0.0f, 1.0f }, 50.0f, 0.1f);
        balance   = addExtParam ("process",   "Process",   "", "%",  {  0.0f,   100.0f, 0.0f, 1.0f }, 50.0f, 0.1f);
        output    = addExtParam ("output",    "Output",    "", "dB", {-24.0f,    24.0f, 0.0f, 1.0f },  0.0f, 0.1f);
        init();
    }

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        gin::Processor::prepareToPlay (sampleRate, samplesPerBlock);
        shaper.setSampleRate (sampleRate);
    }

    void reset() override
    {
        gin::Processor::reset();
        shaper.reset();
    }

    void processBlock (juce::AudioSampleBuffer& buffer, juce::MidiBuffer&) override
    {
        int numSamples = buffer.getNumSamples();
        shaper.setAttack (attack->getProcValue (numSamples));
        shaper.setSustain (sustain->getProcValue (numSamples));
        shaper.setSmoothing (smoothing->getProcValue (numSamples) / 100.0f);
        shaper.setBand (lowCut->getProcValue (numSamples), highCut->getProcValue (numSamples));
        shaper.setDetectorBalance (detector->getProcValue (numSamples) / 100.0f);
        shaper.setProcessBalance (balance->getProcValue (numSamples) / 100.0f);
        shaper.setOutputGain (output->getProcValue (numSamples));
        shaper.process (buffer);
    }

    gin::Parameter::Ptr attack, sustain, smoothing, lowCut, highCut, detector, balance, output;
    gin::TransientShaper shaper;
};
}

juce::AudioProcessor* create_TransientShaper() { return new TransientShaperProcessor(); }
