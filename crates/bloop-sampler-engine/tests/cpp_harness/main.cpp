/*
    Gold-test driver for Just-a-Sample's `CustomSamplerVoice`.

    Reads a program written by `tests/gold.rs` (argv[1]), drives the reference
    voice through it, and writes the rendered stereo output (argv[2]). The Rust
    port runs the same program, and the test compares the two sample by sample.

    The program format is documented alongside `Program` in gold.rs; both sides
    must agree on the parameter ids in `ParamId` below.
*/

#include <JuceHeader.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "Sampler/CustomSamplerVoice.h"

namespace juce
{
/** Grants the driver the access juce::Synthesiser normally has. */
struct VoiceDriver
{
    static void startNote (SynthesiserVoice& v, SynthesiserSound* sound, int note, float velocity, int wheel)
    {
        v.setCurrentlyPlaying (sound, note);
        v.startNote (note, velocity, sound, wheel);
    }

    static void pitchWheelMoved (SynthesiserVoice& v, int value) { v.pitchWheelMoved (value); }

    static void renderNextBlock (SynthesiserVoice& v, AudioBuffer<float>& out, int start, int num)
    {
        v.renderNextBlock (out, start, num);
    }
};
}  // namespace juce

namespace
{

/** Parameter ids, shared with the Rust side. Keep in sync with gold.rs. */
enum ParamId : uint32_t
{
    SemitoneTuning, CentTuning, PitchWheelRange, WideTuning, A4Hz,
    WaveformSemitoneTuning, WaveformCentTuning,
    SpeedFactor, OctaveSpeedFactor, SkipAntialiasing, DisableWavetableMode,
    PlayUntilEnd, IsLooping, LoopingHasStart, LoopingHasEnd,
    SampleStart, SampleEnd, LoopStart, LoopEnd,
    AttackMs, ReleaseMs, AttackShape, ReleaseShape, CrossfadeSamples,
    GainDb, MonoOutput, DisableVelocity,
    MidiStart, MidiEnd, MidiRoot, FollowMidiPitch,
};

enum OpKind : uint8_t
{
    OpRender = 0, OpNoteOn = 1, OpNoteOff = 2, OpSteal = 3, OpPitchWheel = 4, OpSetParam = 5,
};

class Reader
{
public:
    explicit Reader (const char* path)
    {
        auto* file = std::fopen (path, "rb");
        if (file == nullptr) { std::fprintf (stderr, "cannot open %s\n", path); std::exit (2); }
        std::fseek (file, 0, SEEK_END);
        bytes.resize (size_t (std::ftell (file)));
        std::fseek (file, 0, SEEK_SET);
        if (std::fread (bytes.data(), 1, bytes.size(), file) != bytes.size())
        {
            std::fprintf (stderr, "short read on %s\n", path);
            std::exit (2);
        }
        std::fclose (file);
    }

    template <typename T> T read()
    {
        T value {};
        if (offset + sizeof (T) > bytes.size()) { std::fprintf (stderr, "program truncated\n"); std::exit (2); }
        std::memcpy (&value, bytes.data() + offset, sizeof (T));
        offset += sizeof (T);
        return value;
    }

private:
    std::vector<uint8_t> bytes;
    size_t offset { 0 };
};

class HarnessSound final : public juce::SynthesiserSound
{
public:
    bool appliesToNote (int) override { return true; }
    bool appliesToChannel (int) override { return true; }
};

void applyParam (SamplerParameters& params, uint32_t id, double value)
{
    const auto asInt = int (value);
    const auto asFloat = float (value);
    const auto asBool = value != 0.0;

    switch (ParamId (id))
    {
        case SemitoneTuning:         params.semitoneTuningValue.set (asInt); break;
        case CentTuning:             params.centTuningValue.set (asInt); break;
        case PitchWheelRange:        params.pitchWheelRangeValue.set (asFloat); break;
        case WideTuning:             params.wideTuningValue.set (asFloat); break;
        case A4Hz:                   params.a4FreqValue.set (asFloat); break;
        case WaveformSemitoneTuning: params.waveformSemitoneTuningValue.set (asInt); break;
        case WaveformCentTuning:     params.waveformCentTuningValue.set (asInt); break;
        case SpeedFactor:            params.speedFactorValue.set (asFloat); break;
        case OctaveSpeedFactor:      params.octaveSpeedFactorValue.set (asFloat); break;
        case SkipAntialiasing:       params.skipAntialiasingValue.set (asBool); break;
        case DisableWavetableMode:   params.disableWavetableModeValue.set (asBool); break;
        case PlayUntilEnd:           params.playUntilEndValue.set (asBool); break;
        case IsLooping:              params.isLoopingValue.set (asBool); break;
        case LoopingHasStart:        params.loopingHasStartValue.set (asBool); break;
        case LoopingHasEnd:          params.loopingHasEndValue.set (asBool); break;
        case SampleStart:            params.sampleStart = asInt; break;
        case SampleEnd:              params.sampleEnd = asInt; break;
        case LoopStart:              params.loopStart = asInt; break;
        case LoopEnd:                params.loopEnd = asInt; break;
        case AttackMs:               params.attackValue.set (asFloat); break;
        case ReleaseMs:              params.releaseValue.set (asFloat); break;
        case AttackShape:            params.attackShapeValue.set (asFloat); break;
        case ReleaseShape:           params.releaseShapeValue.set (asFloat); break;
        case CrossfadeSamples:       params.crossfadeSamplesValue.set (asInt); break;
        case GainDb:                 params.gainValue.set (asFloat); break;
        case MonoOutput:             params.monoOutputValue.set (asBool); break;
        case DisableVelocity:        params.disableVelocityValue.set (asBool); break;
        case MidiStart:              params.midiStartValue.set (asInt); break;
        case MidiEnd:                params.midiEndValue.set (asInt); break;
        case MidiRoot:               params.midiRootValue.set (asInt); break;
        case FollowMidiPitch:        params.followMidiPitchValue.set (asBool); break;
        default: std::fprintf (stderr, "unknown parameter id %u\n", id); std::exit (2);
    }
}

}  // namespace

int main (int argc, char** argv)
{
    if (argc != 3)
    {
        std::fprintf (stderr, "usage: harness <program> <output>\n");
        return 2;
    }

    Reader reader { argv[1] };

    if (reader.read<uint32_t>() != 0x4a415331u)  // "JAS1"
    {
        std::fprintf (stderr, "bad magic\n");
        return 2;
    }

    const auto appSampleRate = reader.read<double>();
    const auto expectedBlockSize = int (reader.read<uint32_t>());
    const auto srcSampleRate = int (reader.read<uint32_t>());
    const auto numChannels = int (reader.read<uint32_t>());
    const auto numFrames = int (reader.read<uint32_t>());

    juce::AudioBuffer<float> sample { numChannels, numFrames };
    for (int ch = 0; ch < numChannels; ++ch)
        for (int i = 0; i < numFrames; ++i)
            sample.setSample (ch, i, reader.read<float>());

    SamplerParameters params { sample, srcSampleRate };

    const auto numParams = reader.read<uint32_t>();
    for (uint32_t i = 0; i < numParams; ++i)
    {
        const auto id = reader.read<uint32_t>();
        applyParam (params, id, reader.read<double>());
    }

    const auto totalFrames = int (reader.read<uint32_t>());
    juce::AudioBuffer<float> output { 2, totalFrames };
    output.clear();

    HarnessSound sound;
    CustomSamplerVoice voice { params, nullptr, appSampleRate, expectedBlockSize };

    int writePosition = 0;
    const auto numOps = reader.read<uint32_t>();
    for (uint32_t op = 0; op < numOps; ++op)
    {
        switch (OpKind (reader.read<uint8_t>()))
        {
            case OpRender:
            {
                const auto numSamples = int (reader.read<uint32_t>());
                juce::VoiceDriver::renderNextBlock (voice, output, writePosition, numSamples);
                writePosition += numSamples;
                break;
            }
            case OpNoteOn:
            {
                const auto note = int (reader.read<uint32_t>());
                const auto velocity = reader.read<float>();
                const auto wheel = int (reader.read<int32_t>());
                juce::VoiceDriver::startNote (voice, &sound, note, velocity, wheel);
                break;
            }
            case OpNoteOff:
                voice.stopNote (0.f, true);
                break;
            case OpSteal:
                voice.stopNote (0.f, false);
                break;
            case OpPitchWheel:
                juce::VoiceDriver::pitchWheelMoved (voice, int (reader.read<int32_t>()));
                break;
            case OpSetParam:
            {
                const auto id = reader.read<uint32_t>();
                applyParam (params, id, reader.read<double>());
                break;
            }
            default:
                std::fprintf (stderr, "unknown op\n");
                return 2;
        }
    }

    auto* out = std::fopen (argv[2], "wb");
    if (out == nullptr) { std::fprintf (stderr, "cannot write %s\n", argv[2]); return 2; }
    for (int ch = 0; ch < 2; ++ch)
        std::fwrite (output.getReadPointer (ch), sizeof (float), size_t (totalFrames), out);
    std::fclose (out);

    return 0;
}
