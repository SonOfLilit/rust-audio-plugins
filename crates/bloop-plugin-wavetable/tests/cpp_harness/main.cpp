// The gold harness: runs Wavetable's own WavetableAudioProcessor over a
// program read from stdin, and writes what it produced to stdout.
//
//   new <sampleRate>             construct, prepare at 44.1 kHz and then at
//                                <sampleRate>, as the port's
//                                WavetablePlugin::at does; every random
//                                source seeded as the port seeds it
//   tempo <bpm>                  what the host's play head reports, in 4/4
//   set <uid> <bits>             parameter <uid> to the user value whose
//                                IEEE bits are <bits> (hex)
//   defaults                     every parameter to its default and the
//                                modulation matrix emptied, as a program
//                                load begins
//   mod <src> <uid> <bits>       route modulation source <src> (in the
//                                order setupModMatrix adds them) to <uid>
//   presetxml <bytes>            followed by <bytes> of a preset file: its
//                                parameters over the defaults, and its
//                                MODMATRIX through ModMatrix::stateUpdated
//   loaded                       what stateUpdated leaves after a program
//                                load: the next block starts afresh
//   table <osc> <frames> <copies> <size> <notesPerTable>
//                                followed by frames * copies * (size + 1)
//                                raw floats: the band-limited frames
//                                oscillator <osc> plays, built by the port
//   midi <offset> <b0> <b1> <b2> a MIDI message in the next block
//   process <samples>            render a block; left then right go to
//                                stdout as raw little-endian floats
//   wtload <format> <size> <sampleRate> <bytes>
//                                followed by <bytes> of a wav or flac file:
//                                what loadWaveTable builds from it goes to
//                                stdout, as u32 frames, then per frame u32
//                                copies, u32 length and the floats
//
// Wavetable's sources are compiled into the harness unmodified, all but the
// editor (tests/it/gold.rs copies them next to a stub PluginEditor.h). The
// wavetables are handed in rather than read from disk because the original
// band-limits them with JUCE's FFT and the port with its own, which differ
// in the last bits; `wtload` compares those builds on their own.
#include <JuceHeader.h>
#include "PluginProcessor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <sstream>
#include <string>

// Every allocation is zeroed, as the SocaLabs harness does, so a member
// the original never initialises reads the same every run.
void* operator new (std::size_t size)
{
    if (void* p = std::calloc (1, size == 0 ? 1 : size))
        return p;
    throw std::bad_alloc();
}

void* operator new[] (std::size_t size)                 { return operator new (size); }
void operator delete (void* p) noexcept                 { std::free (p); }
void operator delete[] (void* p) noexcept               { std::free (p); }
void operator delete (void* p, std::size_t) noexcept    { std::free (p); }
void operator delete[] (void* p, std::size_t) noexcept  { std::free (p); }

namespace
{

// The white noise seeds itself from std::random_device; the port seeds it.
// Reaching the private generator goes through an explicit instantiation,
// where access checks do not apply.
template <class Tag, typename Tag::type Member>
struct Steal
{
    friend typename Tag::type get (Tag) { return Member; }
};

struct WhiteNoiseOf { using type = gin::WhiteNoise gin::BandLimitedLookupTables::*; friend type get (WhiteNoiseOf); };
struct GeneratorOf  { using type = std::default_random_engine gin::WhiteNoise::*; friend type get (GeneratorOf); };
struct DistOf       { using type = std::normal_distribution<float> gin::WhiteNoise::*; friend type get (DistOf); };

template struct Steal<WhiteNoiseOf, &gin::BandLimitedLookupTables::whiteNoise>;
template struct Steal<GeneratorOf, &gin::WhiteNoise::generator>;
template struct Steal<DistOf, &gin::WhiteNoise::dist>;

struct PlayHead : juce::AudioPlayHead
{
    double bpm = 120.0;

    juce::Optional<PositionInfo> getPosition() const override
    {
        PositionInfo info;
        info.setBpm (bpm);
        info.setTimeSignature (juce::AudioPlayHead::TimeSignature { 4, 4 });
        return info;
    }
};

[[noreturn]] void fail (const std::string& why)
{
    std::cerr << "harness: " << why << "\n";
    std::exit (1);
}

template <class T>
T parse (std::istringstream& words, const std::string& what)
{
    T value {};
    if (! (words >> value))
        fail ("bad " + what);
    return value;
}

float parseBits (std::istringstream& words, const std::string& what)
{
    std::string hex = parse<std::string> (words, what);
    uint32_t bits = uint32_t (std::stoul (hex, nullptr, 16));
    float value;
    std::memcpy (&value, &bits, sizeof (value));
    return value;
}

void readFloats (float* into, size_t n)
{
    if (! std::cin.read (reinterpret_cast<char*> (into), std::streamsize (n * sizeof (float))))
        fail ("short read of floats");
}

void writeU32 (uint32_t v)
{
    std::cout.write (reinterpret_cast<const char*> (&v), sizeof (v));
}

constexpr uint32_t seed = 1;

void seedEverything (WavetableAudioProcessor& p)
{
    p.rng.setSeed (seed);
    juce::Random::getSystemRandom().setSeed (int64_t (seed) ^ 0x5DEECE66);
    auto& noise = p.analogTables.*get (WhiteNoiseOf());
    (noise.*get (GeneratorOf())).seed (seed);
    (noise.*get (DistOf())).reset();
}

}

int main()
{
    std::ios::sync_with_stdio (false);

    std::unique_ptr<WavetableAudioProcessor> processor;
    PlayHead playHead;
    juce::MidiBuffer midi;
    std::string line;

    while (std::getline (std::cin, line))
    {
        std::istringstream words (line);
        std::string command;
        words >> command;

        if (command.empty())
            continue;

        if (command == "new")
        {
            auto sampleRate = parse<double> (words, "sample rate");
            // Airwindows seeds its dither from rand() as the effects are built.
            std::srand (seed);
            processor.reset();
            processor = std::make_unique<WavetableAudioProcessor>();
            processor->playHead = &playHead;
            seedEverything (*processor);
            for (double rate : { 44100.0, sampleRate })
            {
                processor->setRateAndBufferSizeDetails (rate, 512);
                processor->prepareToPlay (rate, 512);
            }
            midi.clear();
            continue;
        }

        if (command == "wtload")
        {
            auto format = parse<std::string> (words, "format");
            auto size = parse<int> (words, "frame size");
            auto sampleRate = parse<double> (words, "sample rate");
            auto bytes = parse<size_t> (words, "byte count");
            juce::MemoryBlock data (bytes);
            if (! std::cin.read (static_cast<char*> (data.getData()), std::streamsize (bytes)))
                fail ("short read of a wavetable file");
            std::cin.ignore (1);
            if (processor == nullptr)
                processor = std::make_unique<WavetableAudioProcessor>();
            gin::Wavetable table;
            if (! processor->loadWaveTable (table, sampleRate, data, format, size))
                fail ("loadWaveTable refused the file");
            writeU32 (uint32_t (table.getNumTables()));
            for (int f = 0; f < table.getNumTables(); f++)
            {
                auto& copies = table[f]->tables;
                writeU32 (uint32_t (copies.size()));
                writeU32 (uint32_t (copies.empty() ? 0 : copies[0].size()));
                for (auto& c : copies)
                    std::cout.write (reinterpret_cast<const char*> (c.data()),
                                     std::streamsize (c.size() * sizeof (float)));
            }
            continue;
        }

        if (processor == nullptr)
            fail ("'" + command + "' before 'new'");

        if (command == "tempo")
        {
            playHead.bpm = parse<double> (words, "tempo");
        }
        else if (command == "set")
        {
            auto uid = parse<std::string> (words, "parameter uid");
            auto value = parseBits (words, "parameter value");
            auto p = processor->getParameter (uid);
            if (p == nullptr)
                fail ("no parameter " + uid);
            p->setUserValue (value);
        }
        else if (command == "defaults")
        {
            for (auto p : processor->getPluginParameters())
                p->setUserValue (p->getUserDefaultValue());
            for (int d = 0; d < 1000; d++)
            {
                auto dst = gin::ModDstId (d);
                if (processor->modMatrix.getParameter (dst) == nullptr)
                    break;
                for (auto [src, depth] : processor->modMatrix.getModDepths (dst))
                    processor->modMatrix.clearModDepth (src, dst);
            }
        }
        else if (command == "mod")
        {
            auto src = parse<int> (words, "mod source");
            auto uid = parse<std::string> (words, "mod destination");
            auto depth = parseBits (words, "mod depth");
            auto p = processor->getParameter (uid);
            if (p == nullptr || p->getModIndex() < 0)
                fail ("no mod destination " + uid);
            processor->modMatrix.setModDepth (gin::ModSrcId (src), gin::ModDstId (p->getModIndex()), depth);
        }
        else if (command == "presetxml")
        {
            // A preset file, as gin::Program::loadFromFile reads it and
            // loadProcessor puts it in place: every parameter to its
            // default, then the file's, and its state — the matrix —
            // through the matrix's own stateUpdated.
            auto bytes = parse<size_t> (words, "byte count");
            std::string text (bytes, '\0');
            if (! std::cin.read (text.data(), std::streamsize (bytes)))
                fail ("short read of a preset");
            std::cin.ignore (1);
            auto root = juce::parseXML (juce::String (text));
            if (root == nullptr)
                fail ("the preset is not XML");
            for (auto p : processor->getPluginParameters())
                p->setUserValue (p->getUserDefaultValue());
            for (auto e = root->getChildByName ("param"); e != nullptr; e = e->getNextElementWithTagName ("param"))
                if (auto p = processor->getParameter (e->getStringAttribute ("uid")))
                    p->setUserValue (float (e->getDoubleAttribute ("val")));
            juce::ValueTree state ("state");
            if (auto s = root->getChildByName ("state"))
                state = juce::ValueTree::fromXml (*s);
            processor->modMatrix.stateUpdated (state);
        }
        else if (command == "loaded")
        {
            processor->presetLoaded = true;
            processor->lastMono = processor->globalParams.mono->isOn();
        }
        else if (command == "table")
        {
            auto osc = parse<int> (words, "oscillator");
            auto frames = parse<int> (words, "frame count");
            auto copies = parse<size_t> (words, "copy count");
            auto size = parse<int> (words, "table size");
            auto notesPerTable = parse<int> (words, "notes per table");
            gin::Wavetable table;
            for (int f = 0; f < frames; f++)
            {
                auto t = new gin::BandLimitedLookupTable();
                t->tableSize = size;
                t->notesPerTable = notesPerTable;
                t->tables.resize (copies);
                for (auto& c : t->tables)
                {
                    c.resize (size_t (size + 1));
                    readFloats (c.data(), c.size());
                }
                table.addTable (t);
            }
            std::cin.ignore (1);
            std::swap (table, osc == 0 ? processor->osc1Tables : processor->osc2Tables);
        }
        else if (command == "midi")
        {
            auto offset = parse<int> (words, "midi offset");
            auto b0 = parse<int> (words, "midi status");
            auto b1 = parse<int> (words, "midi data 1");
            auto b2 = parse<int> (words, "midi data 2");
            midi.addEvent (juce::MidiMessage (b0, b1, b2), offset);
        }
        else if (command == "process")
        {
            auto samples = parse<int> (words, "sample count");
            juce::AudioSampleBuffer buffer (2, samples);
            processor->processBlock (buffer, midi);
            midi.clear();
            for (int ch = 0; ch < 2; ch++)
                std::cout.write (reinterpret_cast<const char*> (buffer.getReadPointer (ch)),
                                 std::streamsize (samples) * std::streamsize (sizeof (float)));
        }
        else
        {
            fail ("unknown command '" + command + "'");
        }
    }

    std::cout.flush();
    return 0;
}
