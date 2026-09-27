// The gold harness: runs one of the original processors over a program
// read from stdin, and writes what it produced to stdout.
//
//   new <plugin> <sampleRate>    construct, prepareToPlay, reset
//   reset <sampleRate>           prepareToPlay and reset again, as a host
//                                does when it (re)starts a processor
//   tempo <bpm>                  what the host's play head reports, in 4/4
//   set <index> <bits>           parameter <index>, in the order the
//                                constructor adds them, to the user value
//                                whose IEEE bits are <bits> (hex)
//   process <samples>            followed by that many left samples, then
//                                that many right, as raw little-endian
//                                floats; the processed block goes to stdout
//                                the same way
//
// Each plugin's own PluginProcessor.cpp is compiled into the harness
// unmodified (tests/gold.rs builds one unit per plugin, see
// slplugins_unit), so what runs here is the original, not a copy of it.
// tests/gold.rs writes the same program to the Rust port and compares the
// two outputs bit for bit.
#include <JuceHeader.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <new>
#include <sstream>
#include <string>

// Every allocation is zeroed. SimpleVerb's constructor reads members it
// never initialised; zero is what a fresh allocation holds, and what the
// port starts them at, so a run means the same thing every time.
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

#define SLPLUGIN(name) juce::AudioProcessor* create_##name();
#include "slplugins.inc"
#undef SLPLUGIN

namespace
{

using Factory = juce::AudioProcessor* (*)();

const std::map<std::string, Factory>& factories()
{
    static const std::map<std::string, Factory> all =
    {
       #define SLPLUGIN(name) { #name, &create_##name },
       #include "slplugins.inc"
       #undef SLPLUGIN
    };
    return all;
}

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

}

int main()
{
    std::ios::sync_with_stdio (false);

    std::unique_ptr<juce::AudioProcessor> owner;
    gin::Processor* processor = nullptr;
    PlayHead playHead;
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
            auto name = parse<std::string> (words, "plugin name");
            auto sampleRate = parse<double> (words, "sample rate");
            auto it = factories().find (name);
            if (it == factories().end())
                fail ("no plugin called " + name);
            owner.reset (it->second());
            processor = dynamic_cast<gin::Processor*> (owner.get());
            if (processor == nullptr)
                fail (name + " is not a gin::Processor");
            processor->playHead = &playHead;
            processor->prepareToPlay (sampleRate, 512);
            processor->reset();
            continue;
        }

        if (processor == nullptr)
            fail ("'" + command + "' before 'new'");

        if (command == "reset")
        {
            auto sampleRate = parse<double> (words, "sample rate");
            processor->prepareToPlay (sampleRate, 512);
            processor->reset();
        }
        else if (command == "tempo")
        {
            playHead.bpm = parse<double> (words, "tempo");
        }
        else if (command == "set")
        {
            auto index = parse<size_t> (words, "parameter index");
            std::string hex = parse<std::string> (words, "parameter value");
            uint32_t bits = uint32_t (std::stoul (hex, nullptr, 16));
            float value;
            std::memcpy (&value, &bits, sizeof (value));
            if (index >= processor->extParameters.size())
                fail ("no parameter " + std::to_string (index));
            processor->extParameters[index]->setUserValue (value);
        }
        else if (command == "process")
        {
            auto samples = parse<int> (words, "sample count");
            juce::AudioSampleBuffer buffer (2, samples);
            for (int ch = 0; ch < 2; ch++)
                if (! std::cin.read (reinterpret_cast<char*> (buffer.getWritePointer (ch)),
                                     std::streamsize (samples) * std::streamsize (sizeof (float))))
                    fail ("short read of process input");
            // The newline that ends the binary block.
            std::cin.ignore (1);
            juce::MidiBuffer midi;
            processor->processBlock (buffer, midi);
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
