// The gold harness: runs one of the original processors over a program
// read from stdin, and writes what it produced to stdout.
//
//   new <plugin> <sampleRate>    construct, prepareToPlay, reset
//   tempo <bpm>                  what the host's play head reports, in 4/4
//   set <index> <value>          parameter <index>, in the order the
//                                constructor adds them, to a user value
//   process <samples>            followed by that many left samples, then
//                                that many right, as raw little-endian
//                                floats; the processed block goes to stdout
//                                the same way
//
// tests/gold.rs writes the same program to the Rust port and compares the
// two outputs bit for bit.
#include "delay.h"
#include "dynamics.h"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

namespace
{

std::unique_ptr<harness::Processor> create (const std::string& name)
{
    if (name == "Compressor")   return std::make_unique<harness::CompressorAudioProcessor>();
    if (name == "Limiter")      return std::make_unique<harness::LimiterAudioProcessor>();
    if (name == "Gate")         return std::make_unique<harness::GateAudioProcessor>();
    if (name == "Expander")     return std::make_unique<harness::ExpanderAudioProcessor>();
    if (name == "Delay")        return std::make_unique<harness::DelayAudioProcessor>();
    return nullptr;
}

[[noreturn]] void fail (const std::string& why)
{
    std::cerr << "harness: " << why << "\n";
    std::exit (1);
}

}

int main()
{
    std::ios::sync_with_stdio (false);

    std::unique_ptr<harness::Processor> processor;
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
            std::string name;
            double sampleRate = 0;
            words >> name >> sampleRate;
            processor = create (name);
            if (processor == nullptr)
                fail ("no plugin called " + name);
            processor->prepareToPlay (sampleRate, 512);
            processor->reset();
            continue;
        }

        if (processor == nullptr)
            fail ("'" + command + "' before 'new'");

        if (command == "set")
        {
            size_t index = 0;
            float value = 0;
            words >> index >> value;
            if (index >= processor->extParameters.size())
                fail ("no parameter " + std::to_string (index));
            processor->extParameters[index]->setUserValue (value);
        }
        else if (command == "tempo")
        {
            words >> processor->playHead.bpm;
        }
        else if (command == "process")
        {
            int samples = 0;
            words >> samples;
            juce::AudioSampleBuffer buffer (2, samples);
            for (int ch = 0; ch < 2; ch++)
                if (! std::cin.read (reinterpret_cast<char*> (buffer.getWritePointer (ch)),
                                     std::streamsize (samples) * std::streamsize (sizeof (float))))
                    fail ("short read of process input");
            // Consume the newline that ends the binary block.
            std::cin.ignore (1);
            processor->processBlock (buffer);
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
