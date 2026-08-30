/*
    Just-a-Sample's `Source/Utilities/BufferUtils.h`, reduced to the one
    function the voice calls. `mixToBuffer` is copied verbatim from the pinned
    reference — it is part of the DSP under test.
*/

#pragma once
#include <JuceHeader.h>

inline void mixToBuffer(const juce::AudioBuffer<float>& src, juce::AudioBuffer<float>& dest, int startSample, int numSamples, bool mixMono)
{
    if (mixMono)
    {
        // Mix all source channels to mono
        for (int ch = 0; ch < src.getNumChannels(); ch++)
        {
            for (int outputCh = 0; outputCh < dest.getNumChannels(); outputCh++)
            {
                dest.addFrom(outputCh, startSample, src.getReadPointer(ch), numSamples, 1.f / float(src.getNumChannels()));
            }
        }
    }
    else if (dest.getNumChannels() < src.getNumChannels())
    {
        // Down mix from more source channels to fewer destination channels
        float ratio = float(src.getNumChannels()) / dest.getNumChannels();
        for (int outputCh = 0; outputCh < dest.getNumChannels(); outputCh++)
        {
            int startCh = int(outputCh * ratio);
            int endCh = int((outputCh + 1) * ratio);

            for (int ch = startCh; ch < endCh; ch++)
            {
                dest.addFrom(outputCh, startSample, src.getReadPointer(ch), numSamples, 1.f / (endCh - startCh));
            }
        }
    }
    else if (dest.getNumChannels() >= src.getNumChannels())
    {
        // Up mix from fewer source channels to more destination channels
        float ratio = float(dest.getNumChannels()) / src.getNumChannels();
        for (int inputCh = 0; inputCh < src.getNumChannels(); inputCh++)
        {
            int startCh = int(inputCh * ratio);
            int endCh = int((inputCh + 1) * ratio);

            for (int outputCh = startCh; outputCh < endCh; outputCh++)
            {
                dest.addFrom(outputCh, startSample, src.getReadPointer(inputCh), numSamples);
            }
        }
    }
}
