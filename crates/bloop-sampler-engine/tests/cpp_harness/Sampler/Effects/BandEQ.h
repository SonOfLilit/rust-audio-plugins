#pragma once
#include "Effect.h"

/** See Effect.h: an inert stand-in for the reference's BandEQ. */
class BandEQ final : public InertEffect
{
public:
    BandEQ() = default;
    explicit BandEQ (int) {}
};
