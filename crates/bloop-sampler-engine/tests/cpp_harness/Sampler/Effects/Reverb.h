#pragma once
#include "Effect.h"

/** See Effect.h: an inert stand-in for the reference's Reverb. */
class Reverb final : public InertEffect
{
public:
    Reverb() = default;
    explicit Reverb (int) {}
};
