#pragma once
#include "Effect.h"

/** See Effect.h: an inert stand-in for the reference's Chorus. */
class Chorus final : public InertEffect
{
public:
    Chorus() = default;
    explicit Chorus (int) {}
};
