#pragma once
#include "Effect.h"

/** See Effect.h: an inert stand-in for the reference's Distortion. */
class Distortion final : public InertEffect
{
public:
    Distortion() = default;
    explicit Distortion (int) {}
};
