// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The gold harness: Bungee Basic, built unmodified from vendor/bungee,
// driven grain by grain from a program on stdin.
//
// stdin, little-endian:
//   u32 input rate, u32 output rate, u32 channels, u32 frames
//   f32 × channels × frames      the input, channel after channel
//   u32 grains
//   per grain: f64 position, f64 speed, f64 pitch, u8 reset
// stdout, per grain:
//   u32 frames, f64 begin position, f64 end position
//   f32 × channels × frames      the output chunk, channel after channel
//
// Input outside 0..frames is zeros, as the port's caller supplies it.

#include <bungee/Bungee.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

template <class T>
static T read()
{
	T value{};
	if (fread(&value, sizeof value, 1, stdin) != 1)
	{
		fprintf(stderr, "harness: short read\n");
		std::exit(2);
	}
	return value;
}

template <class T>
static void write(T value)
{
	fwrite(&value, sizeof value, 1, stdout);
}

int main()
{
	const auto inputRate = read<uint32_t>();
	const auto outputRate = read<uint32_t>();
	const auto channels = read<uint32_t>();
	const auto frames = read<uint32_t>();
	std::vector<float> input(size_t(channels) * frames);
	for (auto &s : input)
		s = read<float>();

	Bungee::Stretcher<Bungee::Basic> stretcher({int(inputRate), int(outputRate)}, int(channels));
	std::vector<float> chunk;

	const auto grains = read<uint32_t>();
	for (uint32_t g = 0; g < grains; ++g)
	{
		Bungee::Request request{};
		request.position = read<double>();
		request.speed = read<double>();
		request.pitch = read<double>();
		request.reset = read<uint8_t>() != 0;

		const auto needed = stretcher.specifyGrain(request);
		const int length = needed.end - needed.begin;
		chunk.assign(size_t(channels) * (length > 0 ? length : 0), 0.f);
		for (uint32_t c = 0; c < channels; ++c)
			for (int i = 0; i < length; ++i)
			{
				const long p = long(needed.begin) + i;
				if (p >= 0 && p < long(frames))
					chunk[size_t(c) * length + i] = input[size_t(c) * frames + p];
			}
		stretcher.analyseGrain(chunk.data(), length > 0 ? length : 0);

		Bungee::OutputChunk output{};
		stretcher.synthesiseGrain(output);
		write<uint32_t>(output.frameCount);
		write<double>(output.request[Bungee::OutputChunk::begin]->position);
		write<double>(output.request[Bungee::OutputChunk::end]->position);
		for (uint32_t c = 0; c < channels; ++c)
			fwrite(output.data + c * output.channelStride, sizeof(float), output.frameCount, stdout);
	}
	return 0;
}
