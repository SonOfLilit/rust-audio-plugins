// The gold harness: NeuralAmpModelerCore, unmodified, driven by a script on
// standard input. Each line is a command:
//
//   load <path>        read a .nam file (and forget the last model)
//   fast <0|1>         NAM's fast tanh off or on, as the plug-in sets it
//                      (`Activation::enable_fast_tanh`), before `reset`
//   reset <rate> <max> `DSP::Reset`, which prewarms
//   process <n>        then 4·n bytes of little-endian float input;
//                      writes 4·n bytes of output
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "NAM/activations.h"
#include "NAM/dsp.h"
#include "NAM/get_dsp.h"

int main()
{
  std::unique_ptr<nam::DSP> model;
  std::string command;
  std::vector<float> in;
  std::vector<NAM_SAMPLE> x, y;
  while (std::cin >> command)
  {
    if (command == "load")
    {
      std::string path;
      std::cin >> path;
      model = nam::get_dsp(std::filesystem::path(path));
    }
    else if (command == "fast")
    {
      int on;
      std::cin >> on;
      if (on)
        nam::activations::Activation::enable_fast_tanh();
      else
        nam::activations::Activation::disable_fast_tanh();
    }
    else if (command == "reset")
    {
      double rate;
      int max;
      std::cin >> rate >> max;
      model->Reset(rate, max);
    }
    else if (command == "process")
    {
      int n;
      std::cin >> n;
      std::cin.get();
      in.resize(n);
      std::cin.read(reinterpret_cast<char*>(in.data()), 4 * n);
      x.assign(in.begin(), in.end());
      y.assign(n, 0);
      NAM_SAMPLE* xp = x.data();
      NAM_SAMPLE* yp = y.data();
      model->process(&xp, &yp, n);
      for (int i = 0; i < n; i++)
      {
        const float v = (float)y[i];
        std::fwrite(&v, 4, 1, stdout);
      }
    }
    else
    {
      std::fprintf(stderr, "unknown command %s\n", command.c_str());
      return 1;
    }
  }
  return 0;
}
