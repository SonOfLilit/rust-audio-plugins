// Runs one Airwindows plug-in (PLUG, from PLUG_H) through a program read
// from stdin, writing every processed block to stdout as raw floats.
//
//   new <rate>          srand(1), then construct, as bloop seeds its own
//   set <i> <hexbits>   setParameter, the value pinned to 0..1 first
//   rate <rate>         what getSampleRate() returns from now on
//   clear               a fresh instance (after srand(1)), knobs carried over
//   process <n>\n       2n little-endian floats, left then right, then \n;
//                       processReplacing in place, both channels written back
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include PLUG_H

double g_sample_rate = 44100.0;

static float pin(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

int main() {
  PLUG* p = nullptr;
  char cmd[32];
  std::vector<float> buf;
  while (scanf("%31s", cmd) == 1) {
    if (!strcmp(cmd, "new")) {
      if (scanf("%lf", &g_sample_rate) != 1) return 2;
      srand(1);
      p = new PLUG(nullptr);
    } else if (!strcmp(cmd, "set")) {
      int i; unsigned bits; float v;
      if (scanf("%d %x", &i, &bits) != 2) return 2;
      memcpy(&v, &bits, 4);
      p->setParameter(i, pin(v));
    } else if (!strcmp(cmd, "rate")) {
      if (scanf("%lf", &g_sample_rate) != 1) return 2;
    } else if (!strcmp(cmd, "clear")) {
      float params[kNumParameters];
      for (int i = 0; i < kNumParameters; i++) params[i] = p->getParameter(i);
      delete p;
      srand(1);
      p = new PLUG(nullptr);
      for (int i = 0; i < kNumParameters; i++) p->setParameter(i, params[i]);
    } else if (!strcmp(cmd, "process")) {
      int n;
      if (scanf("%d", &n) != 1 || getchar() != '\n') return 2;
      buf.resize(2 * (size_t)n);
      if (fread(buf.data(), sizeof(float), 2 * (size_t)n, stdin) != 2 * (size_t)n) return 2;
      float* io[2] = {buf.data(), buf.data() + n};
      p->processReplacing(io, io, n);
      fwrite(buf.data(), sizeof(float), 2 * (size_t)n, stdout);
    } else {
      fprintf(stderr, "unknown command %s\n", cmd);
      return 2;
    }
  }
  delete p;
  return 0;
}
