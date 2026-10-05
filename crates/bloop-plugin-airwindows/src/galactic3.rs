//! Galactic3: a big reverb. A vibrato predelay, a lowpass, then twelve
//! delay lines in three banks of four, each bank mixing its four into the
//! next by "each minus the rest", the last bank fed back across the
//! channels into the first. It runs undersampled — one reverb sample every
//! `1 / derez` input samples, rebuilt in between on a Bezier curve.

use crate::math::{dither, guard, pow, sin, SEEDS};

/// The delay lines' sizes, as the original declares them. The longest
/// delay `Bigness` can ask of each is under its size.
const SIZE_I: usize = 6480;
const SIZE_J: usize = 3660;
const SIZE_K: usize = 1720;
const SIZE_L: usize = 680;
const SIZE_A: usize = 9700;
const SIZE_B: usize = 6000;
const SIZE_C: usize = 2320;
const SIZE_D: usize = 940;
const SIZE_E: usize = 15220;
const SIZE_F: usize = 8460;
const SIZE_G: usize = 4540;
const SIZE_H: usize = 3200;
const SIZE_M: usize = 3111;

// `bez`'s layout.
const BEZ_AL: usize = 0;
const BEZ_AR: usize = 1;
const BEZ_BL: usize = 2;
const BEZ_BR: usize = 3;
const BEZ_CL: usize = 4;
const BEZ_CR: usize = 5;
const BEZ_IN_L: usize = 6;
const BEZ_IN_R: usize = 7;
const BEZ_UN_IN_L: usize = 8;
const BEZ_UN_IN_R: usize = 9;
const BEZ_SAMP_L: usize = 10;
const BEZ_SAMP_R: usize = 11;
const BEZ_CYCLE: usize = 12;
const BEZ_TOTAL: usize = 13;

/// One delay line, both channels, sized as the original declares it.
#[derive(Clone, Debug)]
struct Line {
    l: Box<[f64]>,
    r: Box<[f64]>,
}

impl Line {
    fn new(size: usize) -> Self {
        Self {
            l: vec![0.0; size].into_boxed_slice(),
            r: vec![0.0; size].into_boxed_slice(),
        }
    }

    fn clear(&mut self) {
        self.l.fill(0.0);
        self.r.fill(0.0);
    }
}

/// A position into a line, shared by both channels, and the line's length.
#[derive(Clone, Copy, Debug)]
struct Counter {
    count: i32,
    delay: i32,
}

impl Counter {
    const NEW: Self = Self { count: 1, delay: 0 };

    /// `count++; if (count < 0 || count > delay) count = 0;`
    fn advance(&mut self) {
        self.count += 1;
        if self.count < 0 || self.count > self.delay {
            self.count = 0;
        }
    }

    /// Where to write.
    fn at(self) -> usize {
        self.count as usize
    }

    /// `count - ((count > delay) ? delay + 1 : 0)`: where to read.
    fn read(self) -> usize {
        (self.count
            - if self.count > self.delay {
                self.delay + 1
            } else {
                0
            }) as usize
    }
}

/// Four lines in a bank: what each writes is "itself minus the other
/// three" of the four values it is handed.
fn mix(a: f64, b: f64, c: f64, d: f64) -> [f64; 4] {
    [
        a - (b + c + d),
        b - (a + c + d),
        c - (a + b + d),
        d - (a + b + c),
    ]
}

#[derive(Clone, Debug)]
pub struct Galactic3 {
    pub params: [f32; 6],
    pub sample_rate: f32,
    iir_al: f64,
    iir_bl: f64,
    iir_ar: f64,
    iir_br: f64,
    i: Line,
    j: Line,
    k: Line,
    l: Line,
    a: Line,
    b: Line,
    c: Line,
    d: Line,
    e: Line,
    f: Line,
    g: Line,
    h: Line,
    m: Line,
    oldfpd: f64,
    feedback_l: [f64; 4],
    feedback_r: [f64; 4],
    /// I, J, K, L, A, B, C, D, E, F, G, H, M.
    counters: [Counter; 13],
    vib_m: f64,
    bez: [f64; BEZ_TOTAL],
    fpd_l: u32,
    fpd_r: u32,
}

const CI: usize = 0;
const CJ: usize = 1;
const CK: usize = 2;
const CL: usize = 3;
const CA: usize = 4;
const CB: usize = 5;
const CC: usize = 6;
const CD: usize = 7;
const CE: usize = 8;
const CF: usize = 9;
const CG: usize = 10;
const CH: usize = 11;
const CM: usize = 12;

/// The base lengths `Bigness` scales, I to H in `counters`' order.
const LENGTHS: [f64; 12] = [
    3407.0, 1823.0, 859.0, 331.0, 4801.0, 2909.0, 1153.0, 461.0, 7607.0, 4217.0, 2269.0, 1597.0,
];

/// The knobs as the block uses them.
struct Block {
    regen: f64,
    attenuate: f64,
    lowpass: f64,
    drift: f64,
    derez: f64,
    wet: f64,
}

impl Galactic3 {
    /// A Galactic3 as its constructor leaves it. Its delay lines are a
    /// megabyte, on the heap.
    pub fn boxed(sample_rate: f32) -> Box<Self> {
        let mut this = Box::new(Self {
            params: [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
            sample_rate,
            iir_al: 0.0,
            iir_bl: 0.0,
            iir_ar: 0.0,
            iir_br: 0.0,
            i: Line::new(SIZE_I),
            j: Line::new(SIZE_J),
            k: Line::new(SIZE_K),
            l: Line::new(SIZE_L),
            a: Line::new(SIZE_A),
            b: Line::new(SIZE_B),
            c: Line::new(SIZE_C),
            d: Line::new(SIZE_D),
            e: Line::new(SIZE_E),
            f: Line::new(SIZE_F),
            g: Line::new(SIZE_G),
            h: Line::new(SIZE_H),
            m: Line::new(SIZE_M),
            oldfpd: 0.0,
            feedback_l: [0.0; 4],
            feedback_r: [0.0; 4],
            counters: [Counter::NEW; 13],
            vib_m: 0.0,
            bez: [0.0; BEZ_TOTAL],
            fpd_l: 0,
            fpd_r: 0,
        });
        this.clear();
        this
    }

    /// The constructor's state, in place. The original leaves each line's
    /// last element unset; no `Bigness` reaches it, so zero is as good.
    pub fn clear(&mut self) {
        self.iir_al = 0.0;
        self.iir_ar = 0.0;
        self.iir_bl = 0.0;
        self.iir_br = 0.0;
        self.i.clear();
        self.j.clear();
        self.k.clear();
        self.l.clear();
        self.a.clear();
        self.b.clear();
        self.c.clear();
        self.d.clear();
        self.e.clear();
        self.f.clear();
        self.g.clear();
        self.h.clear();
        self.m.clear();
        self.feedback_l = [0.0; 4];
        self.feedback_r = [0.0; 4];
        for c in &mut self.counters {
            c.count = 1;
        }
        self.vib_m = 3.0;
        self.oldfpd = 429496.7295;
        self.bez = [0.0; BEZ_TOTAL];
        self.bez[BEZ_CYCLE] = 1.0;
        (self.fpd_l, self.fpd_r) = SEEDS;
    }

    fn block(&mut self) -> Block {
        let [a, b, c, d, e, f] = self.params.map(f64::from);
        let mut overallscale = 1.0;
        overallscale /= 44100.0;
        overallscale *= f64::from(self.sample_rate);

        let regen = 0.0625 + ((1.0 - a) * 0.0625);
        let derez = (d / overallscale).clamp(0.0005, 1.0);
        let derez = 1.0 / f64::from((1.0 / derez) as i32);
        let size = (e * 1.77) + 0.1;
        for (counter, length) in self.counters.iter_mut().zip(LENGTHS) {
            counter.delay = (length * size) as i32;
        }
        self.counters[CM].delay = 256;
        Block {
            regen,
            attenuate: (1.0 - (regen / 0.125)) * 1.333,
            lowpass: pow(1.00001 - (1.0 - b), 2.0) / overallscale.sqrt(),
            drift: pow(c, 3.0) * 0.001,
            derez,
            wet: 1.0 - pow(1.0 - f, 3.0),
        }
    }

    /// How long the tail lasts at these knobs: the time for the slowest
    /// path round the loop to fall 100 dB, and one more pass for the
    /// predelay, the filters and the Bezier rebuild to drain. Each pass round it — three
    /// banks of "each minus the other three", a gain of 2 apiece in the
    /// directions that survive, times `regen` — costs `8 * regen`, so
    /// Replace 1 halves the tail every pass and Replace 0 keeps it for
    /// ever (it also stops new input coming in: a freeze).
    pub fn tail_seconds(&self) -> f32 {
        let a = f64::from(self.params[0]);
        let loop_gain = 8.0 * (0.0625 + ((1.0 - a) * 0.0625));
        if loop_gain >= 1.0 {
            return f32::INFINITY;
        }
        let overallscale = f64::from(self.sample_rate) / 44100.0;
        let derez = (f64::from(self.params[3]) / overallscale).clamp(0.0005, 1.0);
        let size = (f64::from(self.params[4]) * 1.77) + 0.1;
        let longest = (LENGTHS[CI] + LENGTHS[CA] + LENGTHS[CE]) * size;
        let passes = 1.0 - 100.0 / (20.0 * libm::log10(loop_gain));
        let samples = passes * longest / derez + 512.0;
        (samples / f64::from(self.sample_rate)) as f32
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let k = self.block();
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let in_l = guard(*l, self.fpd_l);
            let in_r = guard(*r, self.fpd_r);
            let (mut sl, mut sr) = self.predelay(in_l, in_r, &k);

            self.iir_al = (self.iir_al * (1.0 - k.lowpass)) + (sl * k.lowpass);
            sl = self.iir_al;
            self.iir_ar = (self.iir_ar * (1.0 - k.lowpass)) + (sr * k.lowpass);
            sr = self.iir_ar;

            let bez = &mut self.bez;
            bez[BEZ_CYCLE] += k.derez;
            bez[BEZ_SAMP_L] += (sl + bez[BEZ_IN_L]) * k.derez;
            bez[BEZ_SAMP_R] += (sr + bez[BEZ_IN_R]) * k.derez;
            bez[BEZ_IN_L] = sl;
            bez[BEZ_IN_R] = sr;
            if bez[BEZ_CYCLE] > 1.0 {
                bez[BEZ_CYCLE] = 0.0;
                self.reverb(k.regen);
            }
            let bez = &self.bez;
            let cycle = bez[BEZ_CYCLE];
            let cbl = (bez[BEZ_CL] * (1.0 - cycle)) + (bez[BEZ_BL] * cycle);
            let cbr = (bez[BEZ_CR] * (1.0 - cycle)) + (bez[BEZ_BR] * cycle);
            let bal = (bez[BEZ_BL] * (1.0 - cycle)) + (bez[BEZ_AL] * cycle);
            let bar = (bez[BEZ_BR] * (1.0 - cycle)) + (bez[BEZ_AR] * cycle);
            sl = (bez[BEZ_BL] + (cbl * (1.0 - cycle)) + (bal * cycle)) * 0.125;
            sr = (bez[BEZ_BR] + (cbr * (1.0 - cycle)) + (bar * cycle)) * 0.125;

            self.iir_bl = (self.iir_bl * (1.0 - k.lowpass)) + (sl * k.lowpass);
            sl = self.iir_bl;
            self.iir_br = (self.iir_br * (1.0 - k.lowpass)) + (sr * k.lowpass);
            sr = self.iir_br;

            if k.wet < 1.0 {
                sl = (sl * k.wet) + (in_l * (1.0 - k.wet));
                sr = (sr * k.wet) + (in_r * (1.0 - k.wet));
            }
            *l = dither(sl, &mut self.fpd_l) as f32;
            *r = dither(sr, &mut self.fpd_r) as f32;
        }
    }

    /// The predelay: 256 samples, read at an offset a slow sine wobbles,
    /// whose speed takes a new random step every cycle.
    fn predelay(&mut self, in_l: f64, in_r: f64, k: &Block) -> (f64, f64) {
        self.vib_m += self.oldfpd * k.drift;
        if self.vib_m > (3.141592653589793238 * 2.0) {
            self.vib_m = 0.0;
            self.oldfpd = 0.4294967295 + (f64::from(self.fpd_l) * 0.0000000000618);
        }
        let m = self.counters[CM];
        self.m.l[m.at()] = in_l * k.attenuate;
        self.m.r[m.at()] = in_r * k.attenuate;
        self.counters[CM].advance();
        let count = self.counters[CM].count;
        let delay = self.counters[CM].delay;

        let read = |line: &[f64], offset: f64| {
            let working = (f64::from(count) + offset) as i32;
            let at = |w: i32| (w - if w > delay { delay + 1 } else { 0 }) as usize;
            let frac = offset - offset.floor();
            let mut y = line[at(working)] * (1.0 - frac);
            y += line[at(working + 1)] * frac;
            y
        };
        let offset_l = (sin(self.vib_m) + 1.0) * 127.0;
        let offset_r = (sin(self.vib_m + (3.141592653589793238 / 2.0)) + 1.0) * 127.0;
        (read(&self.m.l, offset_l), read(&self.m.r, offset_r))
    }

    /// One reverb sample, from what the Bezier accumulator gathered.
    fn reverb(&mut self, regen: f64) {
        let bez = &mut self.bez;
        let c = &mut self.counters;
        let in_l = bez[BEZ_SAMP_L] + bez[BEZ_UN_IN_L];
        let in_r = bez[BEZ_SAMP_R] + bez[BEZ_UN_IN_R];
        let (fl, fr) = (self.feedback_l, self.feedback_r);

        self.i.l[c[CI].at()] = in_l + (fr[0] * regen);
        self.j.l[c[CJ].at()] = in_l + (fr[1] * regen);
        self.k.l[c[CK].at()] = in_l + (fr[2] * regen);
        self.l.l[c[CL].at()] = in_l + (fr[3] * regen);
        bez[BEZ_UN_IN_L] = bez[BEZ_SAMP_L];
        self.i.r[c[CI].at()] = in_r + (fl[0] * regen);
        self.j.r[c[CJ].at()] = in_r + (fl[1] * regen);
        self.k.r[c[CK].at()] = in_r + (fl[2] * regen);
        self.l.r[c[CL].at()] = in_r + (fl[3] * regen);
        bez[BEZ_UN_IN_R] = bez[BEZ_SAMP_R];
        for n in [CI, CJ, CK, CL] {
            c[n].advance();
        }
        let first_l = [
            self.i.l[c[CI].read()],
            self.j.l[c[CJ].read()],
            self.k.l[c[CK].read()],
            self.l.l[c[CL].read()],
        ];
        let first_r = [
            self.i.r[c[CI].read()],
            self.j.r[c[CJ].read()],
            self.k.r[c[CK].read()],
            self.l.r[c[CL].read()],
        ];

        let [wa, wb, wc, wd] = mix(first_l[0], first_l[1], first_l[2], first_l[3]);
        self.a.l[c[CA].at()] = wa;
        self.b.l[c[CB].at()] = wb;
        self.c.l[c[CC].at()] = wc;
        self.d.l[c[CD].at()] = wd;
        let [wa, wb, wc, wd] = mix(first_r[0], first_r[1], first_r[2], first_r[3]);
        self.a.r[c[CA].at()] = wa;
        self.b.r[c[CB].at()] = wb;
        self.c.r[c[CC].at()] = wc;
        self.d.r[c[CD].at()] = wd;
        for n in [CA, CB, CC, CD] {
            c[n].advance();
        }
        let second_l = [
            self.a.l[c[CA].read()],
            self.b.l[c[CB].read()],
            self.c.l[c[CC].read()],
            self.d.l[c[CD].read()],
        ];
        let second_r = [
            self.a.r[c[CA].read()],
            self.b.r[c[CB].read()],
            self.c.r[c[CC].read()],
            self.d.r[c[CD].read()],
        ];

        let [we, wf, wg, wh] = mix(second_l[0], second_l[1], second_l[2], second_l[3]);
        self.e.l[c[CE].at()] = we;
        self.f.l[c[CF].at()] = wf;
        self.g.l[c[CG].at()] = wg;
        self.h.l[c[CH].at()] = wh;
        let [we, wf, wg, wh] = mix(second_r[0], second_r[1], second_r[2], second_r[3]);
        self.e.r[c[CE].at()] = we;
        self.f.r[c[CF].at()] = wf;
        self.g.r[c[CG].at()] = wg;
        self.h.r[c[CH].at()] = wh;
        for n in [CE, CF, CG, CH] {
            c[n].advance();
        }
        let [el, fl, gl, hl] = [
            self.e.l[c[CE].read()],
            self.f.l[c[CF].read()],
            self.g.l[c[CG].read()],
            self.h.l[c[CH].read()],
        ];
        let [er, fr, gr, hr] = [
            self.e.r[c[CE].read()],
            self.f.r[c[CF].read()],
            self.g.r[c[CG].read()],
            self.h.r[c[CH].read()],
        ];

        self.feedback_l = mix(el, fl, gl, hl);
        self.feedback_r = mix(er, fr, gr, hr);

        let out_l = (el + fl + gl + hl) / 8.0;
        let out_r = (er + fr + gr + hr) / 8.0;
        bez[BEZ_CL] = bez[BEZ_BL];
        bez[BEZ_BL] = bez[BEZ_AL];
        bez[BEZ_AL] = out_l;
        bez[BEZ_SAMP_L] = 0.0;
        bez[BEZ_CR] = bez[BEZ_BR];
        bez[BEZ_BR] = bez[BEZ_AR];
        bez[BEZ_AR] = out_r;
        bez[BEZ_SAMP_R] = 0.0;
    }
}
