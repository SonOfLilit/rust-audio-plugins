//! The effects Wavetable runs over its whole output that Gin's effect
//! ports in `bloop-plugin-socalabs` do not already cover: the gate
//! (`gin_gateeffect.h`), the simple distortion (`gin::Distortion` in
//! `gin_audioutil.h`), the stereo delay as Wavetable drives it, smoothed
//! (`gin_stereodelay.h`), and the output gain (`gin::GainProcessor`).

use super::env::AnalogAdsr;
use bloop_plugin_socalabs::shared::{approximately_equal, jlimit, LinearSmoothedValue, WetDryMix};

/// The most steps a gate has.
pub const GATE_STEPS: usize = 16;

/// `gin::GateEffect`: a step pattern per side, each step opening or
/// closing an envelope, restarted by the first note after silence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gate {
    sample_rate: f64,
    adsr_l: AnalogAdsr,
    adsr_r: AnalogAdsr,
    steps_l: [bool; GATE_STEPS],
    steps_r: [bool; GATE_STEPS],
    len: usize,
    frequency: f32,
    phase: f32,
    mix: f32,
    triggered: bool,
    last_l: bool,
    last_r: bool,
}

impl Default for Gate {
    fn default() -> Self {
        let mut adsr = AnalogAdsr::default();
        adsr.set_decay(0.0);
        adsr.set_sustain_level(1.0);
        Self {
            sample_rate: 44100.0,
            adsr_l: adsr,
            adsr_r: adsr,
            steps_l: [false; GATE_STEPS],
            steps_r: [false; GATE_STEPS],
            len: 0,
            frequency: 0.0,
            phase: 0.0,
            mix: 1.0,
            triggered: false,
            last_l: false,
            last_r: false,
        }
    }
}

impl Gate {
    /// `setSampleRate`. The original hands the rate to the left envelope
    /// twice and never to the right one, which stays at 44.1 kHz; so here.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        self.adsr_l.set_sample_rate(sample_rate);
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.triggered = false;
        self.last_l = false;
        self.last_r = false;
        self.adsr_l.reset();
        self.adsr_r.reset();
    }

    /// `setLength`: steps added are closed.
    pub fn set_length(&mut self, n: usize) {
        let n = n.min(GATE_STEPS);
        for i in self.len..n {
            self.steps_l[i] = false;
            self.steps_r[i] = false;
        }
        self.len = n;
    }

    pub fn set_step(&mut self, step: usize, l: bool, r: bool) {
        if step < GATE_STEPS {
            self.steps_l[step] = l;
            self.steps_r[step] = r;
        }
    }

    pub fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency;
    }

    pub fn set_attack(&mut self, seconds: f32) {
        self.adsr_l.set_attack(seconds);
        self.adsr_r.set_attack(seconds);
    }

    pub fn set_release(&mut self, seconds: f32) {
        self.adsr_l.set_release(seconds);
        self.adsr_r.set_release(seconds);
    }

    /// `process (buffer, noteOnSample, noteOffSample)`: a note-on at
    /// `note_on` restarts the pattern; at `note_off` the gate closes.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32], note_on: i32, note_off: i32) {
        if self.len == 0 {
            return;
        }
        let steps = self.len as f32;
        let step = if self.frequency > 0.0001 {
            (f64::from(self.frequency) / self.sample_rate) as f32
        } else {
            0.0
        };
        for (i, (l, r)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
            let i = i as i32;
            if i == note_on {
                self.phase = 0.0;
                self.triggered = true;
            }
            if i == note_off {
                self.triggered = false;
            }
            self.phase += step;
            while self.phase >= steps {
                self.phase -= steps;
            }
            let at = self.phase as usize;
            let l_on = self.triggered && self.steps_l[at];
            let r_on = self.triggered && self.steps_r[at];
            if l_on != self.last_l {
                self.last_l = l_on;
                if l_on {
                    self.adsr_l.note_on();
                } else {
                    self.adsr_l.note_off();
                }
            }
            if r_on != self.last_r {
                self.last_r = r_on;
                if r_on {
                    self.adsr_r.note_on();
                } else {
                    self.adsr_r.note_off();
                }
            }
            *l = *l * (1.0 - self.mix) + *l * self.adsr_l.process() * self.mix;
            *r = *r * (1.0 - self.mix) + *r * self.adsr_r.process() * self.mix;
        }
    }
}

/// `gin::Distortion::processBlock (buffer, drive, -clip, clip)` with
/// Wavetable's `clip = 1 / (2 · drive)`: a soft clip and a gain.
pub fn distort(left: &mut [f32], right: &mut [f32], drive: f32) {
    if drive <= 0.0 {
        return;
    }
    let clip = 1.0 / (2.0 * drive);
    let gain = if drive < 0.5 {
        2.0 * drive + 1.0
    } else {
        drive * 4.0
    };
    for channel in [left, right] {
        for x in channel.iter_mut() {
            let input = jlimit(-clip, clip, *x);
            *x = (input - drive * input * input.abs()) * gain;
        }
    }
}

/// How long the original's delay line is: `gin::StereoDelay { 120.1 }`.
const ORIGINAL_ROOM: f64 = 120.1;

/// `gin::DelayLine` read the way the original reads its 120-second line,
/// without holding 120 seconds.
///
/// Where a read lands is worked out in float from the line's length, as
/// `readLagrange` does — and at five or six million samples a float has
/// lost the fraction's low bits, so a shorter line would interpolate
/// differently. So the arithmetic runs on the original's length, and the
/// samples live in a ring that holds only as far back as the delay
/// reaches, found by how far back from the newest they are.
#[derive(Clone, Debug)]
struct LongDelayLine {
    channels: [Vec<f32>; 2],
    /// The original's length at this rate.
    logical: i32,
    sample_rate: f64,
    write_pos: i32,
    /// Samples written since the last clear, which is where the ring is.
    written: usize,
}

impl LongDelayLine {
    fn new(room: f64, sample_rate: f64) -> Self {
        let size = ((room * sample_rate).ceil() as usize + 4).next_power_of_two();
        let mut line = Self {
            channels: [vec![0.0; size], vec![0.0; size]],
            logical: 1,
            sample_rate: 44100.0,
            write_pos: 0,
            written: 0,
        };
        line.set_rate(sample_rate);
        line
    }

    /// `setSize (2, 120.1, rate)`, keeping the ring: allocates nothing.
    fn set_rate(&mut self, sample_rate: f64) {
        self.sample_rate = if sample_rate > 0.0 {
            sample_rate
        } else {
            44100.0
        };
        self.logical = ((ORIGINAL_ROOM * self.sample_rate).ceil() as i32).max(1);
        self.clear();
    }

    fn clear(&mut self) {
        self.write_pos = 0;
        self.written = 0;
        for c in &mut self.channels {
            c.fill(0.0);
        }
    }

    /// The sample at logical index `index`, from the ring.
    fn at(&self, channel: usize, index: i32) -> f32 {
        let ring = &self.channels[channel];
        // How far behind the newest write this index is.
        let newest = (self.write_pos - 1).rem_euclid(self.logical);
        let back = (newest - index).rem_euclid(self.logical) as usize;
        if back >= ring.len() || back >= self.written {
            return 0.0;
        }
        let at = (self.written - 1 - back) & (ring.len() - 1);
        ring[at]
    }

    /// `readLagrange (ch, t)`, `t` held to what the ring reaches.
    fn read_lagrange(&self, channel: usize, t: f64) -> f32 {
        let n = self.logical;
        let reach = (self.channels[0].len() - 4) as f64 / self.sample_rate;
        let t = t.clamp(0.0, (f64::from(n - 1) / self.sample_rate).min(reach));
        let read_pos =
            ((f64::from(self.write_pos - 1 + n) - t * self.sample_rate) as f32) % (n as f32);
        let i1 = read_pos.floor() as i32;
        let fraction = read_pos - i1 as f32;
        let ys = [
            self.at(channel, i1),
            self.at(channel, (i1 + 1) % n),
            self.at(channel, (i1 + 2) % n),
            self.at(channel, (i1 + 3) % n),
        ];
        lagrange(&ys, fraction)
    }

    fn write(&mut self, channel: usize, input: f32) {
        let ring = &mut self.channels[channel];
        let at = self.written & (ring.len() - 1);
        ring[at] = input;
    }

    fn write_finished(&mut self) {
        self.written += 1;
        self.write_pos += 1;
        if self.write_pos >= self.logical {
            self.write_pos = 0;
        }
    }
}

/// `gin::Lagrange::interpolate` over the points 0, 1, 2, 3.
fn lagrange(ys: &[f32; 4], x: f32) -> f32 {
    let xs = [0.0f32, 1.0, 2.0, 3.0];
    let mut res = 0.0f32;
    for i in 0..4 {
        let mut term = ys[i];
        for j in 0..4 {
            if i != j {
                let d = xs[i] - xs[j];
                if approximately_equal(d, 0.0) {
                    term = 0.0;
                } else {
                    term = term * (x - xs[j]) / d;
                }
            }
        }
        res += term;
    }
    res
}

/// `gin::StereoDelay` as Wavetable runs it: its settings ramp over 0.2 s
/// when they move (`processSmoothed`), and are taken as they stand when
/// they do not (`process`).
#[derive(Clone, Debug)]
pub struct StereoDelay {
    line: LongDelayLine,
    mix: LinearSmoothedValue,
    delay: LinearSmoothedValue,
    feedback: LinearSmoothedValue,
    crossfeed: LinearSmoothedValue,
}

impl StereoDelay {
    /// A delay that reaches `room` seconds back at `sample_rate` (less at
    /// a higher rate). Allocates, once: nothing after does.
    #[must_use]
    pub fn new(room: f64, sample_rate: f64) -> Self {
        let mut d = Self {
            line: LongDelayLine::new(room, sample_rate),
            mix: LinearSmoothedValue::default(),
            delay: LinearSmoothedValue::default(),
            feedback: LinearSmoothedValue::default(),
            crossfeed: LinearSmoothedValue::default(),
        };
        d.set_sample_rate(sample_rate);
        d
    }

    /// `setSampleRate`: the line emptied and the ramps restarted.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.line.set_rate(sample_rate);
        for v in [
            &mut self.mix,
            &mut self.delay,
            &mut self.feedback,
            &mut self.crossfeed,
        ] {
            v.reset(sample_rate, 0.2);
        }
    }

    pub fn set_params(&mut self, delay: f32, mix: f32, feedback: f32, crossfeed: f32) {
        self.mix.set_target(mix);
        self.delay.set_target(delay);
        self.feedback.set_target(feedback.min(0.99));
        self.crossfeed.set_target(crossfeed.min(0.99));
    }

    pub fn reset(&mut self) {
        self.line.clear();
        self.snap();
    }

    /// Every setting at its target, the line as it is.
    pub fn snap(&mut self) {
        for v in [
            &mut self.mix,
            &mut self.delay,
            &mut self.feedback,
            &mut self.crossfeed,
        ] {
            v.set_current_and_target(v.target());
        }
    }

    fn is_smoothing(&self) -> bool {
        self.mix.is_smoothing()
            || self.delay.is_smoothing()
            || self.feedback.is_smoothing()
            || self.crossfeed.is_smoothing()
    }

    /// `process`: the settings as they are aimed.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let delay = f64::from(self.delay.target());
        let (feedback, crossfeed) = (self.feedback.target(), self.crossfeed.target());
        let wet_dry = WetDryMix::new(self.mix.target());
        for (out_l, out_r) in left.iter_mut().zip(right.iter_mut()) {
            let (in_l, in_r) = (*out_l, *out_r);
            let l = self.line.read_lagrange(0, delay);
            let r = self.line.read_lagrange(1, delay);
            self.line.write(0, in_l + (feedback * l) + (crossfeed * r));
            self.line.write(1, in_r + (feedback * r) + (crossfeed * l));
            self.line.write_finished();
            *out_l = (wet_dry.wet_gain * l) + (wet_dry.dry_gain * in_l);
            *out_r = (wet_dry.wet_gain * r) + (wet_dry.dry_gain * in_r);
        }
    }

    /// `processSmoothed`: each setting a ramp step on per sample, while any
    /// is ramping.
    pub fn process_smoothed(&mut self, left: &mut [f32], right: &mut [f32]) {
        if !self.is_smoothing() {
            self.process(left, right);
            return;
        }
        for (out_l, out_r) in left.iter_mut().zip(right.iter_mut()) {
            let (in_l, in_r) = (*out_l, *out_r);
            let delay = f64::from(self.delay.next_value());
            let l = self.line.read_lagrange(0, delay);
            let r = self.line.read_lagrange(1, delay);
            let feedback = self.feedback.next_value();
            let crossfeed = self.crossfeed.next_value();
            self.line.write(0, in_l + (feedback * l) + (crossfeed * r));
            self.line.write(1, in_r + (feedback * r) + (crossfeed * l));
            self.line.write_finished();
            let wet_dry = WetDryMix::new(self.mix.next_value());
            *out_l = (wet_dry.wet_gain * l) + (wet_dry.dry_gain * in_l);
            *out_r = (wet_dry.wet_gain * r) + (wet_dry.dry_gain * in_r);
        }
    }
}

/// `gin::GainProcessor`: a ramp from the last block's gain to this one's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gain {
    gain: f32,
    last: f32,
}

impl Default for Gain {
    fn default() -> Self {
        Self {
            gain: 1.0,
            last: 1.0,
        }
    }
}

impl Gain {
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain;
    }

    /// At the gain set, with no ramp from the last.
    pub fn snap(&mut self) {
        self.last = self.gain;
    }

    /// `juce::AudioBuffer::applyGainRamp (0, n, last, gain)` on both
    /// channels: a steady gain when the two are approximately equal (none
    /// at 1, silence at 0), a ramp otherwise.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let n = left.len();
        for channel in [&mut *left, &mut *right] {
            if approximately_equal(self.last, self.gain) {
                let gain = self.last;
                if approximately_equal(gain, 1.0) {
                } else if approximately_equal(gain, 0.0) {
                    channel.fill(0.0);
                } else {
                    for x in channel.iter_mut() {
                        *x *= gain;
                    }
                }
            } else {
                let increment = (self.gain - self.last) / n as f32;
                let mut g = self.last;
                for x in channel.iter_mut() {
                    *x *= g;
                    g += increment;
                }
            }
        }
        self.last = self.gain;
    }
}
