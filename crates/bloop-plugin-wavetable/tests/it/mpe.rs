//! bloop's notes as an MPE zone's MIDI ([`mpe::Channels`],
//! bloop#pad-expression): which member channel each note gets, where its
//! expression goes, and what is left on the master channel. What the
//! synth then does with that MIDI is `gold::` (`mpe_*`), against the
//! original.

#![expect(
    clippy::needless_raw_string_hashes,
    reason = "UPDATE_EXPECT writes r#\"…\"# whether or not the snapshot needs the hashes"
)]
#![expect(
    clippy::needless_pass_by_value,
    reason = "the harness takes its snapshot as expect-test hands it, by value"
)]

use std::fmt::Write as _;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

use bloop_core::Note;
use bloop_events::{Expression, NoteEvent, NoteId};
use bloop_plugin_wavetable::{mpe, Event};
use expect_test::{expect, Expect};

/// `on 60 #3`, `off 60 #3` (or no `#` for an id-less note), `bend #3 1.5`,
/// `timbre #3 0.8`, `press #3 0.4`, `wheel 0.75`, `cc 1 0.5`, `choke 60`,
/// and `mpe on` / `mpe off` for the switch from there on.
fn event(line: &str) -> NoteEvent {
    let words: Vec<&str> = line.split_whitespace().collect();
    let id = |w: Option<&&str>| {
        w.and_then(|w| w.strip_prefix('#'))
            .map_or(NoteId::NONE, |n| NoteId(n.parse().unwrap_or(0)))
    };
    let number = |w: &str| w.parse::<f32>().unwrap_or(f32::NAN);
    let note = |w: &str| Note::new(w.parse().unwrap_or(0));
    match words[0] {
        "on" => NoteEvent::NoteOn {
            time: 0,
            channel: 0,
            note: note(words[1]),
            velocity: 1.0,
            id: id(words.get(2)),
        },
        "off" => NoteEvent::NoteOff {
            time: 0,
            channel: 0,
            note: note(words[1]),
            id: id(words.get(2)),
        },
        "bend" | "timbre" | "press" => NoteEvent::NoteExpression {
            time: 0,
            id: id(words.get(1)),
            expression: match words[0] {
                "bend" => Expression::Bend(number(words[2])),
                "timbre" => Expression::Timbre(number(words[2])),
                _ => Expression::Pressure(number(words[2])),
            },
        },
        "wheel" => NoteEvent::PitchBend {
            time: 0,
            channel: 0,
            value: number(words[1]),
        },
        "cc" => NoteEvent::CC {
            time: 0,
            channel: 0,
            cc: words[1].parse().unwrap_or(0),
            value: number(words[2]),
        },
        "choke" => NoteEvent::Choke {
            time: 0,
            channel: 0,
            note: note(words[1]),
        },
        other => NoteEvent::Aftertouch {
            time: 0,
            channel: 0,
            note: note(other),
            pressure: 0.0,
        },
    }
}

fn said(event: Event) -> String {
    match event {
        Event::NoteOn {
            channel,
            note,
            velocity,
        } => format!("ch{channel} on {note} v{velocity}"),
        Event::NoteOff { channel, note } => format!("ch{channel} off {note}"),
        Event::PitchWheel { channel, value } => format!("ch{channel} wheel {value}"),
        Event::Controller { channel, cc, value } => format!("ch{channel} cc{cc} {value}"),
        Event::ChannelPressure { channel, value } => format!("ch{channel} pressure {value}"),
        Event::Choke { channel, note } => format!("ch{channel} choke {note}"),
    }
}

/// Each line, and what the zone sends for it.
fn check(lines: &[&str], expect: Expect) {
    let mut channels = mpe::Channels::default();
    let mut on = false;
    let mut out = String::new();
    for line in lines {
        match *line {
            "mpe on" => on = true,
            "mpe off" => on = false,
            _ => {
                let mut sent = Vec::new();
                channels.translate(on, &event(line), |e| sent.push(said(e)));
                let _ = writeln!(out, "{line:<16} {}", sent.join(", "));
                continue;
            }
        }
        let _ = writeln!(out, "{line}");
    }
    expect.assert_eq(&out);
}

/// MPE off: one channel for everything, as before, and expression
/// dropped.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn off_is_one_channel_and_no_expression() {
    check(
        &[
            "on 60 #1",
            "on 64 #2",
            "bend #1 2",
            "timbre #2 0.9",
            "press #1 0.5",
            "wheel 0.75",
            "cc 1 0.5",
            "off 60 #1",
            "choke 64",
        ],
        expect![[r#"
            on 60 #1         ch1 on 60 v127
            on 64 #2         ch1 on 64 v127
            bend #1 2        
            timbre #2 0.9    
            press #1 0.5     
            wheel 0.75       ch1 wheel 12287
            cc 1 0.5         ch1 cc1 64
            off 60 #1        ch1 off 60
            choke 64         ch1 choke 64
        "#]],
    );
}

/// MPE on: each note its own member channel, its expression there, the
/// wheel and the controllers on the master; an expression for a note let
/// go of, or with no id, says nothing.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn on_gives_each_note_a_channel_and_its_expression() {
    check(
        &[
            "mpe on",
            "on 60 #1",
            "on 64 #2",
            "bend #1 12",
            "bend #2 -48",
            "bend #2 48",
            "bend #2 0",
            "timbre #1 0",
            "timbre #1 0.5",
            "timbre #1 1",
            "press #2 0.25",
            "wheel 0.5",
            "cc 74 1",
            "off 64 #2",
            "bend #2 1",
            "bend none 1",
            "Aftertouch",
        ],
        expect![[r#"
            mpe on
            on 60 #1         ch2 on 60 v127
            on 64 #2         ch3 on 64 v127
            bend #1 12       ch2 wheel 10240
            bend #2 -48      ch3 wheel 0
            bend #2 48       ch3 wheel 16383
            bend #2 0        ch3 wheel 8192
            timbre #1 0      ch2 cc74 0
            timbre #1 0.5    ch2 cc74 64
            timbre #1 1      ch2 cc74 127
            press #2 0.25    ch3 pressure 32
            wheel 0.5        ch1 wheel 8192
            cc 74 1          ch1 cc74 127
            off 64 #2        ch3 off 64
            bend #2 1        
            bend none 1      
            Aftertouch       
        "#]],
    );
}

/// A note goes on the channel let go of longest ago: a release still
/// ringing keeps its channel while another will do.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_new_note_takes_the_channel_freed_longest_ago() {
    check(
        &[
            "mpe on",
            "on 60 #1",
            "on 62 #2",
            "on 64 #3",
            "off 62 #2",
            "off 60 #1",
            "on 65 #4",
            "on 67 #5",
            "on 69 #6",
        ],
        expect![[r#"
            mpe on
            on 60 #1         ch2 on 60 v127
            on 62 #2         ch3 on 62 v127
            on 64 #3         ch4 on 64 v127
            off 62 #2        ch3 off 62
            off 60 #1        ch2 off 60
            on 65 #4         ch5 on 65 v127
            on 67 #5         ch6 on 67 v127
            on 69 #6         ch7 on 69 v127
        "#]],
    );
}

/// Sixteen notes held: the sixteenth goes on the master channel and
/// takes no expression; let go of, it comes off the master.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn the_sixteenth_note_held_goes_on_the_master() {
    let mut lines = vec!["mpe on".to_string()];
    for n in 1..=16 {
        lines.push(format!("on {} #{n}", 40 + n));
    }
    lines.push("bend #16 3".to_string());
    lines.push("off 56 #16".to_string());
    lines.push("bend #15 3".to_string());
    let lines: Vec<&str> = lines.iter().map(String::as_str).collect();
    check(
        &lines,
        expect![[r#"
        mpe on
        on 41 #1         ch2 on 41 v127
        on 42 #2         ch3 on 42 v127
        on 43 #3         ch4 on 43 v127
        on 44 #4         ch5 on 44 v127
        on 45 #5         ch6 on 45 v127
        on 46 #6         ch7 on 46 v127
        on 47 #7         ch8 on 47 v127
        on 48 #8         ch9 on 48 v127
        on 49 #9         ch10 on 49 v127
        on 50 #10        ch11 on 50 v127
        on 51 #11        ch12 on 51 v127
        on 52 #12        ch13 on 52 v127
        on 53 #13        ch14 on 53 v127
        on 54 #14        ch15 on 54 v127
        on 55 #15        ch16 on 55 v127
        on 56 #16        ch1 on 56 v127
        bend #16 3       
        off 56 #16       ch1 off 56
        bend #15 3       ch16 wheel 8704
    "#]],
    );
}

/// Id-less notes (a Pattern's) take channels too, and come off by pitch;
/// two strikes of one pitch on two ids come off by id; a choke cuts the
/// pitch on every channel holding it, or on the master if none does.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn notes_without_ids_and_chokes() {
    check(
        &[
            "mpe on",
            "on 60",
            "on 60 #7",
            "on 62",
            "off 60",
            "off 60 #7",
            "on 64 #8",
            "on 64 #9",
            "choke 64",
            "choke 70",
            "off 62",
        ],
        expect![[r#"
            mpe on
            on 60            ch2 on 60 v127
            on 60 #7         ch3 on 60 v127
            on 62            ch4 on 62 v127
            off 60           ch2 off 60
            off 60 #7        ch3 off 60
            on 64 #8         ch5 on 64 v127
            on 64 #9         ch6 on 64 v127
            choke 64         ch5 choke 64, ch6 choke 64
            choke 70         ch1 choke 70
            off 62           ch4 off 62
        "#]],
    );
}

/// Every member taken, one by a Pattern's note (no id) of the pitch a
/// live strike then plays on the master: the live strike's release comes
/// off the master, and the Pattern's note keeps sounding until its own.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_release_by_id_never_takes_a_pattern_notes_channel() {
    let mut lines = vec!["mpe on".to_string(), "on 60".to_string()];
    for n in 1..=14 {
        lines.push(format!("on {} #{n}", 70 + n));
    }
    lines.push("on 60 #20".to_string());
    lines.push("off 60 #20".to_string());
    lines.push("off 60".to_string());
    let lines: Vec<&str> = lines.iter().map(String::as_str).collect();
    check(&lines, expect![[r#"
        mpe on
        on 60            ch2 on 60 v127
        on 71 #1         ch3 on 71 v127
        on 72 #2         ch4 on 72 v127
        on 73 #3         ch5 on 73 v127
        on 74 #4         ch6 on 74 v127
        on 75 #5         ch7 on 75 v127
        on 76 #6         ch8 on 76 v127
        on 77 #7         ch9 on 77 v127
        on 78 #8         ch10 on 78 v127
        on 79 #9         ch11 on 79 v127
        on 80 #10        ch12 on 80 v127
        on 81 #11        ch13 on 81 v127
        on 82 #12        ch14 on 82 v127
        on 83 #13        ch15 on 83 v127
        on 84 #14        ch16 on 84 v127
        on 60 #20        ch1 on 60 v127
        off 60 #20       ch1 off 60
        off 60           ch2 off 60
    "#]]);
}

/// The softest strike goes out at velocity 1, not 0 (a note-on at 0 is a
/// note-off); a release naming an id no member holds comes off the
/// master, and the member holding its pitch under another id keeps it.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn a_whisper_still_strikes_and_a_stranger_comes_off_the_master() {
    let mut channels = mpe::Channels::default();
    let mut out = Vec::new();
    let soft = NoteEvent::NoteOn {
        time: 0,
        channel: 0,
        note: Note::new(60),
        velocity: 0.003,
        id: NoteId(1),
    };
    for (mpe, event) in [
        (false, soft),
        (true, soft),
        (true, event("on 62 #2")),
        (true, event("off 62 #9")),
        (true, event("off 62 #2")),
    ] {
        channels.translate(mpe, &event, |e| out.push(said(e)));
    }
    expect![[r#"
        ch1 on 60 v1
        ch2 on 60 v1
        ch3 on 62 v127
        ch1 off 62
        ch3 off 62"#]]
    .assert_eq(&out.join("\n"));
}

/// Switching the zone on or off forgets every channel: the synth lets go
/// of every note as it switches.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn switching_forgets_the_channels() {
    check(
        &[
            "mpe on",
            "on 60 #1",
            "on 62 #2",
            "mpe off",
            "off 60 #1",
            "mpe on",
            "on 64 #3",
            "bend #1 1",
        ],
        expect![[r#"
            mpe on
            on 60 #1         ch2 on 60 v127
            on 62 #2         ch3 on 62 v127
            mpe off
            off 60 #1        ch1 off 60
            mpe on
            on 64 #3         ch2 on 64 v127
            bend #1 1        
        "#]],
    );
}

/// A bend in semitones as a member channel's wheel: 48 either way is the
/// whole range, rest is 8192, and past the range or no number holds at
/// the end or at rest.
#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn bends_as_fourteen_bits() {
    let bends = [
        -96.0,
        -48.0,
        -24.0,
        -1.0,
        0.0,
        1.0,
        24.0,
        48.0,
        96.0,
        f32::NAN,
    ];
    let out: Vec<String> = bends
        .iter()
        .map(|st| {
            format!(
                "{st} st -> {}",
                mpe::fourteen_bit_bend(st / mpe::NOTE_BEND_RANGE)
            )
        })
        .collect();
    expect![[r#"
        -96 st -> 0
        -48 st -> 0
        -24 st -> 4096
        -1 st -> 8021
        0 st -> 8192
        1 st -> 8363
        24 st -> 12288
        48 st -> 16383
        96 st -> 16383
        NaN st -> 8192"#]]
    .assert_eq(&out.join("\n"));
}
