//! Compiles every Airwindows LinuxVST plug-in in `vendor/airwindows` (or
//! `BLOOP_AIRWINDOWS_SRC`, a checkout of airwindows) and writes the list of
//! them, `catalog.rs`, into OUT_DIR.
//!
//! Each plug-in is one translation unit of its own, written here: the
//! stand-in base class (cpp/audioeffectx.h), then the plug-in's two .cpp
//! files inside a namespace named after it, so that no two plug-ins'
//! classes, helpers or `createEffectInstance` meet at link time, and
//! three `extern "C"` symbols the registry lists it by. Inside that
//! namespace `std` is `bloop_no_std` (audioeffectx.h): the one thing of
//! the standard library the plug-ins use is a `std::set<std::string>` of
//! host capabilities, which nothing reads and whose allocations would
//! keep a plug-in from being constructed again in place.
//!
//! On wasm32 nothing is compiled and the list is empty: there is no C++
//! toolchain for it here.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// Plug-ins left out, and why. A plug-in left out is not compiled.
const SKIPPED: [(&str, &str); 7] = [
    (
        "Baxandall2",
        "ported to Rust (bloop-plugin-airwindows); the Browser lists the port",
    ),
    (
        "Console7Buss",
        "ported to Rust (bloop-plugin-airwindows); the Browser lists the port",
    ),
    (
        "Console7Channel",
        "ported to Rust (bloop-plugin-airwindows); the Browser lists the port",
    ),
    (
        "Density2",
        "ported to Rust (bloop-plugin-airwindows); the Browser lists the port",
    ),
    (
        "Galactic3",
        "ported to Rust (bloop-plugin-airwindows); the Browser lists the port",
    ),
    (
        "ToTape8",
        "ported to Rust (bloop-plugin-airwindows); the Browser lists the port",
    ),
    (
        "PunchyGuitar",
        "its output grows without bound at its defaults: inf within a second of noise",
    ),
];

fn main() -> Result<(), Box<dyn Error>> {
    let out = PathBuf::from(std::env::var("OUT_DIR")?);
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH")?;
    let target_os = std::env::var("CARGO_CFG_TARGET_OS")?;
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rerun-if-env-changed=BLOOP_AIRWINDOWS_SRC");

    if target_arch == "wasm32" {
        fs::write(
            out.join("catalog.rs"),
            catalog(&[], &BTreeMap::new(), &BTreeMap::new()),
        )?;
        return Ok(());
    }

    let root = std::env::var_os("BLOOP_AIRWINDOWS_SRC")
        .map_or_else(|| manifest.join("../../vendor/airwindows"), PathBuf::from);
    let src = root.join("plugins/LinuxVST/src");
    if !src.is_dir() {
        return Err(format!(
            "the Airwindows sources are not at {}: check out the submodule with \
             `git -c submodule.vendor/airwindows.update=checkout submodule update --init --depth 1 vendor/airwindows`, \
             or point BLOOP_AIRWINDOWS_SRC at a checkout of github.com/airwindows/airwindows",
            src.display()
        )
        .into());
    }
    println!("cargo:rerun-if-changed={}", src.display());
    let encyclopedia = root.join("Airwindopedia.txt");
    let what = root.join("what.txt");
    println!("cargo:rerun-if-changed={}", encyclopedia.display());
    println!("cargo:rerun-if-changed={}", what.display());

    let plugins = plugin_names(&src)?;
    let categories = categories(&fs::read_to_string(&encyclopedia)?);
    let descriptions = descriptions(&fs::read_to_string(&what)?);

    let units = out.join("units");
    fs::create_dir_all(&units)?;
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++14")
        .opt_level(2)
        .debug(false)
        // Third-party code, compiled as published: its warnings are
        // Airwindows', and would be printed on every build as cargo's.
        .flag("-w")
        .flag("-fno-rtti")
        .include(manifest.join("cpp"));
    if target_os == "android" {
        // As oboe-sys links it: one copy of libc++ inside the app's .so.
        build.cpp_link_stdlib("c++_static");
    }
    for name in &plugins {
        let unit = units.join(format!("{name}.cpp"));
        write_if_changed(&unit, &wrapper(name, &src.join(name)))?;
        build.file(&unit);
    }
    let registry = out.join("registry.cpp");
    write_if_changed(&registry, &registry_source(&plugins))?;
    build.file(&registry);
    build.file(manifest.join("cpp/api.cpp"));
    build.compile("bloop_airwindows");

    if target_os == "linux" {
        count_allocations(&manifest, &out);
    }

    fs::write(
        out.join("catalog.rs"),
        catalog(&plugins, &categories, &descriptions),
    )?;
    Ok(())
}

/// The plug-ins' folders under `src`, by name, without the skipped ones.
fn plugin_names(src: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let mut names = Vec::new();
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let has = |file: String| entry.path().join(file).is_file();
        if has(format!("{name}.cpp"))
            && has(format!("{name}Proc.cpp"))
            && !SKIPPED.iter().any(|(skip, _)| *skip == name)
        {
            names.push(name);
        }
    }
    names.sort_by_key(|name| name.to_lowercase());
    Ok(names)
}

/// One plug-in's translation unit.
fn wrapper(name: &str, dir: &Path) -> String {
    let cpp = dir.join(format!("{name}.cpp"));
    let proc = dir.join(format!("{name}Proc.cpp"));
    format!(
        "#include \"audioeffectx.h\"\n\
         namespace bloop_aw_{name} {{\n\
         namespace std = ::bloop_no_std;\n\
         #include \"{cpp}\"\n\
         #include \"{proc}\"\n\
         }}\n\
         extern \"C\" AudioEffectX* bloop_aw_construct_{name}(void* at, BloopHost* host) {{\n\
         \x20 return new (at) bloop_aw_{name}::{name}(host);\n\
         }}\n\
         extern \"C\" const unsigned long bloop_aw_size_{name} = sizeof(bloop_aw_{name}::{name});\n\
         extern \"C\" const int bloop_aw_params_{name} = bloop_aw_{name}::kNumParameters;\n",
        cpp = cpp.display(),
        proc = proc.display(),
    )
}

/// The table api.cpp finds each plug-in by: name, size, knob count and
/// constructor, in the catalog's order.
fn registry_source(plugins: &[String]) -> String {
    let mut s = String::from(
        "#include \"audioeffectx.h\"\n\
         struct BloopEntry { const char* name; unsigned long size; int params; \
         AudioEffectX* (*construct)(void*, BloopHost*); };\n",
    );
    for name in plugins {
        let _ = writeln!(
            s,
            "extern \"C\" AudioEffectX* bloop_aw_construct_{name}(void*, BloopHost*);\n\
             extern \"C\" const unsigned long bloop_aw_size_{name};\n\
             extern \"C\" const int bloop_aw_params_{name};"
        );
    }
    s.push_str("extern \"C\" const BloopEntry bloop_aw_entries[] = {\n");
    for name in plugins {
        let _ = writeln!(
            s,
            "  {{\"{name}\", bloop_aw_size_{name}, bloop_aw_params_{name}, bloop_aw_construct_{name}}},"
        );
    }
    s.push_str(
        "};\n\
         extern \"C\" const int bloop_aw_entry_count = sizeof(bloop_aw_entries) / sizeof(bloop_aw_entries[0]);\n",
    );
    s
}

/// The first category Airwindopedia's "# Categories" lists each plug-in
/// under.
fn categories(text: &str) -> BTreeMap<String, String> {
    let mut found = BTreeMap::new();
    let section = text
        .split("# Categories")
        .nth(1)
        .unwrap_or_default()
        .split("############")
        .next()
        .unwrap_or_default();
    for line in section.lines() {
        let Some((category, list)) = line.split_once(':') else {
            continue;
        };
        for name in list.split([',', '.']) {
            let name = name.trim();
            if !name.is_empty() {
                found
                    .entry(name.to_string())
                    .or_insert_with(|| category.trim().to_string());
            }
        }
    }
    found
}

/// what.txt's one line about each plug-in, without its collection tags.
fn descriptions(text: &str) -> BTreeMap<String, String> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            let name = line.split_whitespace().next()?;
            let about = line.split("[coll=").next().unwrap_or(line).trim();
            Some((name.to_string(), about.to_string()))
        })
        .collect()
}

fn catalog(
    plugins: &[String],
    categories: &BTreeMap<String, String>,
    descriptions: &BTreeMap<String, String>,
) -> String {
    let mut s = format!(
        "/// How many plug-ins were compiled.\npub const COUNT: usize = {};\n\
         /// Each compiled plug-in, in the order the C registry lists them.\n\
         pub static PLUGINS: [Listing; COUNT] = [\n",
        plugins.len()
    );
    for name in plugins {
        let category = categories.get(name).map_or("", String::as_str);
        let about = descriptions.get(name).map_or("", String::as_str);
        let _ = writeln!(
            s,
            "    Listing {{ name: {name:?}, category: {category:?}, about: {about:?} }},"
        );
    }
    s.push_str("];\n/// The plug-ins left out, and why.\n");
    let skipped: Vec<&(&str, &str)> = if plugins.is_empty() {
        Vec::new()
    } else {
        SKIPPED.iter().collect()
    };
    let _ = writeln!(
        s,
        "pub static SKIPPED: [(&str, &str); {}] = [",
        skipped.len()
    );
    for (name, why) in skipped {
        let _ = writeln!(s, "    ({name:?}, {why:?}),");
    }
    s.push_str("];\n");
    s
}

/// For this crate's tests on Linux: every malloc, calloc, realloc and
/// operator new (plain or nothrow) the test binary makes, counted per thread
/// (cpp/count_allocations.c), so a test can hold a plug-in's processing to
/// none — the C++'s as well as Rust's, which the Rust allocator alone
/// would not see.
fn count_allocations(manifest: &Path, out: &Path) {
    cc::Build::new()
        .file(manifest.join("cpp/count_allocations.c"))
        .opt_level(2)
        .debug(false)
        .cargo_metadata(false)
        .compile("bloop_aw_count_allocations");
    println!(
        "cargo:rustc-link-arg-tests=-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=_Znwm,--wrap=_Znam,--wrap=_ZnwmRKSt9nothrow_t,--wrap=_ZnamRKSt9nothrow_t"
    );
    println!(
        "cargo:rustc-link-arg-tests={}",
        out.join("libbloop_aw_count_allocations.a").display()
    );
}

/// Leave a file that already says `contents` untouched, so cc does not
/// see it changed.
fn write_if_changed(path: &Path, contents: &str) -> std::io::Result<()> {
    if fs::read_to_string(path).is_ok_and(|have| have == contents) {
        return Ok(());
    }
    fs::write(path, contents)
}
