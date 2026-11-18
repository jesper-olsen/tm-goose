// Checkpoints for tm.goose: save a machine and the random generator that trains it, and load them
// back, so that a long run can be stopped, resumed or evaluated. Resuming is exact: the generator is a
// single word of state, so training 3 epochs gives the same machine as training 1, saving, loading and
// training 2 more.
//
//   tm_io::save("ckpt.tm", m, r, epochs_done);                 // false if the file cannot be written
//   let err = tm_io::load("ckpt.tm", m, r, epochs_done);       // then tm_io::is_ok(err)
//
// `load` checks the whole file before it touches anything: if it does not return Ok, `m`, `r` and
// `epochs_done` are exactly as they were. `save` writes atomically, so an interrupted save leaves the
// previous checkpoint in place.
//
// The file is little-endian. The automaton states are the machine; the include masks follow from them.
//
//   offset  size  contents
//        0     7  "TMGOOSE"
//        7     1  format version, 1
//        8     4  n_features
//       12     4  n_classes
//       16     4  n_clauses (per class)
//       20     4  n_states (N)
//       24     4  threshold (T)
//       28     4  boost, 0 or 1
//       32     8  s, as an IEEE double
//       40     8  epochs done (a counter for the caller; the machine does not use it)
//       48     8  state of the random generator
//       56     n  one byte per automaton, 1..2N: n_classes * n_clauses * 2 * n_features of them
namespace tm_io;

import std;
import os;
import binary;
import tm;

const HEADER_SIZE = 56;
const VERSION = 1;
const MAX_DIM = 1048576;            // largest n_features, n_classes or n_clauses accepted (2^20)

enum LoadError {
    Ok,
    // The file could not be read.
    Io,
    // Not a checkpoint: it does not start with the magic bytes.
    BadMagic,
    // A checkpoint, but of a format version this code does not know.
    UnsupportedVersion { found: u8 },
    // The file is not as long as its header says (sizes in bytes).
    BadSize { expected: i64, found: i64 },
    // A header field is out of range.
    BadHeader,
    // An automaton state is outside 1..2N.
    BadState,
}

fn is_ok(e: LoadError) -> bool {
    match e { Ok => true, _ => false }
}

fn format(out: u8[>..]&, e: LoadError) {
    match e {
        Ok => { format(out, "ok"); },
        Io => { format(out, "I/O error: cannot read file"); },
        BadMagic => { format(out, "not a tm-goose checkpoint"); },
        UnsupportedVersion v => { format(out, "unsupported checkpoint version ", v.found, " (this code reads version ", VERSION, ")"); },
        BadSize b => { format(out, "checkpoint has the wrong size: expected ", b.expected, " bytes, found ", b.found); },
        BadHeader => { format(out, "checkpoint header has a value out of range"); },
        BadState => { format(out, "checkpoint holds an automaton state out of range"); },
    }
}

// Writes the machine, the generator and the epoch counter to `path`. False if that failed.
fn save(path: const u8[:], m: tm::TM&, r: rng&, epochs_done: i64) -> bool {
    var buf: u8[>..] = [];
    buf.append("TMGOOSE");
    binary::write_u8(buf, VERSION);
    binary::write_u32le(buf, m.n_features as u32);
    binary::write_u32le(buf, m.n_classes as u32);
    binary::write_u32le(buf, m.n_clauses as u32);
    binary::write_u32le(buf, m.n_states as u32);
    binary::write_u32le(buf, m.threshold as u32);
    binary::write_u32le(buf, if m.boost { 1 } else { 0 });
    binary::write_f64le(buf, m.s);
    binary::write_u64le(buf, epochs_done as u64);
    binary::write_u64le(buf, r.s);
    assert(buf.len == HEADER_SIZE);
    buf.append(m.state);
    return write_file_atomic(path, buf);
}

// Reads a checkpoint into `m`, `r` and `epochs_done`. Anything but Ok leaves all three untouched.
fn load(path: const u8[:], m: tm::TM&, r: rng&, epochs_done: i64&) -> LoadError {
    var data: u8[>..] = [];
    guard read_file(path, data) else { return LoadError.Io; }

    let magic = "TMGOOSE";
    guard data.len >= magic.len && data[0..magic.len] == magic else { return LoadError.BadMagic; }
    guard data.len >= magic.len + 1 else { return LoadError.BadSize { HEADER_SIZE, data.len }; }
    guard data[7] == VERSION else { return LoadError.UnsupportedVersion { data[7] }; }
    guard data.len >= HEADER_SIZE else { return LoadError.BadSize { HEADER_SIZE, data.len }; }

    let n_features = binary::u32le(data, 8) as i64;
    let n_classes = binary::u32le(data, 12) as i64;
    let n_clauses = binary::u32le(data, 16) as i64;
    let n_states = binary::u32le(data, 20) as i64;
    let threshold = binary::u32le(data, 24) as i64;
    let boost = binary::u32le(data, 28) as i64;
    let s = binary::f64_of_bits(binary::u64le(data, 32));
    let epochs = binary::u64le(data, 40);
    let rng_state = binary::u64le(data, 48);

    // the same conditions tm::init asserts, plus limits that keep the size below from overflowing
    guard n_features >= 1 && n_features <= MAX_DIM && n_classes >= 1 && n_classes <= MAX_DIM &&
          n_clauses >= 1 && n_clauses <= MAX_DIM && n_states >= 1 && n_states <= 127 &&
          threshold >= 1 && boost <= 1 && s > 1.0 && (s - s) == 0.0 && epochs < (1 as u64) << 62
          else { return LoadError.BadHeader; }

    let n_automata = n_classes * n_clauses * 2 * n_features;
    guard data.len == HEADER_SIZE + n_automata else { return LoadError.BadSize { HEADER_SIZE + n_automata, data.len }; }

    for i in n_automata {
        let st = data[HEADER_SIZE + i] as i64;
        guard st >= 1 && st <= 2 * n_states else { return LoadError.BadState; }
    }

    // everything checks out: from here on nothing can fail
    var scratch = rng { 0 };                                // init draws random states, which are overwritten
    tm::init(m, n_features, n_classes, n_clauses, n_states, threshold, s, scratch, boost == 1);
    copy_into(m.state, data[HEADER_SIZE..data.len]);
    tm::rebuild_masks(m);
    r.s = rng_state;
    epochs_done = epochs as i64;
    return LoadError.Ok;
}
