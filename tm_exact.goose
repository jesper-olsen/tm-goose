// Tsetlin machine core (Granmo 2018): a multiclass machine, no weights, no boosting.
//
// Inputs are Boolean feature vectors packed one bit per feature into u64 words
// (bit k of the example is bit k % 64 of word k / 64; see `pack`). Each class has
// `n_clauses` clauses of its own, each a conjunction over 2 * n_features literals
// (x_k and !x_k), and each literal has a Tsetlin automaton with 2 * n_states states:
// states 1..n_states exclude the literal, n_states+1..2*n_states include it. Within a
// class, even-numbered clauses vote for it and odd-numbered ones against it. A class
// scores an example by its clipped vote sum; `predict` takes the highest score.
//
// All classes live in one struct, because Goose cannot hold several of these in an
// array: clause i of class c is clause j = c * n_clauses + i, the index that
// `included`, `format_clause` and the feedback functions take.
//
// The automaton states are the truth. Alongside them each clause keeps a bit mask of
// the literals it includes, so that evaluating a clause is a few word operations
// instead of a loop over literals; the mask bit of a literal flips exactly when its
// state crosses between n_states and n_states + 1.
//
//   var m = tm::TM { .. };
//   tm::init(m, n_features, n_classes, n_clauses, n_states, threshold, s, r);
//   var xs: u64[>..] = [];
//   tm::pack(bits, xs);                  // appends tm::n_words(n_features) words
//   tm::update(m, xs[i * w..(i + 1) * w], label, r);   // one example, label in 0..n_classes
//   let label_hat = tm::predict(m, xs[i * w..(i + 1) * w]);
//
// `update` is the standard multiclass step: the true class gets y = 1 feedback and one
// other class, drawn by `negative_class`, gets y = 0 feedback. `update_class` is the
// underlying single-class step, usable directly (e.g. with n_classes = 1 as a binary
// machine scored by `class_sum(m, 0) >= 0`).
//
// `TM` is a root struct (independently sized arrays): keep it in a variable.
namespace tm_exact;

import std;

root struct TM {
    n_features: i64 = 0,
    n_words: i64 = 0,        // u64 words per packed example: n_words(n_features)
    n_classes: i64 = 0,
    n_clauses: i64 = 0,      // clauses per class
    n_states: i64 = 0,       // N: automaton states per action; 1 <= N <= 127
    threshold: i64 = 0,      // T: votes are clipped to -T..T; larger T trains more clauses per example
    s: f64 = 1.0,            // specificity: larger s gives longer, more specific clauses
    // Automaton states, one row of 2 * n_features per clause (clauses of class 0 first):
    // first the n_features literals x_k, then the n_features negated literals !x_k.
    state: u8[>..] = [],
    // Included-literal masks, one row of 2 * n_words per clause: n_words words for the
    // x_k literals, then n_words for the !x_k literals; bit k of a half is literal k.
    // Bits past n_features are always 0.
    inc: u64[>..] = [],
    // Output of each clause for the example last passed to `evaluate` / `evaluate_class`.
    out: u8[>..] = [],
}

// Words needed to pack n Boolean features.
fn n_words(n_features: i64) -> i64 { (n_features + 63) / 64 }

// Appends the n_words(bits.len) words packing `bits` (each 0 or 1) to `out`.
fn pack(bits: u8[:], out: u64[>..]&) {
    let first = out.len;
    out.resize(first + n_words(bits.len), 0);
    for k in bits.len {
        assert(bits[k] <= 1);
        if bits[k] == 1 { out[first + k / 64] |= (1 as u64) << (k % 64); }
    }
}

// Bit k of a packed example.
fn bit(x: u64[:], k: i64) -> u8 { ((x[k / 64] >> (k % 64)) & 1) as u8 }

// Value of literal k of a packed example: k < n is x_k, k >= n is !x_(k - n).
fn literal(x: u64[:], n: i64, k: i64) -> u8 {
    if k < n { bit(x, k) } else { 1 - bit(x, k - n) }
}

// Where literal k of clause j sits in `inc`: the word, and the bit within it.
fn inc_word(m: TM&, j: i64, k: i64) -> i64 {
    let w = m.n_words;
    if k < m.n_features { j * 2 * w + k / 64 } else { j * 2 * w + w + (k - m.n_features) / 64 }
}
fn inc_bit(m: TM&, k: i64) -> u64 {
    let b = if k < m.n_features { k } else { k - m.n_features };
    (1 as u64) << (b % 64)
}
fn set_inc(m: TM&, j: i64, k: i64) { m.inc[inc_word(m, j, k)] |= inc_bit(m, k); }
fn clear_inc(m: TM&, j: i64, k: i64) { m.inc[inc_word(m, j, k)] &= ~inc_bit(m, k); }

fn included(m: TM&, j: i64, k: i64) -> bool {
    (m.inc[inc_word(m, j, k)] & inc_bit(m, k)) != 0
}

// Recomputes every mask from the automaton states. `init` calls it; call it after
// writing `state` directly.
fn rebuild_masks(m: TM&) {
    for i in m.inc.len { m.inc[i] = 0; }
    for j in m.n_classes * m.n_clauses {
        for k in 2 * m.n_features {
            if m.state[j * 2 * m.n_features + k] > m.n_states { set_inc(m, j, k); }
        }
    }
}

fn init(m: TM&, n_features: i64, n_classes: i64, n_clauses: i64, n_states: i64, threshold: i64, s: f64, r: rng&) {
    assert(n_features > 0 && n_classes > 0 && n_clauses > 0 && threshold > 0 && s > 1.0);
    assert(n_states >= 1 && n_states <= 127);
    m.n_features = n_features;
    m.n_words = n_words(n_features);
    m.n_classes = n_classes;
    m.n_clauses = n_clauses;
    m.n_states = n_states;
    m.threshold = threshold;
    m.s = s;
    m.state.clear();
    m.inc.clear();
    m.out.clear();
    m.state.resize(n_classes * n_clauses * 2 * n_features, 0);
    m.inc.resize(n_classes * n_clauses * 2 * m.n_words, 0);
    m.out.resize(n_classes * n_clauses, 0);
    // Start every automaton on the boundary, at random on either side of it.
    for i in m.state.len { m.state[i] = (n_states + r.rand_int(2)) as u8; }
    rebuild_masks(m);
}

// Output of clause j: 1 iff every included literal is true, i.e. no included x_k is
// 0 and no included !x_k has x_k = 1. A clause that includes no literals outputs 1
// while learning and 0 when predicting.
fn clause_output(m: TM&, j: i64, x: u64[:], learning: bool) -> u8 {
    let w = m.n_words;
    let base = j * 2 * w;
    var any: u64 = 0;
    for i in w {
        let pos = m.inc[base + i];
        let neg = m.inc[base + w + i];
        if ((pos & ~x[i]) | (neg & x[i])) != 0 { return 0; }
        any |= pos | neg;
    }
    return if any == 0 && !learning { 0 } else { 1 };
}

// Outputs of the clauses of class c for example x.
fn evaluate_class(m: TM&, c: i64, x: u64[:], learning: bool) {
    assert(x.len == m.n_words);
    for j in c * m.n_clauses..(c + 1) * m.n_clauses { m.out[j] = clause_output(m, j, x, learning); }
}

// Outputs of the clauses of every class.
fn evaluate(m: TM&, x: u64[:], learning: bool) {
    for c in m.n_classes { evaluate_class(m, c, x, learning); }
}

// Class c's score: its positive clauses (even i) add their output, its negative
// clauses (odd i) subtract it; clipped to -T..T. Uses the outputs of the last
// `evaluate` / `evaluate_class` of that class.
fn class_sum(m: TM&, c: i64) -> i64 {
    var v = 0;
    for i in m.n_clauses {
        let o = m.out[c * m.n_clauses + i] as i64;
        v += if i % 2 == 0 { o } else { -o };
    }
    return clamp(v, -m.threshold, m.threshold);
}

// The class with the highest score; the lowest-numbered one among equals.
fn predict(m: TM&, x: u64[:]) -> i64 {
    evaluate(m, x, false);
    var best = 0;
    var best_sum = class_sum(m, 0);
    for c in 1..m.n_classes {
        let v = class_sum(m, c);
        if v > best_sum { best = c; best_sum = v; }
    }
    return best;
}

// Type I feedback: reinforces clauses to recognize the example (used for y = 1 on
// positive clauses and y = 0 on negative ones).
fn type_i(m: TM&, j: i64, x: u64[:], r: rng&) {
    let n = m.n_features;
    let base = j * 2 * n;
    let top = 2 * m.n_states;
    let p_hit = (m.s - 1.0) / m.s;
    let p_miss = 1.0 / m.s;
    let fired = m.out[j] == 1;
    for k in 2 * n {
        let st = m.state[base + k] as i64;
        if fired && literal(x, n, k) == 1 {
            // Clause true and literal true: move towards including it.
            if r.rand_flt() < p_hit && st < top {
                m.state[base + k] = (st + 1) as u8;
                if st == m.n_states { set_inc(m, j, k); }
            }
        } else {
            // Otherwise: move towards excluding it.
            if r.rand_flt() < p_miss && st > 1 {
                m.state[base + k] = (st - 1) as u8;
                if st == m.n_states + 1 { clear_inc(m, j, k); }
            }
        }
    }
}

// Type II feedback: makes a clause that wrongly fired stop doing so, by including
// a false literal it has excluded (used for y = 0 on positive clauses and y = 1 on
// negative ones).
fn type_ii(m: TM&, j: i64, x: u64[:]) {
    if m.out[j] == 0 { return; }
    let n = m.n_features;
    let base = j * 2 * n;
    for k in 2 * n {
        let st = m.state[base + k] as i64;
        if literal(x, n, k) == 0 && st <= m.n_states {
            m.state[base + k] = (st + 1) as u8;
            if st == m.n_states { set_inc(m, j, k); }
        }
    }
}

// One training step of class c on example x with target y in {0, 1}: y = 1 if the
// example belongs to c, 0 if not.
fn update_class(m: TM&, c: i64, x: u64[:], y: i64, r: rng&) {
    assert(y == 0 || y == 1);
    evaluate_class(m, c, x, true);
    let v = class_sum(m, c);
    let t = m.threshold;
    // Probability that a clause receives feedback: high while the vote is far from
    // the target (T for y = 1, -T for y = 0), zero once it is reached.
    let p = if y == 1 { (t - v) as f64 / (2 * t) } else { (t + v) as f64 / (2 * t) };
    for i in m.n_clauses {
        if r.rand_flt() < p {
            let positive = i % 2 == 0;
            let j = c * m.n_clauses + i;
            if (y == 1) == positive { type_i(m, j, x, r); } else { type_ii(m, j, x); }
        }
    }
}

// The class that gets y = 0 feedback for an example of class `target`: one of the
// other classes, uniformly at random (as in Granmo's multiclass machine).
fn negative_class(m: TM&, target: i64, r: rng&) -> i64 {
    let c = r.rand_int(m.n_classes - 1);
    return if c >= target { c + 1 } else { c };
}

// One multiclass training step on example (x, target): the target class learns to
// recognize it, one other class learns to reject it, the rest are left alone.
fn update(m: TM&, x: u64[:], target: i64, r: rng&) {
    assert(m.n_classes >= 2 && target >= 0 && target < m.n_classes);
    update_class(m, target, x, 1, r);
    update_class(m, negative_class(m, target, r), x, 0, r);
}

// Writes clause j (class j / n_clauses, clause j % n_clauses within it) as the
// conjunction of its included literals, e.g. "x0 & !x1"; an empty clause is written
// "(empty)".
fn format_clause(out: u8[>..]&, m: TM&, j: i64) {
    var any = false;
    for k in 2 * m.n_features {
        if included(m, j, k) {
            if any { format(out, " & "); }
            if k < m.n_features { format(out, "x", k); } else { format(out, "!x", k - m.n_features); }
            any = true;
        }
    }
    if !any { format(out, "(empty)"); }
}
