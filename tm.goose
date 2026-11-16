// Tsetlin machine core (Granmo 2018): a two-class machine, no weights, no boosting.
//
// Inputs are Boolean feature vectors, passed as `u8` slices holding 0 or 1.
// Each of the `n_clauses` clauses is a conjunction over 2 * n_features literals
// (x_k and !x_k), and each literal has a Tsetlin automaton with 2 * n_states
// states: states 1..n_states exclude the literal, n_states+1..2*n_states include it.
// Even-numbered clauses vote for class 1 and odd-numbered ones against it.
//
//   var m = tm::TM { .. };
//   tm::init(m, n_features, n_clauses, n_states, threshold, s, r);
//   tm::update(m, x, y, r);              // one training example, y in {0, 1}
//   let y_hat = tm::predict(m, x);
//
// `TM` is a root struct (two independently sized arrays): keep it in a variable.
namespace tm;

import std;

root struct TM {
    n_features: i64 = 0,
    n_clauses: i64 = 0,
    n_states: i64 = 0,       // N: automaton states per action; 1 <= N <= 127
    threshold: i64 = 0,      // T: votes are clipped to -T..T; larger T trains more clauses per example
    s: f64 = 1.0,            // specificity: larger s gives longer, more specific clauses
    // Automaton states, one row of 2 * n_features per clause: first the n_features
    // literals x_k, then the n_features negated literals !x_k.
    state: u8[>..] = [],
    // Output of each clause for the example last passed to `evaluate`.
    out: u8[>..] = [],
}

fn init(m: TM&, n_features: i64, n_clauses: i64, n_states: i64, threshold: i64, s: f64, r: rng&) {
    assert(n_features > 0 && n_clauses > 0 && threshold > 0 && s > 1.0);
    assert(n_states >= 1 && n_states <= 127);
    m.n_features = n_features;
    m.n_clauses = n_clauses;
    m.n_states = n_states;
    m.threshold = threshold;
    m.s = s;
    m.state.clear();
    m.out.clear();
    m.state.resize(n_clauses * 2 * n_features, 0);
    m.out.resize(n_clauses, 0);
    // Start every automaton on the boundary, at random on either side of it.
    for i in m.state.len { m.state[i] = (n_states + r.rand_int(2)) as u8; }
}

fn included(m: TM&, j: i64, k: i64) -> bool {
    m.state[j * 2 * m.n_features + k] > m.n_states
}

// Output of clause j: 1 iff every included literal is true. A clause that includes
// no literals outputs 1 while learning and 0 when predicting.
fn clause_output(m: TM&, j: i64, x: u8[:], learning: bool) -> u8 {
    let n = m.n_features;
    var empty = true;
    for k in n {
        if included(m, j, k) {
            empty = false;
            if x[k] == 0 { return 0; }
        }
        if included(m, j, n + k) {
            empty = false;
            if x[k] == 1 { return 0; }
        }
    }
    return if empty && !learning { 0 } else { 1 };
}

fn evaluate(m: TM&, x: u8[:], learning: bool) {
    assert(x.len == m.n_features);
    for j in m.n_clauses { m.out[j] = clause_output(m, j, x, learning); }
}

// Positive clauses (even j) add their output, negative clauses (odd j) subtract it;
// clipped to -T..T. Uses the outputs of the last `evaluate`.
fn class_sum(m: TM&) -> i64 {
    var v = 0;
    for j in m.n_clauses {
        let o = m.out[j] as i64;
        v += if j % 2 == 0 { o } else { -o };
    }
    return clamp(v, -m.threshold, m.threshold);
}

fn predict(m: TM&, x: u8[:]) -> i64 {
    evaluate(m, x, false);
    return if class_sum(m) >= 0 { 1 } else { 0 };
}

fn literal(x: u8[:], n: i64, k: i64) -> u8 {
    if k < n { x[k] } else { 1 - x[k - n] }
}

// Type I feedback: reinforces clauses to recognize the example (used for y = 1 on
// positive clauses and y = 0 on negative ones).
fn type_i(m: TM&, j: i64, x: u8[:], r: rng&) {
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
            if r.rand_flt() < p_hit && st < top { m.state[base + k] = (st + 1) as u8; }
        } else {
            // Otherwise: move towards excluding it.
            if r.rand_flt() < p_miss && st > 1 { m.state[base + k] = (st - 1) as u8; }
        }
    }
}

// Type II feedback: makes a clause that wrongly fired stop doing so, by including
// a false literal it has excluded (used for y = 0 on positive clauses and y = 1 on
// negative ones).
fn type_ii(m: TM&, j: i64, x: u8[:]) {
    if m.out[j] == 0 { return; }
    let n = m.n_features;
    let base = j * 2 * n;
    for k in 2 * n {
        let st = m.state[base + k] as i64;
        if literal(x, n, k) == 0 && st <= m.n_states { m.state[base + k] = (st + 1) as u8; }
    }
}

// One training step on example (x, y), y in {0, 1}.
fn update(m: TM&, x: u8[:], y: i64, r: rng&) {
    assert(y == 0 || y == 1);
    evaluate(m, x, true);
    let v = class_sum(m);
    let t = m.threshold;
    // Probability that a clause receives feedback: high while the vote is far from
    // the target (T for y = 1, -T for y = 0), zero once it is reached.
    let p = if y == 1 { (t - v) as f64 / (2 * t) } else { (t + v) as f64 / (2 * t) };
    for j in m.n_clauses {
        if r.rand_flt() < p {
            let positive = j % 2 == 0;
            if (y == 1) == positive { type_i(m, j, x, r); } else { type_ii(m, j, x); }
        }
    }
}

// Writes clause j as the conjunction of its included literals, e.g. "x0 & !x1";
// an empty clause is written "(empty)".
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
