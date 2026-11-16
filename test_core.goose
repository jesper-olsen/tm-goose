import tm;

// Builds a machine with n features, c clauses, N = 10, T = 5 and every state set to `st`.
fn setup(m: tm::TM&, n: i64, c: i64, s: f64, st: u8, r: rng&) {
    tm::init(m, n, c, 10, 5, s, r);
    for i in m.state.len { m.state[i] = st; }
}
fn check(name: u8[:], ok: bool) { print(if ok { "ok   " } else { "FAIL " }, name); assert(ok); }
fn set(m: tm::TM&, j: i64, k: i64, st: u8) { m.state[j * 2 * m.n_features + k] = st; }
fn get(m: tm::TM&, j: i64, k: i64) -> u8 { m.state[j * 2 * m.n_features + k] }

fn main() {
    var r = rng { 7 };
    // two features: literals are [x0, x1, !x0, !x1]
    var m = tm::TM { .. }; setup(m, 2, 4, 2.0, 5, r);       // all excluded (<= N = 10)
    let x: u8[2] = [1, 0];

    // --- clause output
    tm::evaluate(m, x, true);
    check("empty clause outputs 1 while learning", m.out[0] == 1);
    tm::evaluate(m, x, false);
    check("empty clause outputs 0 when predicting", m.out[0] == 0);
    set(m, 0, 0, 11);                        // include x0
    set(m, 1, 1, 11);                        // include x1
    set(m, 2, 3, 11);                        // include !x1
    set(m, 3, 0, 11); set(m, 3, 3, 11);      // include x0 & !x1
    tm::evaluate(m, x, false);
    check("x0 true -> 1", m.out[0] == 1);
    check("x1 false -> 0", m.out[1] == 0);
    check("!x1 true -> 1", m.out[2] == 1);
    check("x0 & !x1 true -> 1", m.out[3] == 1);
    let y: u8[2] = [0, 0];
    tm::evaluate(m, y, false);
    check("x0 false -> 0", m.out[0] == 0);
    check("x0 & !x1 with x=(0,0) -> 0", m.out[3] == 0);

    // --- class sum: even clauses +, odd -, clipped to +-T (T = 5)
    tm::evaluate(m, x, false);               // outputs 1, 0, 1, 1
    check("class sum = +1 -0 +1 -1 = 1", tm::class_sum(m) == 1);
    var big = tm::TM { .. }; setup(big, 1, 14, 2.0, 5, r);     // 14 empty clauses
    tm::evaluate(big, [1], true);            // all 1: 7 - 7 = 0
    check("balanced sum is 0", tm::class_sum(big) == 0);
    for j in 14 { if j % 2 == 1 { big.out[j] = 0; } }
    check("clipped to +T", tm::class_sum(big) == 5);        // 7 positive, clipped
    for j in 14 { big.out[j] = (j % 2) as u8; }
    check("clipped to -T", tm::class_sum(big) == -5);

    // --- Type II: clause fired -> every excluded false literal moves up by one, nothing else
    var t2 = tm::TM { .. }; setup(t2, 2, 2, 2.0, 5, r);
    set(t2, 0, 0, 11);                       // clause 0 includes x0 and fires on x = (1, 0)
    tm::evaluate(t2, x, true);
    tm::type_ii(t2, 0, x);
    check("type II: false literal x1 (excluded) 5 -> 6", get(t2, 0, 1) == 6);
    check("type II: false literal !x0 (excluded) 5 -> 6", get(t2, 0, 2) == 6);
    check("type II: true literal !x1 untouched", get(t2, 0, 3) == 5);
    check("type II: included true literal x0 untouched", get(t2, 0, 0) == 11);
    set(t2, 0, 1, 11);                       // now includes the false x1: clause output 0
    tm::evaluate(t2, x, true);
    tm::type_ii(t2, 0, x);
    check("type II: unfired clause unchanged", get(t2, 0, 2) == 6);

    // --- Type I with s huge: p(1 - 1/s) ~ 1 for true literals, 1/s ~ 0 otherwise
    var t1 = tm::TM { .. }; setup(t1, 2, 1, 1.0e12, 5, r);
    tm::evaluate(t1, x, true);               // empty clause, learning: fires
    tm::type_i(t1, 0, x, r);
    check("type I fired: true literal x0 5 -> 6", get(t1, 0, 0) == 6);
    check("type I fired: false literal x1 stays 5", get(t1, 0, 1) == 5);
    check("type I fired: false literal !x0 stays 5", get(t1, 0, 2) == 5);
    check("type I fired: true literal !x1 5 -> 6", get(t1, 0, 3) == 6);
    // unfired clause: with s tiny-above-1, p(1/s) ~ 1 so everything decrements
    var t0 = tm::TM { .. }; setup(t0, 2, 1, 1.0000001, 5, r);
    set(t0, 0, 1, 11);                       // includes false x1: clause does not fire
    tm::evaluate(t0, x, true);
    check("clause with false literal does not fire", t0.out[0] == 0);
    tm::type_i(t0, 0, x, r);
    var all_down = true;
    for k in 4 { if get(t0, 0, k) >= (if k == 1 { 11 } else { 5 }) { all_down = false; } }
    check("type I unfired: every automaton moves down", all_down);

    // --- saturation at both ends
    var hi = tm::TM { .. }; setup(hi, 2, 1, 1.0e12, 20, r);   // top state is 2 * N = 20
    tm::evaluate(hi, x, true);
    tm::type_i(hi, 0, x, r);
    check("state saturates at 2N", get(hi, 0, 0) == 20);
    var lo = tm::TM { .. }; setup(lo, 2, 1, 1.0000001, 1, r);
    set(lo, 0, 1, 11);
    tm::evaluate(lo, x, true);
    tm::type_i(lo, 0, x, r);
    check("state saturates at 1", get(lo, 0, 0) == 1);

    // --- update's feedback probability: p = (T - v) / 2T for y = 1, (T + v) / 2T for y = 0.
    // 14 clauses (7 positive, 7 negative), s huge so type I only moves true literals up.
    // x = (1, 0). Clauses that include x1 never fire; empty clauses fire while learning.
    // u: all 7 positive clauses fire, negative clause 1 fires, the other negative ones do not:
    //    v = 7 - 1 = 6, clipped to +T = 5.
    var u = tm::TM { .. }; setup(u, 2, 14, 1.0e12, 5, r);
    for j in 3..14 { if j % 2 == 1 { set(u, j, 1, 11); } }
    let before = copy(u.state);
    tm::update(u, x, 1, r);                                  // y = 1, v = T: p = 0, nothing may change
    check("update y=1 at v=+T changes nothing", u.state == before);
    tm::update(u, x, 0, r);                                  // y = 0, v = +T: p = 1, every clause gets feedback
    check("y=0, v=+T: positive clause 0 got type II (false x1 5 -> 6)", get(u, 0, 1) == 6);
    check("y=0, v=+T: positive clause 0 type II leaves true x0 alone", get(u, 0, 0) == 5);
    check("y=0, v=+T: negative clause 1 got type I (true x0 5 -> 6)", get(u, 1, 0) == 6);
    check("y=0, v=+T: negative clause 1 got type I (true !x1 5 -> 6)", get(u, 1, 3) == 6);
    check("y=0, v=+T: negative clause 1 type I leaves false x1 alone", get(u, 1, 1) == 5);
    check("y=0, v=+T: unfired negative clause 3 untouched by type I (s huge)", get(u, 3, 1) == 11);

    // w: the mirror image. All 7 negative clauses fire, positive clause 0 fires, the other
    // positive ones do not: v = 1 - 7 = -6, clipped to -T.
    var w = tm::TM { .. }; setup(w, 2, 14, 1.0e12, 5, r);
    for j in 2..14 { if j % 2 == 0 { set(w, j, 1, 11); } }
    let before_w = copy(w.state);
    tm::update(w, x, 0, r);                                  // y = 0, v = -T: p = 0
    check("update y=0 at v=-T changes nothing", w.state == before_w);
    tm::update(w, x, 1, r);                                  // y = 1, v = -T: p = 1
    check("y=1, v=-T: positive clause 0 got type I (true x0 5 -> 6)", get(w, 0, 0) == 6);
    check("y=1, v=-T: negative clause 1 got type II (false x1 5 -> 6)", get(w, 1, 1) == 6);
    check("y=1, v=-T: negative clause 1 type II leaves true x0 alone", get(w, 1, 0) == 5);
    print("all checks passed");
}
