import tm;

// Builds a machine with n features, c clauses, N = 10, T = 5 and every state set to `st`.
fn setup(m: tm::TM&, n: i64, c: i64, s: f64, st: u8, r: rng&) {
    tm::init(m, n, 1, c, 10, 5, s, r);
    for i in m.state.len { m.state[i] = st; }
    tm::rebuild_masks(m);
}
fn check(name: u8[:], ok: bool) { print(if ok { "ok   " } else { "FAIL " }, name); assert(ok); }
fn set(m: tm::TM&, j: i64, k: i64, st: u8) {
    m.state[j * 2 * m.n_features + k] = st;
    tm::rebuild_masks(m);
}
fn get(m: tm::TM&, j: i64, k: i64) -> u8 { m.state[j * 2 * m.n_features + k] }


// Runs `trials` Type I updates on an unfired one-clause machine whose 2 * n literals all start just
// inside "include" (state N + 1) and compares how they move with independent coin flips of
// probability p = 1 / s: per position (or per group of positions when moves are rare), the number
// of moves per trial (mean and variance) and the correlation between literals 1, 2 and 7 apart.
fn check_decay(n: i64, s: f64, trials: i64, r: rng&) {
    var m = tm::TM { .. };
    tm::init(m, n, 1, 1, 10, 5, s, r);                      // N = 10: states 11 (include) -> 10 (exclude)
    var zeros: u8[>..] = [];
    for _ in n { zeros.push(0); }
    var x: u64[>..] = [];
    tm::pack(zeros, x);
    let len = 2 * n;
    let p = tm::decay_probability(s);                       // 1/s, rounded to a multiple of 2^-16 when flipping 64 coins at a time
    let q = 1.0 - p;
    var counts: i64[>..] = [];
    counts.resize(len, 0);
    var moved: u8[>..] = [];
    moved.resize(len, 0);
    var total = 0.0;
    var total_sq = 0.0;
    var pairs: i64[4] = [0; 4];
    let lags: i64[4] = [1, 2, 7, 64];                         // 64: the same coin in the next word
    var consistent = true;
    for _ in trials {
        for i in m.state.len { m.state[i] = 11; }
        tm::rebuild_masks(m);
        m.out[0] = 0;                                         // the clause did not fire
        tm::type_i(m, 0, x, r);
        var c = 0;
        for k in len {
            let st = m.state[k];
            let mv = st == 10;
            moved[k] = if mv { 1 } else { 0 };
            if mv { counts[k] += 1; c++; }
            if st != 10 && st != 11 { consistent = false; }
            if tm::included(m, 0, k) != (st == 11) { consistent = false; }
        }
        total += c;
        total_sq += c * c;
        for li in 4 {
            if len > lags[li] {
                for k in len - lags[li] { pairs[li] += moved[k] * moved[k + lags[li]]; }
            }
        }
    }
    let t = trials * 1.0;
    let tag = str("decay n=", n, " s=", s, ": ");

    check(str(tag, "states stay in {N, N+1} and the masks agree"), consistent);
    if q <= 0.0 {                                          // p = 1: every literal moves, every time
        var all = true;
        for k in len { if counts[k] != trials { all = false; } }
        check(str(tag, "p = 1: every literal moves in every trial"), all);
        return;
    }

    // frequency per position (grouped when moves are rare, so that each group sees plenty of them)
    let groups = if t * p >= 100.0 { len } else { min(len, 16) };
    var worst = 0.0;
    for g in groups {
        let lo = g * len / groups;
        let hi = (g + 1) * len / groups;
        var obs = 0;
        for k in lo..hi { obs += counts[k]; }
        let cells = (hi - lo) * t;
        let z = abs(obs - cells * p) / sqrt(cells * p * q);
        if z > worst { worst = z; }
    }
    check(str(tag, "move frequency per position, worst |z| = ", worst), worst < 5.5);

    // moves per trial: mean len * p, variance len * p * q (binomial)
    let mean = total / t;
    let var_obs = (total_sq - t * mean * mean) / (t - 1.0);
    let mu = len * p;
    let sigma2 = len * p * q;
    let z_mean = abs(mean - mu) / sqrt(sigma2 / t);
    let excess = (1.0 - 6.0 * p * q) / sigma2;                 // excess kurtosis of the binomial
    let se_var = sigma2 * sqrt(2.0 / (t - 1.0) + excess / t);
    let z_var = abs(var_obs - sigma2) / se_var;
    check(str(tag, "moves per trial: mean ", mean, " vs ", mu, ", |z| = ", z_mean), z_mean < 5.5);
    check(str(tag, "moves per trial: variance ", var_obs, " vs ", sigma2, ", |z| = ", z_var), z_var < 5.5);

    // two literals a few places apart, or a word apart, move together with probability p^2 (variance inflated for overlapping pairs)
    var worst_pair = 0.0;
    for li in 4 {
        let cells = (len - lags[li]) * t;
        if len > lags[li] && cells * p * p >= 30.0 {
            let z = abs(pairs[li] - cells * p * p) / sqrt(2.0 * cells * p * p * (1.0 - p * p));
            if z > worst_pair { worst_pair = z; }
        }
    }
    check(str(tag, "joint moves at lags 1, 2, 7 and 64, worst |z| = ", worst_pair), worst_pair < 5.5);
}

// Runs `trials` Type I updates on a clause that fired, all of whose 2 * n literals start at state 5
// (excluded, N = 10), on an input whose literals alternate true and false, and compares how they move
// with the probabilities they should have: a true literal steps up with probability (s-1)/s (1 with
// boost, which must never fail), a false one steps down with probability 1/s; nothing else may move.
fn check_fired(n: i64, s: f64, boost: bool, trials: i64, r: rng&) {
    var m = tm::TM { .. };
    tm::init(m, n, 1, 1, 10, 5, s, r, boost);
    var bits: u8[>..] = [];
    for k in n { bits.push((k % 2) as u8); }                 // x_k = k mod 2, so literal k is true iff k is odd
    var x: u64[>..] = [];
    tm::pack(bits, x);
    let len = 2 * n;
    let p_up = if boost { 1.0 } else { (s - 1.0) / s };
    let p_down = 1.0 / s;
    var ups: i64[>..] = [];  ups.resize(len, 0);
    var downs: i64[>..] = []; downs.resize(len, 0);
    var stray = 0;                                           // moves that must never happen
    for _ in trials {
        for i in m.state.len { m.state[i] = 5; }
        tm::rebuild_masks(m);
        m.out[0] = 1;                                        // the clause fired
        tm::type_i(m, 0, x, r);
        for k in len {
            let st = m.state[k];
            let is_true = tm::literal(x, n, k) == 1;
            if st == 6 { if is_true { ups[k] += 1; } else { stray++; } }
            else if st == 4 { if is_true { stray++; } else { downs[k] += 1; } }
            else if st != 5 { stray++; }
        }
    }
    let tag = str("fired n=", n, " s=", s, " boost=", boost, ": ");
    check(str(tag, "true literals only step up, false ones only down"), stray == 0);
    let t = trials * 1.0;
    var worst_up = 0.0;
    var worst_down = 0.0;
    var all_up = true;
    for kind in 2 {                                          // 0: true literals (odd k), 1: false literals
        let p = if kind == 0 { p_up } else { p_down };
        let q = 1.0 - p;
        var cells = 0;
        var obs = 0;
        var worst = 0.0;
        var group_cells = 0;
        var group_obs = 0;
        var in_group = 0;
        let group_size = if t * min(p, q) >= 100.0 { 1 } else { max(1, n / 16) };
        for k in len {
            let literal_true = tm::literal(x, n, k) == 1;
            if literal_true != (kind == 0) { continue; }
            let moved = if kind == 0 { ups[k] } else { downs[k] };
            if kind == 0 && boost && moved != trials { all_up = false; }
            group_cells += trials;
            group_obs += moved;
            in_group++;
            if in_group == group_size {
                if q > 0.0 {
                    let z = abs(group_obs - group_cells * p) / sqrt(group_cells * p * q);
                    if z > worst { worst = z; }
                }
                group_cells = 0; group_obs = 0; in_group = 0;
            }
        }
        if kind == 0 { worst_up = worst; } else { worst_down = worst; }
    }
    if boost { check(str(tag, "with boost every true literal steps up in every trial"), all_up); }
    else { check(str(tag, "true literals step up with probability (s-1)/s, worst |z| = ", worst_up), worst_up < 5.5); }
    check(str(tag, "false literals step down with probability 1/s, worst |z| = ", worst_down), worst_down < 5.5);
}

fn main() {
    var r = rng { 7 };
    // two features: literals are [x0, x1, !x0, !x1]
    var m = tm::TM { .. }; setup(m, 2, 4, 2.0, 5, r);       // all excluded (<= N = 10)
    let x: u8[2] = [1, 0];
    var xp: u64[>..] = [];
    tm::pack(x, xp);

    // --- clause output
    tm::evaluate(m, xp, true);
    check("empty clause outputs 1 while learning", m.out[0] == 1);
    tm::evaluate(m, xp, false);
    check("empty clause outputs 0 when predicting", m.out[0] == 0);
    set(m, 0, 0, 11);                        // include x0
    set(m, 1, 1, 11);                        // include x1
    set(m, 2, 3, 11);                        // include !x1
    set(m, 3, 0, 11); set(m, 3, 3, 11);      // include x0 & !x1
    tm::evaluate(m, xp, false);
    check("x0 true -> 1", m.out[0] == 1);
    check("x1 false -> 0", m.out[1] == 0);
    check("!x1 true -> 1", m.out[2] == 1);
    check("x0 & !x1 true -> 1", m.out[3] == 1);
    let y: u8[2] = [0, 0];
    var yp: u64[>..] = [];
    tm::pack(y, yp);
    tm::evaluate(m, yp, false);
    check("x0 false -> 0", m.out[0] == 0);
    check("x0 & !x1 with x=(0,0) -> 0", m.out[3] == 0);

    // --- class sum: even clauses +, odd -, clipped to +-T (T = 5)
    tm::evaluate(m, xp, false);               // outputs 1, 0, 1, 1
    check("class sum = +1 -0 +1 -1 = 1", tm::class_sum(m, 0) == 1);
    let b1: u8[1] = [1];
    var bp: u64[>..] = [];
    tm::pack(b1, bp);
    var big = tm::TM { .. }; setup(big, 1, 14, 2.0, 5, r);     // 14 empty clauses
    tm::evaluate(big, bp, true);            // all 1: 7 - 7 = 0
    check("balanced sum is 0", tm::class_sum(big, 0) == 0);
    for j in 14 { if j % 2 == 1 { big.out[j] = 0; } }
    check("clipped to +T", tm::class_sum(big, 0) == 5);        // 7 positive, clipped
    for j in 14 { big.out[j] = (j % 2) as u8; }
    check("clipped to -T", tm::class_sum(big, 0) == -5);

    // --- Type II: clause fired -> every excluded false literal moves up by one, nothing else
    var t2 = tm::TM { .. }; setup(t2, 2, 2, 2.0, 5, r);
    set(t2, 0, 0, 11);                       // clause 0 includes x0 and fires on x = (1, 0)
    tm::evaluate(t2, xp, true);
    tm::type_ii(t2, 0, xp);
    check("type II: false literal x1 (excluded) 5 -> 6", get(t2, 0, 1) == 6);
    check("type II: false literal !x0 (excluded) 5 -> 6", get(t2, 0, 2) == 6);
    check("type II: true literal !x1 untouched", get(t2, 0, 3) == 5);
    check("type II: included true literal x0 untouched", get(t2, 0, 0) == 11);
    set(t2, 0, 1, 11);                       // now includes the false x1: clause output 0
    tm::evaluate(t2, xp, true);
    tm::type_ii(t2, 0, xp);
    check("type II: unfired clause unchanged", get(t2, 0, 2) == 6);

    // --- Type I with s huge: p(1 - 1/s) ~ 1 for true literals, 1/s ~ 0 otherwise
    var t1 = tm::TM { .. }; setup(t1, 2, 1, 1.0e12, 5, r);
    tm::evaluate(t1, xp, true);               // empty clause, learning: fires
    tm::type_i(t1, 0, xp, r);
    check("type I fired: true literal x0 5 -> 6", get(t1, 0, 0) == 6);
    check("type I fired: false literal x1 stays 5", get(t1, 0, 1) == 5);
    check("type I fired: false literal !x0 stays 5", get(t1, 0, 2) == 5);
    check("type I fired: true literal !x1 5 -> 6", get(t1, 0, 3) == 6);
    // unfired clause: with s tiny-above-1, p(1/s) ~ 1 so everything decrements
    var t0 = tm::TM { .. }; setup(t0, 2, 1, 1.0000001, 5, r);
    set(t0, 0, 1, 11);                       // includes false x1: clause does not fire
    tm::evaluate(t0, xp, true);
    check("clause with false literal does not fire", t0.out[0] == 0);
    tm::type_i(t0, 0, xp, r);
    var all_down = true;
    for k in 4 { if get(t0, 0, k) >= (if k == 1 { 11 } else { 5 }) { all_down = false; } }
    check("type I unfired: every automaton moves down", all_down);

    // --- saturation at both ends
    var hi = tm::TM { .. }; setup(hi, 2, 1, 1.0e12, 20, r);   // top state is 2 * N = 20
    tm::evaluate(hi, xp, true);
    tm::type_i(hi, 0, xp, r);
    check("state saturates at 2N", get(hi, 0, 0) == 20);
    var lo = tm::TM { .. }; setup(lo, 2, 1, 1.0000001, 1, r);
    set(lo, 0, 1, 11);
    tm::evaluate(lo, xp, true);
    tm::type_i(lo, 0, xp, r);
    check("state saturates at 1", get(lo, 0, 0) == 1);

    // --- update's feedback probability: p = (T - v) / 2T for y = 1, (T + v) / 2T for y = 0.
    // 14 clauses (7 positive, 7 negative), s huge so type I only moves true literals up.
    // x = (1, 0). Clauses that include x1 never fire; empty clauses fire while learning.
    // u: all 7 positive clauses fire, negative clause 1 fires, the other negative ones do not:
    //    v = 7 - 1 = 6, clipped to +T = 5.
    var u = tm::TM { .. }; setup(u, 2, 14, 1.0e12, 5, r);
    for j in 3..14 { if j % 2 == 1 { set(u, j, 1, 11); } }
    let before = copy(u.state);
    tm::update_class(u, 0, xp, 1, r);                                  // y = 1, v = T: p = 0, nothing may change
    check("update y=1 at v=+T changes nothing", u.state == before);
    tm::update_class(u, 0, xp, 0, r);                                  // y = 0, v = +T: p = 1, every clause gets feedback
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
    tm::update_class(w, 0, xp, 0, r);                                  // y = 0, v = -T: p = 0
    check("update y=0 at v=-T changes nothing", w.state == before_w);
    tm::update_class(w, 0, xp, 1, r);                                  // y = 1, v = -T: p = 1
    check("y=1, v=-T: positive clause 0 got type I (true x0 5 -> 6)", get(w, 0, 0) == 6);
    check("y=1, v=-T: negative clause 1 got type II (false x1 5 -> 6)", get(w, 1, 1) == 6);
    check("y=1, v=-T: negative clause 1 type II leaves true x0 alone", get(w, 1, 0) == 5);

    // --- packing: bit k lives in word k / 64, bit k % 64; appends; bits past n stay 0
    var pk: u64[>..] = [];
    var bits70: u8[70] = [0; 70];
    bits70[0] = 1; bits70[63] = 1; bits70[64] = 1; bits70[69] = 1;
    tm::pack(bits70, pk);
    check("pack: 70 bits take 2 words", pk.len == 2 && tm::n_words(70) == 2);
    check("pack: word 0 holds bits 0 and 63", pk[0] == (1 | ((1 as u64) << 63)));
    check("pack: word 1 holds bits 64 and 69 at positions 0 and 5", pk[1] == (1 | ((1 as u64) << 5)));
    tm::pack(bits70, pk);
    check("pack appends", pk.len == 4 && pk[2] == pk[0] && pk[3] == pk[1]);
    check("n_words at the boundaries", tm::n_words(1) == 1 && tm::n_words(64) == 1 && tm::n_words(65) == 2 && tm::n_words(128) == 2 && tm::n_words(129) == 3);

    // --- packed clause evaluation equals the definition, straight from the states,
    // for random states and inputs around the word boundaries
    var agree = true;
    var fired = 0;
    for n in [1, 2, 63, 64, 65, 127, 128, 129, 200] {
        var p = tm::TM { .. };
        tm::init(p, n, 1, 12, 3, 5, 2.0, r);                  // N = 3: random states 1..6, about half included
        for trial in 40 {
            // sparse inclusion makes firing clauses likely: reset most automata to "exclude"
            for i in p.state.len { p.state[i] = if r.rand_int(12) == 0 { 4 + r.rand_int(3) as u8 } else { 1 + r.rand_int(3) as u8 }; }
            tm::rebuild_masks(p);
            var bits: u8[>..] = [];
            for _ in n { bits.push(r.rand_int(2) as u8); }
            var xq: u64[>..] = [];
            tm::pack(bits, xq);
            for learning in [false, true] {
                tm::evaluate(p, xq, learning);
                for j in 12 {
                    var all_true = true;
                    var any = false;
                    for k in n {
                        if p.state[j * 2 * n + k] > 3 { any = true; if bits[k] == 0 { all_true = false; } }
                        if p.state[j * 2 * n + n + k] > 3 { any = true; if bits[k] == 1 { all_true = false; } }
                    }
                    let expected = if all_true && (any || learning) { 1 } else { 0 };
                    if p.out[j] != expected { agree = false; }
                    if expected == 1 && any { fired++; }
                }
            }
        }
    }
    check("packed clause output equals the definition (n up to 200)", agree);
    check("the property test exercised firing, non-empty clauses", fired > 50);

    // --- masks follow the states through training, including across the include boundary
    var q = tm::TM { .. };
    tm::init(q, 70, 1, 6, 4, 4, 2.0, r);
    var consistent = true;
    for step in 400 {
        var bits: u8[>..] = [];
        for _ in 70 { bits.push(r.rand_int(2) as u8); }
        var xq: u64[>..] = [];
        tm::pack(bits, xq);
        tm::update_class(q, 0, xq, bits[0] ^ bits[1], r);
        let saved = copy(q.inc);
        tm::rebuild_masks(q);
        if q.inc != saved { consistent = false; }
    }
    check("masks equal the ones rebuilt from the states after 400 updates", consistent);

    // ===================================================== several classes
    // --- classes own disjoint clause ranges; polarity restarts in every class
    // (3 clauses per class, so the parity of the global index differs from the in-class one)
    var mc = tm::TM { .. };
    tm::init(mc, 2, 3, 3, 10, 5, 2.0, r);                     // 2 features, 3 classes, 3 clauses each
    for i in mc.state.len { mc.state[i] = 5; }
    tm::rebuild_masks(mc);
    for j in 3..6 { set(mc, j, 0, 11); }                       // class 1: every clause includes x0
    set(mc, 6, 0, 11); set(mc, 7, 0, 11); set(mc, 8, 1, 11);  // class 2: x0, x0, x1
    tm::evaluate(mc, xp, false);                               // x = (1, 0)
    check("class 0 (all clauses empty) scores 0", tm::class_sum(mc, 0) == 0);
    check("class 1 scores +1 -1 +1 = 1 (polarity restarts per class)", tm::class_sum(mc, 1) == 1);
    check("class 2 scores +1 -1 +0 = 0", tm::class_sum(mc, 2) == 0);
    check("predict is the highest score", tm::predict(mc, xp) == 1);
    set(mc, 8, 1, 5); set(mc, 8, 0, 11);                       // class 2 now also scores +1
    check("ties go to the lowest class", tm::predict(mc, xp) == 1 && tm::class_sum(mc, 2) == 1);
    var quiet = tm::TM { .. };
    tm::init(quiet, 2, 3, 3, 10, 5, 2.0, r);
    for i in quiet.state.len { quiet.state[i] = 5; }
    tm::rebuild_masks(quiet);
    check("all scores equal -> class 0", tm::predict(quiet, xp) == 0);

    // --- evaluate_class writes only its own class's outputs
    for j in mc.out.len { mc.out[j] = 7; }
    tm::evaluate_class(mc, 1, xp, false);
    var only_own = true;
    for j in 9 { if (j >= 3 && j < 6) == (mc.out[j] == 7) { only_own = false; } }
    check("evaluate_class touches exactly clauses 3..6 of 9", only_own);

    // --- negative_class: never the target, uniform over the others
    var four = tm::TM { .. };
    tm::init(four, 2, 4, 2, 10, 5, 2.0, r);
    var cnt: i64[4] = [0; 4];
    for _ in 4000 { cnt[tm::negative_class(four, 2, r)] += 1; }
    check("negative_class never returns the target", cnt[2] == 0);
    check("negative_class is uniform over the other classes",
          cnt[0] > 1150 && cnt[0] < 1520 && cnt[1] > 1150 && cnt[1] < 1520 && cnt[3] > 1150 && cnt[3] < 1520);

    // --- update: the target class and exactly one other class receive feedback.
    // x = (1, 0, 1, 0), T = 1, s huge. The target class is set to v = -T (so y = 1 has p = 1, and
    // its firing negative clause gets type II); every other class to v = +T (so y = 0 has p = 1,
    // and its firing positive clause gets type II).
    let xb: u8[4] = [1, 0, 1, 0];
    var xb_p: u64[>..] = [];
    tm::pack(xb, xb_p);
    var uc = tm::TM { .. };
    tm::init(uc, 4, 3, 2, 10, 1, 1.0e12, r);
    let per_class = 2 * 2 * 4;                                 // clauses * 2 * features
    var changed_by_class: i64[3] = [0; 3];
    var bad = 0;
    for trial in 300 {
        let t = trial % 3;
        for i in uc.state.len { uc.state[i] = 5; }
        for c in 3 {
            if c == t { uc.state[(c * 2) * 8 + 1] = 11; } else { uc.state[(c * 2 + 1) * 8 + 1] = 11; }
        }
        tm::rebuild_masks(uc);
        let before = copy(uc.state);
        tm::update(uc, xb_p, t, r);
        var n_changed = 0;
        var target_changed = false;
        for c in 3 {
            var differs = false;
            for i in c * per_class..(c + 1) * per_class { if uc.state[i] != before[i] { differs = true; } }
            if differs {
                n_changed++;
                if c == t { target_changed = true; } else { changed_by_class[c] += 1; }
            }
        }
        if n_changed != 2 || !target_changed { bad++; }
    }
    check("update changes the target class and exactly one other class, every time", bad == 0);
    check("every class is drawn as the negative (300 trials, 100 per target)",
          changed_by_class[0] > 50 && changed_by_class[1] > 50 && changed_by_class[2] > 50);

    // ============================================ sampling of the decay of an unfired clause
    // Each literal of a clause that did not fire must move towards "exclude" with probability 1/s,
    // independently of the others, however that is sampled.
    check("ln_one_minus matches ln(1 - p): 0.5", abs(tm::ln_one_minus(0.5) - log(0.5)) < 1.0e-15);
    check("ln_one_minus matches ln(1 - p): 1e-3", abs(tm::ln_one_minus(1.0e-3) / log(1.0 - 1.0e-3) - 1.0) < 1.0e-12);
    check("ln_one_minus for tiny p: 1e-9", abs(tm::ln_one_minus(1.0e-9) / -1.0000000005e-9 - 1.0) < 1.0e-12);
    // either side of the point where it switches from the series to log(1 - p), against a 4-term series
    for p in [0.99999e-4, 1.00001e-4] {
        let series = -(p + p * p / 2.0 + p * p * p / 3.0 + p * p * p * p / 4.0);
        check(str("ln_one_minus agrees with the series near the switch, p = ", p), abs(tm::ln_one_minus(p) / series - 1.0) < 1.0e-11);
    }
    check("decay probability: s = 10 is 6554 / 65536 (coins flipped 64 at a time)", tm::decay_probability(10.0) == 6554.0 / 65536.0);
    check("decay probability: s = 32 is exactly 1/32", tm::decay_probability(32.0) == 1.0 / 32.0);
    check("decay probability: s = 100 is exactly 1/100 (gaps)", tm::decay_probability(100.0) == 1.0 / 100.0);
    check("decay probability: s just above 1 rounds to 1", tm::decay_probability(1.0000001) == 1.0);
    check_decay(1, 2.0, 20000, r);            // the smallest clause: 2 literals
    check_decay(3, 1.0000001, 2000, r);       // p ~ 1: everything moves
    check_decay(3, 1.0e12, 2000, r);          // p ~ 0: nothing moves
    check_decay(100, 1.25, 20000, r);         // p = 0.8
    check_decay(100, 2.0, 20000, r);          // p = 0.5
    check_decay(100, 3.9, 20000, r);          // p = 0.256
    check_decay(100, 10.0, 20000, r);         // p = 0.1
    check_decay(100, 31.0, 20000, r);         // the last s that flips coins 64 at a time, below s = 32
    check_decay(100, 33.0, 20000, r);         // the first that draws gaps
    check_decay(32, 5.0, 20000, r);           // 64 literals: exactly one full word
    check_decay(33, 5.0, 20000, r);           // 66 literals: a second word with 2 literals
    check_decay(784, 10.0, 6000, r);          // MNIST sized
    check_decay(784, 100.0, 6000, r);         // p = 0.01
    check_decay(784, 1000.0, 6000, r);        // p = 0.001

    // ======================================================================== boost
    // s ~ 1 makes (s-1)/s ~ 1e-7: without boost a true literal of a fired clause practically never
    // steps up, with boost it always does; false literals decay the same either way.
    let xb2: u8[2] = [1, 0];                                  // literals [x0, x1, !x0, !x1] = [1, 0, 0, 1]
    var xb2p: u64[>..] = [];
    tm::pack(xb2, xb2p);
    var nb = tm::TM { .. }; setup(nb, 2, 1, 1.0000001, 5, r);
    tm::evaluate(nb, xb2p, true);
    tm::type_i(nb, 0, xb2p, r);
    check("no boost, s ~ 1: true literals of a fired clause stay put", get(nb, 0, 0) == 5 && get(nb, 0, 3) == 5);
    check("no boost, s ~ 1: false literals step down", get(nb, 0, 1) == 4 && get(nb, 0, 2) == 4);
    var bb = tm::TM { .. }; setup(bb, 2, 1, 1.0000001, 5, r); bb.boost = true;
    tm::evaluate(bb, xb2p, true);
    tm::type_i(bb, 0, xb2p, r);
    check("boost, s ~ 1: true literals step up", get(bb, 0, 0) == 6 && get(bb, 0, 3) == 6);
    check("boost leaves false literals to the usual decay", get(bb, 0, 1) == 4 && get(bb, 0, 2) == 4);
    var bt = tm::TM { .. }; setup(bt, 2, 1, 1.0000001, 5, r); bt.boost = true;
    set(bt, 0, 0, 20); set(bt, 0, 3, 20);                     // the two true literals are included, at the top state 2N = 20
    tm::evaluate(bt, xb2p, true);
    check("(the clause fires: it includes only true literals)", bt.out[0] == 1);
    tm::type_i(bt, 0, xb2p, r);
    check("boost: states saturate at 2N", get(bt, 0, 0) == 20 && get(bt, 0, 3) == 20);
    var bc = tm::TM { .. }; setup(bc, 2, 1, 1.0000001, 10, r); bc.boost = true;    // state N: the last excluded one
    tm::evaluate(bc, xb2p, true);
    tm::type_i(bc, 0, xb2p, r);
    check("boost: a true literal crossing N + 1 is included and the mask follows",
          get(bc, 0, 0) == 11 && tm::included(bc, 0, 0) && get(bc, 0, 3) == 11 && tm::included(bc, 0, 3) && !tm::included(bc, 0, 1));
    var bu = tm::TM { .. }; setup(bu, 2, 1, 1.0000001, 5, r); bu.boost = true;
    set(bu, 0, 1, 11);                                        // includes the false literal x1: the clause does not fire
    tm::evaluate(bu, xb2p, true);
    tm::type_i(bu, 0, xb2p, r);
    check("boost does not touch an unfired clause: every automaton steps down",
          bu.out[0] == 0 && get(bu, 0, 0) == 4 && get(bu, 0, 1) == 10 && get(bu, 0, 2) == 4 && get(bu, 0, 3) == 4);

    check_fired(50, 3.9, false, 20000, r);
    check_fired(50, 3.9, true, 20000, r);
    check_fired(50, 10.0, false, 20000, r);
    check_fired(50, 10.0, true, 20000, r);
    check_fired(3, 2.0, false, 20000, r);
    check_fired(3, 2.0, true, 20000, r);
    check_fired(200, 40.0, false, 6000, r);                   // large s: a true literal fails to step up 1 time in 40
    check_fired(200, 40.0, true, 6000, r);

    // masks follow the states through training with boost, as without
    var qb = tm::TM { .. };
    tm::init(qb, 70, 1, 6, 4, 4, 2.0, r, true);
    var consistent_boost = true;
    for step in 400 {
        var bits: u8[>..] = [];
        for _ in 70 { bits.push(r.rand_int(2) as u8); }
        var xq: u64[>..] = [];
        tm::pack(bits, xq);
        tm::update_class(qb, 0, xq, bits[0] ^ bits[1], r);
        let saved = copy(qb.inc);
        tm::rebuild_masks(qb);
        if qb.inc != saved { consistent_boost = false; }
    }
    check("with boost the masks equal the ones rebuilt from the states after 400 updates", consistent_boost);
    print("all checks passed");
}
