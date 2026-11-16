// Noisy XOR, the classic Tsetlin machine toy problem (Granmo 2018): 12 Boolean
// inputs, of which only the first two matter; y = x0 XOR x1. Training labels are
// flipped with probability NOISE, the test set is noise free.
import tm;

const N_FEATURES = 12;
const N_TRAIN = 5000;
const N_TEST = 5000;
const NOISE = 0.4;
const EPOCHS = 200;

const N_CLAUSES = 10;
const N_STATES = 100;
const THRESHOLD = 15;
const SPECIFICITY = 3.9;

// Fills xs with n rows of N_FEATURES random bits and ys with the XOR labels,
// each flipped with probability `noise`.
fn make_data(n: i64, noise: f64, r: rng&, xs: u8[>..]&, ys: u8[>..]&) {
    for _ in n {
        let row = xs.len;
        for _ in N_FEATURES { xs.push(r.rand_int(2) as u8); }
        var y = xs[row] ^ xs[row + 1];
        if r.rand_flt() < noise { y = 1 - y; }
        ys.push(y);
    }
}

fn accuracy(m: tm::TM&, xs: u8[:], ys: u8[:]) -> f64 {
    var correct = 0;
    for i in ys.len {
        if tm::predict(m, xs[i * N_FEATURES..(i + 1) * N_FEATURES]) == ys[i] { correct++; }
    }
    return correct as f64 / ys.len;
}

fn main() {
    var r = rng { 42 };
    var x_train: u8[>..] = [];
    var y_train: u8[>..] = [];
    var x_test: u8[>..] = [];
    var y_test: u8[>..] = [];
    make_data(N_TRAIN, NOISE, r, x_train, y_train);
    make_data(N_TEST, 0.0, r, x_test, y_test);

    var m = tm::TM { .. };
    tm::init(m, N_FEATURES, N_CLAUSES, N_STATES, THRESHOLD, SPECIFICITY, r);

    for epoch in 1..EPOCHS + 1 {
        for i in N_TRAIN {
            tm::update(m, x_train[i * N_FEATURES..(i + 1) * N_FEATURES], y_train[i], r);
        }
        if epoch % 10 == 0 || epoch == 1 {
            print("epoch ", epoch, ": train ", accuracy(m, x_train, y_train), " test ", accuracy(m, x_test, y_test));
        }
    }

    print("clauses (even = votes for 1, odd = votes against):");
    for j in N_CLAUSES {
        var s: u8[>..] = [];
        tm::format_clause(s, m, j);
        print("  ", j, if j % 2 == 0 { " +  " } else { " -  " }, s);
    }
}
