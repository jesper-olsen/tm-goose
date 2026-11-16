// A four-class toy problem for the multiclass machine: 12 Boolean inputs of which only
// the first four matter, and class = 2 * (x0 XOR x1) + (x2 XOR x3). Training labels are
// replaced by a random other class with probability NOISE; the test set is noise free.
import tm;

const N_FEATURES = 12;
const N_WORDS = (N_FEATURES + 63) / 64;     // packed u64 words per example
const N_CLASSES = 4;
const N_TRAIN = 5000;
const N_TEST = 5000;
const NOISE = 0.2;
const EPOCHS = 100;

const N_CLAUSES = 20;                       // per class
const N_STATES = 100;
const THRESHOLD = 15;
const SPECIFICITY = 3.9;

// Appends n examples to xs (N_WORDS packed words each) and ys.
fn make_data(n: i64, noise: f64, r: rng&, xs: u64[>..]&, ys: u8[>..]&) {
    var bits: u8[N_FEATURES] = [0; N_FEATURES];
    for _ in n {
        for k in N_FEATURES { bits[k] = r.rand_int(2) as u8; }
        var y: i64 = 2 * (bits[0] ^ bits[1]) + (bits[2] ^ bits[3]);
        if r.rand_flt() < noise { y = (y + 1 + r.rand_int(N_CLASSES - 1)) % N_CLASSES; }
        ys.push(y as u8);
        tm::pack(bits, xs);
    }
}

fn accuracy(m: tm::TM&, xs: u64[:], ys: u8[:]) -> f64 {
    var correct = 0;
    for i in ys.len {
        if tm::predict(m, xs[i * N_WORDS..(i + 1) * N_WORDS]) == ys[i] { correct++; }
    }
    return correct as f64 / ys.len;
}

fn main() {
    var r = rng { 42 };
    var x_train: u64[>..] = [];
    var y_train: u8[>..] = [];
    var x_test: u64[>..] = [];
    var y_test: u8[>..] = [];
    make_data(N_TRAIN, NOISE, r, x_train, y_train);
    make_data(N_TEST, 0.0, r, x_test, y_test);

    var m = tm::TM { .. };
    tm::init(m, N_FEATURES, N_CLASSES, N_CLAUSES, N_STATES, THRESHOLD, SPECIFICITY, r);

    for epoch in 1..EPOCHS + 1 {
        for i in N_TRAIN {
            tm::update(m, x_train[i * N_WORDS..(i + 1) * N_WORDS], y_train[i], r);
        }
        if epoch % 10 == 0 || epoch == 1 {
            print("epoch ", epoch, ": train ", accuracy(m, x_train, y_train), " test ", accuracy(m, x_test, y_test));
        }
    }
}
