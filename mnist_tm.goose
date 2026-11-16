// Tsetlin machine on MNIST: thresholds the pixels to bits, trains the multiclass
// machine in tm.goose and prints the test and training accuracy after every epoch.
//
//   goose build mnist_tm.goose
//   ./mnist_tm --data-dir MNIST --clauses 2000 --T 50 --s 10 --epochs 10
//
// Run with --help for the options. Training visits the images in file order, every epoch.
import mnist;
import tm;
import os;

const N_FEATURES = mnist::NPIXELS;
const N_WORDS = (N_FEATURES + 63) / 64;      // packed u64 words per image
const N_STATES = 127;                        // automaton states per action (the maximum tm supports)
const PIXEL_THRESHOLD = 76;                  // a pixel >= 76 (i.e. > 75) is a 1

const DEFAULT_DATA_DIR = "MNIST";
const DEFAULT_CLAUSES = 2000;
const DEFAULT_T = 50;
const DEFAULT_S = 10.0;
const DEFAULT_EPOCHS = 10;
const DEFAULT_SEED = 1;
const TRAIN_ACCURACY_IMAGES = 10000;         // training accuracy is measured on at most this many images

fn flt(x: f64, digits: i64) -> u8[>..] {
    var o: u8[>..] = [];
    format_flt(o, x, digits);
    return o;
}

fn usage() {
    print("Trains a multiclass Tsetlin machine on MNIST. After every epoch it prints the test accuracy and the");
    print("training accuracy, measured on the first ", TRAIN_ACCURACY_IMAGES, " unshifted training images.");
    print("");
    print("usage: mnist_tm [options]");
    print("  --data-dir DIR  directory holding the four MNIST IDX files [", DEFAULT_DATA_DIR, "]");
    print("  --clauses N     clauses per class [", DEFAULT_CLAUSES, "]");
    print("  --T N           vote threshold T [", DEFAULT_T, "]");
    print("  --s X           specificity s, greater than 1 [", flt(DEFAULT_S, 1), "]");
    print("  --epochs N      passes over the training set [", DEFAULT_EPOCHS, "]");
    print("  --seed N        random seed [", DEFAULT_SEED, "]");
    print("  --shift N       augment the training set with every shift of up to N pixels along both");
    print("                  axes, the original included: it grows (2N+1)^2 times, each image followed");
    print("                  by its shifted copies. The test set is not touched. Shifts of 3 or more on the");
    print("                  full set need a larger data stack: goose build --stack-reserve 16G ... [0 = off]");
    print("  --train N       train on only the first N training images, counted after augmentation [all]");
    print("  -h, --help      show this text");
}

fn fail(msg: u8[:]) {
    write_stderr(str("error: ", msg, " (see --help)\n"));
    exit(1);
}

// A whole-number option value of at least `min`.
fn int_arg(name: u8[:], text: u8[:], min: i64) -> i64 {
    let n, ok = parse_int(text);
    guard ok && n >= min else { fail(str("bad value for ", name, ": ", text)); }
    return n;
}

// Appends the first n images as packed bits: pixel >= PIXEL_THRESHOLD is 1.
fn pack_images(images: mnist::Image[:], n: i64, out: u64[>..]&) {
    var bits: u8[N_FEATURES] = [0; N_FEATURES];
    for i in n {
        for k in N_FEATURES { bits[k] = if images[i].pixels[k] >= PIXEL_THRESHOLD { 1 } else { 0 }; }
        tm::pack(bits, out);
    }
}

fn accuracy(m: tm::TM&, xs: u64[:], labels: u8[:]) -> f64 {
    var correct = 0;
    for i in labels.len {
        if tm::predict(m, xs[i * N_WORDS..(i + 1) * N_WORDS]) == labels[i] { correct++; }
    }
    return correct as f64 / labels.len;
}

fn main() {
    let argv = args();
    var data_dir: const u8[:] = DEFAULT_DATA_DIR;
    var clauses = DEFAULT_CLAUSES;
    var threshold = DEFAULT_T;
    var s = DEFAULT_S;
    var epochs = DEFAULT_EPOCHS;
    var seed = DEFAULT_SEED;
    var shift = 0;
    var limit = 0;
    var i = 1;
    while i < argv.len {
        let a = argv[i];
        if a == "--help" || a == "-h" { usage(); exit(0); }
        guard i + 1 < argv.len else { fail(str("missing value for ", a)); }
        let v = argv[i + 1];
        if a == "--data-dir" { data_dir = v; }
        else if a == "--clauses" { clauses = int_arg(a, v, 1); }
        else if a == "--T" { threshold = int_arg(a, v, 1); }
        else if a == "--s" {
            let f, ok = parse_flt(v);
            guard ok && f > 1.0 else { fail(str("bad value for --s: ", v)); }
            s = f;
        }
        else if a == "--epochs" { epochs = int_arg(a, v, 1); }
        else if a == "--seed" { seed = int_arg(a, v, 1); }
        else if a == "--shift" { shift = int_arg(a, v, 0); }
        else if a == "--train" { limit = int_arg(a, v, 1); }
        else { fail(str("unknown option: ", a)); }
        i += 2;
    }

    var data = mnist::Mnist { .. };
    let err = if shift > 0 { mnist::load_with_shift_augmentation(data_dir, shift, data) }
              else { mnist::load(data_dir, data) };
    guard mnist::is_ok(err) else { fail(str("loading ", data_dir, ": ", err)); }
    let n_train = if limit > 0 && limit < data.train_images.len { limit } else { data.train_images.len };

    var train_x: u64[>..] = [];
    var test_x: u64[>..] = [];
    pack_images(data.train_images, n_train, train_x);
    pack_images(data.test_images, data.test_images.len, test_x);

    // Training accuracy is measured on unshifted images only. With --shift N each original
    // image i is followed by its (2N+1)^2 - 1 shifted copies; the unshifted one sits in the
    // middle of its group, at index i * stride + center. Only images inside the part of the
    // training set that is actually used (the first n_train) are counted.
    let stride = (2 * shift + 1) * (2 * shift + 1);
    let center = (stride - 1) / 2;
    var acc_x: u64[>..] = [];
    var acc_labels: u8[>..] = [];
    var bits: u8[N_FEATURES] = [0; N_FEATURES];
    var n_acc = 0;
    while n_acc < TRAIN_ACCURACY_IMAGES && n_acc * stride + center < n_train {
        let k = n_acc * stride + center;
        for p in N_FEATURES { bits[p] = if data.train_images[k].pixels[p] >= PIXEL_THRESHOLD { 1 } else { 0 }; }
        tm::pack(bits, acc_x);
        acc_labels.push(data.train_labels[k]);
        n_acc++;
    }

    var r = rng { seed as u64 };
    var m = tm::TM { .. };
    tm::init(m, N_FEATURES, mnist::NUM_LABELS, clauses, N_STATES, threshold, s, r);
    print("clauses/class ", clauses, ", T ", threshold, ", s ", flt(s, 1), ", states ", N_STATES,
          ", seed ", seed, ", shift ", shift, ", train ", n_train, ", test ", data.test_images.len);

    for epoch in 1..epochs + 1 {
        let t0 = clock();
        for k in n_train {
            tm::update(m, train_x[k * N_WORDS..(k + 1) * N_WORDS], data.train_labels[k], r);
        }
        let t1 = clock();
        let test_acc = accuracy(m, test_x, data.test_labels);
        let train_acc = accuracy(m, acc_x, acc_labels);
        let t2 = clock();
        print("epoch ", epoch, ": test accuracy ", flt(test_acc * 100.0, 2), "%  train accuracy ", flt(train_acc * 100.0, 2),
              "%  (train ", flt(t1 - t0, 1), " s, eval ", flt(t2 - t1, 1), " s)");
    }
}
