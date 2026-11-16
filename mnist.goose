// MNIST IDX loader for Goose - a port of jesper-olsen/mnist-rs.
//
// Reads the four uncompressed IDX files of the MNIST (or Fashion-MNIST) dataset:
//   train-images-idx3-ubyte  train-labels-idx1-ubyte
//   t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte
//
//   import mnist;
//
//   var data = mnist::Mnist { .. };
//   let err = mnist::load("MNIST", data);
//   if !mnist::is_ok(err) { abort(str("mnist: ", err)); }
//   print(data.train_images[0]);           // ASCII art
//
// Differences from the Rust crate, all forced by Goose's memory model:
//   * `Mnist` is a `root struct` (several independently growing arrays), so it
//     must live in a variable and cannot be returned: `load` fills one in.
//   * There is no Result type: loaders return `<data>, MnistError`, where the
//     `Ok` variant means success.
//   * A file whose header promises more data than it holds is a `Truncated`
//     error, and mismatched image/label counts are a `CountMismatch` error,
//     where the Rust crate gives an Io error and an assert respectively.
namespace mnist;

import std;
import os;
import binary;

const NUM_LABELS = 10;
const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const NPIXELS = 784;                       // IMAGE_WIDTH * IMAGE_HEIGHT

const LABELS_MAGIC = 2049;
const IMAGES_MAGIC = 2051;

// A single 28x28 image, row-major.
struct Image { pixels: u8[NPIXELS] }

// ---------------------------------------------------------------- errors

enum MnistError {
    Ok,
    // The file could not be read (missing, permission denied, ...).
    Io,
    // The magic number was wrong: a corrupt file or the wrong file type.
    InvalidMagicNumber { expected: u32, found: u32 },
    // The header's image dimensions are not 28x28.
    InvalidImageDimensions { expected_width: u32, expected_height: u32,
                             found_width: u32, found_height: u32 },
    // The file is shorter than its header says (sizes in bytes).
    Truncated { expected: i64, found: i64 },
    // The image and label files of a split hold different numbers of items.
    CountMismatch { images: i64, labels: i64 },
}

fn is_ok(e: MnistError) -> bool {
    match e { Ok => true, _ => false }
}

fn format(out: u8[>..]&, e: MnistError) {
    match e {
        Ok => { format(out, "ok"); },
        Io => { format(out, "I/O error: cannot read file"); },
        InvalidMagicNumber m => {
            format(out, "Invalid magic number. Expected ", m.expected, ", but found ", m.found);
        },
        InvalidImageDimensions d => {
            format(out, "Invalid image dimensions. Expected ", d.expected_width, "x", d.expected_height,
                   ", but found ", d.found_width, "x", d.found_height);
        },
        Truncated t => {
            format(out, "Truncated file. Expected at least ", t.expected, " bytes, but found ", t.found);
        },
        CountMismatch c => {
            format(out, "Count mismatch: ", c.images, " images but ", c.labels, " labels");
        },
    }
}

// ----------------------------------------------------------------- Image

// ' ', '.', ':', '*', '@' by brightness; one text row per image row.
fn format(out: u8[>..]&, img: Image&) {
    let gradient = " .:*@";
    for i in NPIXELS {
        out.push(gradient[img.pixels[i] as i64 * 5 / 256]);
        if (i + 1) % IMAGE_WIDTH == 0 { out.push('\n'); }
    }
}

// Creates an image from normalized values in 0.0..1.0 (values outside are clamped).
fn from_f64_array(fa: f64[:]) -> Image {
    assert(fa.len == NPIXELS);
    var img = Image { pixels: [0; NPIXELS] };
    for i in NPIXELS { img.pixels[i] = clamp(fa[i] * 255.0, 0.0, 255.0) as! u8; }
    return img;
}

// Pixels normalized to 0.0..1.0.
fn as_f32_array(img: Image&) -> f32[NPIXELS] {
    var out: f32[NPIXELS] = [0.0; NPIXELS];
    for i in NPIXELS { out[i] = img.pixels[i] / 255.0; }
    return out;
}

fn as_f64_array(img: Image&) -> f64[NPIXELS] {
    var out: f64[NPIXELS] = [0.0; NPIXELS];
    for i in NPIXELS { out[i] = img.pixels[i] / 255.0; }
    return out;
}

// The image shifted by (shift_x, shift_y), i.e. result(x, y) = img(x + shift_x,
// y + shift_y): a positive shift_x moves the content LEFT and a positive
// shift_y moves it UP (the Rust crate's code does the same, though its doc
// comment says right/down). Pixels shifted out of frame are dropped and vacated
// pixels are 0 (black). A shift of a whole image width or more gives a black image.
fn shifted(img: Image&, shift_x: i64, shift_y: i64) -> Image {
    var out = Image { pixels: [0; NPIXELS] };
    let y_start = max(0, -shift_y);
    let y_end = min(IMAGE_HEIGHT, IMAGE_HEIGHT - shift_y);
    let x_start = max(0, -shift_x);
    let x_end = min(IMAGE_WIDTH, IMAGE_WIDTH - shift_x);
    let copy_width = x_end - x_start;
    if copy_width > 0 {
        for y in y_start..y_end {
            let dst = y * IMAGE_WIDTH + x_start;
            let src = (y + shift_y) * IMAGE_WIDTH + x_start + shift_x;
            copy_into(out.pixels[dst..dst + copy_width], img.pixels[src..src + copy_width]);
        }
    }
    return out;
}

// Shifted randomly by up to `max_shift` pixels in each direction.
fn random_shift(img: Image&, r: rng&, max_shift: i64) -> Image {
    let shift_x = r.rand_int(2 * max_shift + 1) - max_shift;
    let shift_y = r.rand_int(2 * max_shift + 1) - max_shift;
    return shifted(img, shift_x, shift_y);
}

// Every shift in -max_shift..=max_shift on both axes, y outermost, including the
// unshifted original: (2 * max_shift + 1)^2 images.
fn all_shifts(img: Image&, max_shift: i64) -> Image[>..] {
    var out: Image[>..] = [];
    for shift_y in -max_shift..max_shift + 1 {
        for shift_x in -max_shift..max_shift + 1 {
            out.push(shifted(img, shift_x, shift_y));
        }
    }
    return out;
}

// --------------------------------------------------------------- loading

// Labels of an IDX1 file, one byte each.
fn read_labels(path: const u8[:]) -> u8[>..], MnistError {
    var data: u8[>..] = [];
    guard read_file(path, data) else { return [], MnistError.Io; }

    let magic, magic_ok = binary::read_u32be(data, 0);
    guard magic_ok else { return [], MnistError.Truncated { 4, data.len }; }
    guard magic == LABELS_MAGIC else {
        return [], MnistError.InvalidMagicNumber { LABELS_MAGIC, magic };
    }
    let count, count_ok = binary::read_u32be(data, 4);
    guard count_ok else { return [], MnistError.Truncated { 8, data.len }; }

    let n = count as i64;
    guard binary::range_ok(data, 8, n) else {
        return [], MnistError.Truncated { 8 + n, data.len };
    }
    var labels: u8[>..] = [];
    labels.append(data[8..8 + n]);
    return labels, MnistError.Ok;
}

// Images of an IDX3 file, 28x28 pixels each.
fn read_images(path: const u8[:]) -> Image[>..], MnistError {
    var data: u8[>..] = [];
    guard read_file(path, data) else { return [], MnistError.Io; }

    let magic, magic_ok = binary::read_u32be(data, 0);
    guard magic_ok else { return [], MnistError.Truncated { 4, data.len }; }
    guard magic == IMAGES_MAGIC else {
        return [], MnistError.InvalidMagicNumber { IMAGES_MAGIC, magic };
    }
    let count, count_ok = binary::read_u32be(data, 4);
    let rows, rows_ok = binary::read_u32be(data, 8);
    let cols, cols_ok = binary::read_u32be(data, 12);
    guard count_ok && rows_ok && cols_ok else { return [], MnistError.Truncated { 16, data.len }; }
    guard rows == IMAGE_HEIGHT && cols == IMAGE_WIDTH else {
        return [], MnistError.InvalidImageDimensions { IMAGE_WIDTH, IMAGE_HEIGHT, cols, rows };
    }

    let n = count as i64;
    guard binary::range_ok(data, 16, n * NPIXELS) else {
        return [], MnistError.Truncated { 16 + n * NPIXELS, data.len };
    }
    var images: Image[>..] = [];
    for i in n {
        let at = 16 + i * NPIXELS;
        let img .= images.push(Image { pixels: [0; NPIXELS] });
        copy_into(img.pixels, data[at..at + NPIXELS]);
    }
    return images, MnistError.Ok;
}

// The four arrays of the dataset. A root struct: keep it in a variable and let
// `load` fill it.
root struct Mnist {
    train_images: Image[>..] = [],
    train_labels: u8[>..] = [],
    test_images: Image[>..] = [],
    test_labels: u8[>..] = [],
}

fn join_path(dir: const u8[:], name: const u8[:]) -> u8[>..] {
    var p: u8[>..] = [];
    p.append(dir);
    if dir.len > 0 && dir[dir.len - 1] != '/' && dir[dir.len - 1] != '\\' { p.push('/'); }
    p.append(name);
    return p;
}

// Loads the four files from `dir` into `m` (which should be empty). On an error
// `m` may be partly filled and should be discarded.
fn load(dir: const u8[:], m: Mnist&) -> MnistError {
    var err = MnistError.Ok;

    m.train_labels, err = read_labels(join_path(dir, "train-labels-idx1-ubyte"));
    guard is_ok(err) else { return err; }
    m.train_images, err = read_images(join_path(dir, "train-images-idx3-ubyte"));
    guard is_ok(err) else { return err; }
    m.test_labels, err = read_labels(join_path(dir, "t10k-labels-idx1-ubyte"));
    guard is_ok(err) else { return err; }
    m.test_images, err = read_images(join_path(dir, "t10k-images-idx3-ubyte"));
    guard is_ok(err) else { return err; }

    if m.train_images.len != m.train_labels.len {
        return MnistError.CountMismatch { m.train_images.len, m.train_labels.len };
    }
    if m.test_images.len != m.test_labels.len {
        return MnistError.CountMismatch { m.test_images.len, m.test_labels.len };
    }
    return MnistError.Ok;
}

// Replaces the training set by every shift variant (-max_shift..=max_shift on
// both axes, including the original) of each training image; each image is
// followed by its variants, all carrying its label. The size grows by
// (2 * max_shift + 1)^2. The test set is left alone: augmenting evaluation
// data would leak correlated variants across the train/test boundary.
fn augment_train_with_shifts(m: Mnist&, max_shift: i64) {
    assert(max_shift >= 0);
    let images = copy(m.train_images);
    let labels = copy(m.train_labels);
    m.train_images.clear();
    m.train_labels.clear();
    for i in images.len {
        for shift_y in -max_shift..max_shift + 1 {
            for shift_x in -max_shift..max_shift + 1 {
                m.train_images.push(shifted(images[i], shift_x, shift_y));
                m.train_labels.push(labels[i]);
            }
        }
    }
}

// `load`, then `augment_train_with_shifts`.
fn load_with_shift_augmentation(dir: const u8[:], max_shift: i64, m: Mnist&) -> MnistError {
    let err = load(dir, m);
    guard is_ok(err) else { return err; }
    augment_train_with_shifts(m, max_shift);
    return MnistError.Ok;
}
