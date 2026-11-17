# tm-goose

A multiclass Tsetlin machine (Granmo, 2018) in [Goose](https://github.com/aardappel/goose),
with an MNIST loader and a training driver.

## Requirements

- The Goose compiler, and a C compiler for `goose build`.
- MNIST as four uncompressed IDX files in one directory:
  `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`.

## Quick start

```
make                 # builds ./mnist_tm
make test            # unit and statistical tests, a few seconds
./mnist_tm --data-dir MNIST --clauses 2000 --T 50 --s 10 --epochs 20 --shift 1
./mnist_tm --help    # all options
```

`make run DATA=MNIST ARGS="--epochs 20"` does the same through make. After every epoch the driver
prints the test accuracy, the training accuracy and the time spent.

## Files

| File | |
|---|---|
| `tm.goose` | The machine: packed bit inputs, per-clause include masks, multiclass update and predict |
| `mnist.goose` | IDX loader, a port of [mnist-rs](https://github.com/jesper-olsen/mnist-rs), with shift augmentation |
| `mnist_tm.goose` | The MNIST driver |
| `xor.goose`, `multi.goose` | Toy problems: noisy XOR with one binary machine, and a four-class problem |
| `test_core.goose` | Unit tests of the feedback rules, and statistical tests of the sampling |
| `tm_exact.goose`, `compare.goose` | The same machine with one random draw per literal, and a seed-by-seed comparison with `tm.goose` |

## How it works

- **Inputs** are Boolean vectors packed one bit per feature into `u64` words (`tm::pack`). MNIST pixels
  of 76 or more are 1.
- **Clauses**: each class has its own clauses, each a conjunction over the features and their negations.
  Every literal has an automaton with `2 * N` states (`u8`, so `N` is at most 127). Each clause also keeps
  a bit mask of the literals it includes, so evaluating a clause is a few word operations. Even clauses
  vote for their class and odd ones against it.
- **Classes** share one `TM` struct (Goose cannot hold several of these in an array): clause `i` of class
  `c` is clause `c * n_clauses + i`.
- **Training**: for each example the true class gets y = 1 feedback and one other class, chosen uniformly
  at random, gets y = 0 feedback. Examples are visited in file order in every epoch.
- **Prediction** is the class with the highest vote sum, clipped to `-T..T`; ties go to the lowest class.
- **Sampling**: nine in ten Type I updates on MNIST are on a clause that did not fire, where each literal
  moves with probability 1/s. For s up to 32 the coins are flipped 64 at a time from a few random words
  (the probability is rounded to a multiple of 2^-16); above that the gaps between the literals that move
  are drawn directly. `tm_exact.goose` draws once per literal and is the reference these are compared with.
- **`--shift N`** augments the training set with every shift of up to N pixels on both axes, the original
  included, so it grows `(2N+1)^2` times and each image is followed by its copies. The test set is not
  touched. The training accuracy is always measured on the first 10000 unshifted training images.

## Testing

`make test` checks the clause evaluation against its definition, both feedback types, the multiclass
update, and the sampling of the decay against the binomial distribution it should follow: frequency per
position, mean and variance of the number of moves, and independence between literals.
`make compare` trains `tm.goose` and `tm_exact.goose` on the same data over 16 seeds and prints the paired
difference in test accuracy.

## Example run

Train on the 60k MNIST training set & augment the dataset with shifted versions of all images (8 directions).

```
./mnist_tm --shift 1 --epochs 100
```

```
clauses/class 2000, T 50, s 10.0, states 127, seed 1, shift 1, train 540000, test 10000
epoch 1: test accuracy 96.44%  train accuracy 95.90%  (train 69.8 s, eval 3.3 s)
epoch 2: test accuracy 97.20%  train accuracy 96.84%  (train 57.6 s, eval 3.3 s)
epoch 3: test accuracy 97.45%  train accuracy 97.34%  (train 53.6 s, eval 3.7 s)
epoch 4: test accuracy 97.93%  train accuracy 97.93%  (train 51.2 s, eval 3.5 s)
epoch 5: test accuracy 98.05%  train accuracy 97.92%  (train 49.8 s, eval 3.4 s)
epoch 6: test accuracy 98.10%  train accuracy 98.02%  (train 48.5 s, eval 3.4 s)
epoch 7: test accuracy 98.07%  train accuracy 98.26%  (train 47.1 s, eval 3.4 s)
epoch 8: test accuracy 98.13%  train accuracy 98.10%  (train 46.5 s, eval 3.5 s)
epoch 9: test accuracy 98.23%  train accuracy 98.33%  (train 47.1 s, eval 3.8 s)
epoch 10: test accuracy 98.24%  train accuracy 98.59%  (train 48.4 s, eval 3.6 s)
epoch 11: test accuracy 98.27%  train accuracy 98.46%  (train 47.1 s, eval 3.7 s)
epoch 12: test accuracy 98.26%  train accuracy 98.59%  (train 47.3 s, eval 5.5 s)
epoch 13: test accuracy 98.36%  train accuracy 98.56%  (train 53.8 s, eval 4.0 s)
epoch 14: test accuracy 98.32%  train accuracy 98.74%  (train 51.3 s, eval 5.4 s)
epoch 15: test accuracy 98.27%  train accuracy 98.68%  (train 48.9 s, eval 3.6 s)
epoch 16: test accuracy 98.29%  train accuracy 98.66%  (train 54.8 s, eval 3.8 s)
epoch 17: test accuracy 98.47%  train accuracy 98.69%  (train 48.3 s, eval 4.9 s)
epoch 18: test accuracy 98.25%  train accuracy 98.67%  (train 50.3 s, eval 3.9 s)
epoch 19: test accuracy 98.55%  train accuracy 98.76%  (train 49.4 s, eval 5.5 s)
epoch 20: test accuracy 98.43%  train accuracy 98.81%  (train 49.8 s, eval 4.1 s)
epoch 21: test accuracy 98.42%  train accuracy 98.76%  (train 49.2 s, eval 4.5 s)
epoch 22: test accuracy 98.36%  train accuracy 98.77%  (train 47.9 s, eval 3.7 s)
epoch 23: test accuracy 98.27%  train accuracy 98.74%  (train 47.4 s, eval 3.7 s)
epoch 24: test accuracy 98.41%  train accuracy 98.85%  (train 48.4 s, eval 4.4 s)
epoch 25: test accuracy 98.34%  train accuracy 98.79%  (train 48.8 s, eval 4.1 s)
epoch 26: test accuracy 98.39%  train accuracy 98.84%  (train 46.6 s, eval 3.9 s)
epoch 27: test accuracy 98.48%  train accuracy 98.83%  (train 45.7 s, eval 4.6 s)
epoch 28: test accuracy 98.49%  train accuracy 98.86%  (train 50.0 s, eval 4.7 s)
epoch 29: test accuracy 98.52%  train accuracy 98.79%  (train 51.9 s, eval 4.0 s)
epoch 30: test accuracy 98.52%  train accuracy 98.98%  (train 47.0 s, eval 4.6 s)
epoch 31: test accuracy 98.30%  train accuracy 98.83%  (train 50.2 s, eval 3.7 s)
epoch 32: test accuracy 98.41%  train accuracy 98.93%  (train 40.8 s, eval 3.5 s)
epoch 33: test accuracy 98.46%  train accuracy 98.86%  (train 39.0 s, eval 3.3 s)
epoch 34: test accuracy 98.36%  train accuracy 98.82%  (train 38.4 s, eval 3.3 s)
epoch 35: test accuracy 98.42%  train accuracy 98.96%  (train 38.3 s, eval 3.3 s)
epoch 36: test accuracy 98.47%  train accuracy 98.93%  (train 38.2 s, eval 3.3 s)
epoch 37: test accuracy 98.42%  train accuracy 98.80%  (train 38.1 s, eval 3.3 s)
epoch 38: test accuracy 98.39%  train accuracy 98.86%  (train 38.0 s, eval 3.3 s)
epoch 39: test accuracy 98.45%  train accuracy 98.99%  (train 38.1 s, eval 3.3 s)
epoch 40: test accuracy 98.54%  train accuracy 98.97%  (train 37.9 s, eval 3.3 s)
epoch 41: test accuracy 98.50%  train accuracy 98.92%  (train 38.5 s, eval 3.3 s)
epoch 42: test accuracy 98.56%  train accuracy 98.94%  (train 46.4 s, eval 4.3 s)
epoch 43: test accuracy 98.49%  train accuracy 98.88%  (train 45.5 s, eval 3.3 s)
epoch 44: test accuracy 98.46%  train accuracy 98.92%  (train 38.3 s, eval 3.3 s)
epoch 45: test accuracy 98.45%  train accuracy 98.96%  (train 37.8 s, eval 3.4 s)
epoch 46: test accuracy 98.53%  train accuracy 99.00%  (train 37.9 s, eval 3.3 s)
epoch 47: test accuracy 98.49%  train accuracy 99.04%  (train 37.8 s, eval 3.3 s)
epoch 48: test accuracy 98.43%  train accuracy 98.94%  (train 38.2 s, eval 3.5 s)
epoch 49: test accuracy 98.44%  train accuracy 99.03%  (train 37.8 s, eval 3.3 s)
epoch 50: test accuracy 98.59%  train accuracy 98.99%  (train 37.5 s, eval 3.3 s)
epoch 51: test accuracy 98.55%  train accuracy 99.07%  (train 37.6 s, eval 3.4 s)
epoch 52: test accuracy 98.64%  train accuracy 99.13%  (train 38.2 s, eval 3.4 s)
epoch 53: test accuracy 98.63%  train accuracy 99.09%  (train 38.2 s, eval 3.3 s)
epoch 54: test accuracy 98.55%  train accuracy 99.13%  (train 37.3 s, eval 3.3 s)
epoch 55: test accuracy 98.56%  train accuracy 99.05%  (train 37.5 s, eval 3.4 s)
epoch 56: test accuracy 98.52%  train accuracy 99.02%  (train 37.5 s, eval 3.4 s)
epoch 57: test accuracy 98.58%  train accuracy 98.91%  (train 38.7 s, eval 3.7 s)
epoch 58: test accuracy 98.61%  train accuracy 98.92%  (train 39.8 s, eval 3.5 s)
epoch 59: test accuracy 98.56%  train accuracy 98.99%  (train 39.1 s, eval 3.5 s)
epoch 60: test accuracy 98.45%  train accuracy 99.01%  (train 38.5 s, eval 3.5 s)
epoch 61: test accuracy 98.43%  train accuracy 99.11%  (train 38.5 s, eval 3.6 s)
epoch 62: test accuracy 98.51%  train accuracy 99.01%  (train 38.6 s, eval 3.5 s)
epoch 63: test accuracy 98.64%  train accuracy 99.09%  (train 38.1 s, eval 3.5 s)
epoch 64: test accuracy 98.53%  train accuracy 99.07%  (train 37.8 s, eval 3.3 s)
epoch 65: test accuracy 98.58%  train accuracy 99.11%  (train 37.0 s, eval 3.5 s)
epoch 66: test accuracy 98.51%  train accuracy 99.07%  (train 39.7 s, eval 3.7 s)
epoch 67: test accuracy 98.59%  train accuracy 99.15%  (train 37.1 s, eval 3.3 s)
epoch 68: test accuracy 98.58%  train accuracy 99.17%  (train 36.8 s, eval 3.5 s)
epoch 69: test accuracy 98.61%  train accuracy 99.14%  (train 38.4 s, eval 3.3 s)
epoch 70: test accuracy 98.52%  train accuracy 99.04%  (train 37.3 s, eval 3.3 s)
epoch 71: test accuracy 98.58%  train accuracy 99.01%  (train 36.7 s, eval 3.3 s)
epoch 72: test accuracy 98.72%  train accuracy 99.06%  (train 37.3 s, eval 3.3 s)
epoch 73: test accuracy 98.61%  train accuracy 98.92%  (train 36.7 s, eval 3.3 s)
epoch 74: test accuracy 98.56%  train accuracy 99.07%  (train 36.7 s, eval 3.4 s)
epoch 75: test accuracy 98.70%  train accuracy 99.08%  (train 36.7 s, eval 3.3 s)
epoch 76: test accuracy 98.59%  train accuracy 99.08%  (train 36.6 s, eval 3.3 s)
epoch 77: test accuracy 98.54%  train accuracy 99.04%  (train 36.5 s, eval 3.3 s)
epoch 78: test accuracy 98.59%  train accuracy 99.10%  (train 36.5 s, eval 3.3 s)
epoch 79: test accuracy 98.60%  train accuracy 99.19%  (train 36.6 s, eval 3.4 s)
epoch 80: test accuracy 98.61%  train accuracy 99.11%  (train 36.6 s, eval 3.3 s)
epoch 81: test accuracy 98.58%  train accuracy 99.03%  (train 36.6 s, eval 3.3 s)
epoch 82: test accuracy 98.68%  train accuracy 99.16%  (train 36.8 s, eval 3.3 s)
epoch 83: test accuracy 98.66%  train accuracy 99.06%  (train 36.6 s, eval 3.3 s)
epoch 84: test accuracy 98.57%  train accuracy 99.10%  (train 36.8 s, eval 3.4 s)
epoch 85: test accuracy 98.56%  train accuracy 99.11%  (train 36.4 s, eval 3.3 s)
epoch 86: test accuracy 98.62%  train accuracy 99.04%  (train 36.5 s, eval 3.3 s)
epoch 87: test accuracy 98.70%  train accuracy 99.06%  (train 36.4 s, eval 3.3 s)
epoch 88: test accuracy 98.53%  train accuracy 99.04%  (train 37.8 s, eval 3.4 s)
epoch 89: test accuracy 98.56%  train accuracy 99.09%  (train 37.2 s, eval 3.4 s)
epoch 90: test accuracy 98.64%  train accuracy 99.06%  (train 36.3 s, eval 3.4 s)
epoch 91: test accuracy 98.67%  train accuracy 99.10%  (train 37.4 s, eval 3.3 s)
epoch 92: test accuracy 98.57%  train accuracy 99.11%  (train 36.4 s, eval 3.3 s)
epoch 93: test accuracy 98.60%  train accuracy 99.12%  (train 36.2 s, eval 3.3 s)
epoch 94: test accuracy 98.54%  train accuracy 99.06%  (train 36.1 s, eval 3.3 s)
epoch 95: test accuracy 98.46%  train accuracy 99.02%  (train 36.2 s, eval 3.3 s)
epoch 96: test accuracy 98.62%  train accuracy 99.12%  (train 36.2 s, eval 3.4 s)
epoch 97: test accuracy 98.43%  train accuracy 99.02%  (train 36.2 s, eval 3.3 s)
epoch 98: test accuracy 98.52%  train accuracy 99.09%  (train 36.0 s, eval 3.3 s)
epoch 99: test accuracy 98.55%  train accuracy 99.11%  (train 36.3 s, eval 3.3 s)
epoch 100: test accuracy 98.56%  train accuracy 99.12%  (train 36.5 s, eval 3.8 s)
```

## Notes

- Results are not bit-identical to other implementations: the random streams differ.
- There is no model saving and no boost option.
- `--shift 3` or more on the full training set needs a larger data stack than Goose reserves by default:
  `make GOOSEFLAGS="--stack-reserve 16G"`.
