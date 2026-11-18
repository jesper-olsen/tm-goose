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
./mnist_tm --help    # all options: --clauses --T --s --states --boost --shift --train --seed ...
```

`make run DATA=MNIST ARGS="--epochs 20"` does the same through make. After every epoch the driver
prints the test accuracy, the training accuracy and the time spent.

A long run can be stopped and resumed:

```
./mnist_tm --data-dir MNIST --shift 1 --epochs 100 --save run.tm               # checkpoint after every epoch
./mnist_tm --data-dir MNIST --shift 1 --epochs 100 --load run.tm --save run.tm # continue from the last one
./mnist_tm --data-dir MNIST --shift 1 --load run.tm --epochs 0                 # only evaluate it
```

`--epochs` is the total. A resumed run continues exactly as if it had not stopped, as long as it is given
the same `--data-dir`, `--shift` and `--train`; the machine, its parameters and the random generator come from
the checkpoint, so `--clauses`, `--T`, `--s`, `--boost` and `--seed` are not accepted together with `--load`.

## Results

`--shift 1 --epochs 100`, 2000 clauses per class, T 50, s 10, 127 states, seed 1, on a MacBook Air (Apple silicon), one run each:

| | test accuracy | training accuracy | time |
|---|---|---|---|
| `--boost 0` | 98.56% | 99.12% | 1 h 23 min |
| `--boost 1` | 98.66% | 99.45% | 1 h 11 min |

## Files

| File | |
|---|---|
| `tm.goose` | The machine: packed bit inputs, per-clause include masks, multiclass update and predict |
| `mnist.goose` | IDX loader, a port of [mnist-rs](https://github.com/jesper-olsen/mnist-rs), with shift augmentation |
| `tm_io.goose` | Checkpoints: save and load a machine and its random generator |
| `mnist_tm.goose` | The MNIST driver |
| `xor.goose`, `multi.goose` | Toy problems: noisy XOR with one binary machine, and a four-class problem |
| `test_core.goose` | Unit tests of the feedback rules, and statistical tests of the sampling |
| `test_io.goose` | Tests of the checkpoints: round trips, exact resume, every kind of damaged file |
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
- **Boost** (`--boost 1`, off by default) is the "Boosting of True Positive Feedback" of Section 3.3.3 of
  Granmo's paper (arXiv:1804.01508), `boost_true_positive_feedback` in pyTsetlinMachine, where it is on by
  default: a true literal of a clause that fired always steps towards include, instead of doing so with
  probability (s-1)/s. Nothing else changes, and without it the behaviour and the random stream are exactly as
  before. On MNIST it trains faster and, in the runs below, about as accurately; on a small noisy toy problem it
  lowered the accuracy, in this implementation and in pyTsetlinMachine alike. Try both.
- **Checkpoints** (`tm_io.goose`) hold the automaton states, the parameters, the epoch counter and the state of
  the random generator, which is a single word, so resuming is exact. A checkpoint is `56 + one byte per
  automaton` bytes (31 MB for 2000 clauses per class on MNIST), little-endian, and the format is described at
  the top of the file. A load checks the whole file first and leaves the machine untouched if anything is wrong;
  a save is atomic, so an interrupted run keeps its previous checkpoint.
- **`--shift N`** augments the training set with every shift of up to N pixels on both axes, the original
  included, so it grows `(2N+1)^2` times and each image is followed by its copies. The test set is not
  touched. The training accuracy is always measured on the first 10000 unshifted training images.

## Testing

`make test` checks the clause evaluation against its definition, both feedback types, the multiclass
update, and the sampling of the decay against the binomial distribution it should follow: frequency per
position, mean and variance of the number of moves, and independence between literals.
The checkpoint tests cover round trips, resuming exactly, and each way a file can be wrong (a missing or
foreign file, an unknown version, a wrong size, header values out of range, automaton states out of range).
`make compare` trains `tm.goose` and `tm_exact.goose` on the same data over 16 seeds and prints the paired
difference in test accuracy.

## Notes

- Results are not bit-identical to other implementations: the random streams differ.
- Checkpoints hold the machine, not the data options: resume with the same `--data-dir`, `--shift` and `--train`.
- `--shift 3` or more on the full training set needs a larger data stack than Goose reserves by default:
  `make GOOSEFLAGS="--stack-reserve 16G"`.
