# tm-goose: a Tsetlin machine on MNIST, in Goose
#
#   make              build ./mnist_tm
#   make test         unit and statistical tests of the machine and of the checkpoints (a few seconds)
#   make run          train on MNIST: make run DATA=path/to/mnist ARGS="--epochs 20 --shift 1"
#   make xor multi    the two toy problems
#   make compare      the sampling in tm.goose against tm_exact.goose on the toys (a few minutes)
#   make clean
#
# GOOSE is the compiler, GOOSEFLAGS are extra flags for building mnist_tm, e.g.
#   make GOOSEFLAGS="--stack-reserve 16G"    for --shift 3 or more on the full training set

GOOSE      ?= goose
GOOSEFLAGS ?=
DATA       ?= MNIST
ARGS       ?=

.PHONY: all test run xor multi compare clean

all: mnist_tm

mnist_tm: mnist_tm.goose mnist.goose tm.goose tm_io.goose
	$(GOOSE) build $(GOOSEFLAGS) mnist_tm.goose

test:
	$(GOOSE) --debug test_core.goose
	$(GOOSE) --debug test_io.goose

run: mnist_tm
	./mnist_tm --data-dir $(DATA) $(ARGS)

xor:
	$(GOOSE) xor.goose

multi:
	$(GOOSE) multi.goose

compare: compare.goose tm.goose tm_exact.goose
	$(GOOSE) build compare.goose
	./compare

clean:
	rm -f mnist_tm compare
