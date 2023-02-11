# Program for compiling C programs; default CC
CC=clang
# Extra flags to give to the C compiler
CFLAGS=-std=c17 -pedantic -Wall -Wno-deprecated-declarations
CFLAGS:=-g $(CFLAGS)
# Extra flags to give to the C preprocessor
CPPFLAGS=
# Extra flags to give to compilers when they are supposed to invoke the linker
LDFLAGS=-lm -Wall -pedantic

test_bins=test_tensor test_layer_dense

# Dependencies for the main application
deps_nn=obj/main.o obj/layer_dense.o obj/tensor.o

# Dependencies for the tests
deps_test_tensor=obj/test_tensor.o obj/tensor.o
deps_test_layer_dense=obj/test_layer_dense.o obj/layer_dense.o obj/tensor.o

