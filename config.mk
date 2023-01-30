# Program for compiling C programs; default CC
CC=clang
# Extra flags to give to the C compiler
CFLAGS=-std=c17 -pedantic -Wall -Wno-deprecated-declarations
# Extra flags to give to the C preprocessor
CPPFLAGS=
# Extra flags to give to compilers when they are supposed to invoke the linker
LDFLAGS=-lm -Wall -pedantic

