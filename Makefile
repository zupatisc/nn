include config.mk

# $@ Outputs target name
# $? Outputs all prerequisites newer than the target
# $^ Outputs all prerequisites

all: bin/nn tests

# Object files
obj/%.o : src/%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

obj/%.o : src/tests/%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

# obj/test_tensor.o: src/tests/test_tensor.c
# 	clang -c $^ -o $@

bin/nn: obj/main.o obj/layer_dense.o obj/tensor.o
	clang -o bin/nn obj/main.o obj/layer_dense.o obj/tensor.o

# Tests
tests: bin/test_tensor

bin/test_tensor: obj/test_tensor.o obj/tensor.o
	clang -o $@ $^

print: $(wildcard src/*.c)
	echo $?
	ls -la $?

objects: $(wildcard obj/*.o)
	echo $?

clean:
	rm bin/nn obj/main.o obj/layer_dense.o obj/tensor.o

.PHONY: clean all tests
