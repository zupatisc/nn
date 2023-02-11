include config.mk

# $@ Outputs target name
# $? Outputs all prerequisites newer than the target
# $^ Outputs all prerequisites

.DELETE_ON_ERROR:
all: bin/nn tests

# Main binary
bin/nn: $(deps_nn)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@


# Object files
obj/%.o : src/%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

obj/%.o : src/tests/%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@


# Tests
tests: bin/test_tensor bin/test_layer_dense

bin/test_tensor: $(deps_test_tensor)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_layer_dense: $(deps_test_layer_dense)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@


clean:
	rm bin/nn obj/main.o obj/layer_dense.o obj/tensor.o

.PHONY: clean all tests

print: $(wildcard src/*.c)
	echo $?
	ls -la $?

objects: $(wildcard obj/*.o)
	echo $?
