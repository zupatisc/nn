include config.mk
# vpath %.o ./obj

# $@ Outputs target name
# $? Outputs all prerequisites newer than the target
# $^ Outputs all prerequisites

.DELETE_ON_ERROR:
all: bin/nn tests

# Main binary
# bin/$(TARGET_BINARIES): $(OBJECTS)
# 	$(CC) $^ -o $@ $(LDFLAGS)
bin/nn: $(deps_nn)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@


# Object files
# $(OBJECTS): $(SOURCES)
# 	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

obj/%.o : src/%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

obj/%.o : src/tests/%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@


# Tests
tests: bin/test_tensor bin/test_layer_dense bin/test_activation_tanh bin/test_network bin/test_loss_mse
	@$(foreach test,$^,./$(test);)

bin/test_tensor: $(deps_test_tensor)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_layer_dense: $(deps_test_layer_dense)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_activation_tanh: $(deps_test_activation_tanh)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_network: $(deps_test_network)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_loss_mse: $(deps_test_loss_mse)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm bin/* obj/*

memcheck: bin/test_tensor bin/test_layer_dense bin/test_activation_tanh bin/test_network
	@$(foreach test,$^,valgrind --leak-check=yes ./$(test);)

.PHONY: clean all tests print test_1

-include $(DEPS)

test_1: $(OBJECTS)

print:
	@echo $(OBJECTS)
	@echo $(SOURCES)
	@echo $(DEPS)
	@echo $(INC_DIRS)
	@echo $(INC_FLAGS)
