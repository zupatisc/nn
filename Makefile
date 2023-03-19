include config.mk
# vpath %.o ./obj

# $@ Outputs target name
# $? Outputs all prerequisites newer than the target
# $^ Outputs all prerequisites

.DELETE_ON_ERROR:
all: bin/nn

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
tests: bin/test_tensor bin/test_layer_dense bin/test_activation_tanh bin/test_network bin/test_loss_mse bin/test_optimizer_sgd bin/test_activation_relu
	@$(foreach test,$^,./$(test);)

bin/test_tensor: $(deps_test_tensor)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_layer_dense: $(deps_test_layer_dense)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_activation_tanh: $(deps_test_activation_tanh)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_activation_relu: $(deps_test_activation_relu)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_network: $(deps_test_network)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_loss_mse: $(deps_test_loss_mse)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_optimizer_sgd: $(deps_test_optimizer_sgd)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm bin/* obj/*

.PHONY: clean all tests print

-include $(DEPS)

print:
	@echo $(OBJECTS)
	@echo $(SOURCES)
	@echo $(DEPS)
	@echo $(INC_DIRS)
	@echo $(INC_FLAGS)
