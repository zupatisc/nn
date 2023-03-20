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
bin/nn: $(deps_nn) | bin
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@


# Object files
# $(OBJECTS): $(SOURCES)
# 	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

obj/%.o : src/%.c | obj
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

obj/%.o : src/tests/%.c | obj
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@


# Tests
tests: bin/test_tensor bin/test_layer_dense bin/test_activation_tanh bin/test_network bin/test_loss_mse bin/test_optimizer_sgd bin/test_activation_relu
	@$(foreach test,$^,./$(test);)

bin/test_tensor: $(deps_test_tensor) | bin
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_layer_dense: $(deps_test_layer_dense) | bin
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_activation_tanh: $(deps_test_activation_tanh) | bin
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_activation_relu: $(deps_test_activation_relu) | bin
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_network: $(deps_test_network) | bin
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_loss_mse: $(deps_test_loss_mse) | bin
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

bin/test_optimizer_sgd: $(deps_test_optimizer_sgd) | bin
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@


obj:
	mkdir -p $@

bin:
	mkdir -p $@


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
