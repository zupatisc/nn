# Program for compiling C programs; default CC
CC=clang
# Extra flags to give to the C compiler
CFLAGS=-std=c17 -pedantic -Wall -Wno-deprecated-declarations
CFLAGS:=-fsanitize=address -fno-omit-frame-pointer -g $(CFLAGS)
# Extra flags to give to the C preprocessor
# CPPFLAGS=
# Extra flags to give to compilers when they are supposed to invoke the linker
LDFLAGS=-lm -Wall -pedantic -fsanitize=address -fno-omit-frame-pointer -g

SRC_DIR := ./src
OBJ_DIR := ./obj

SOURCES := $(shell find $(SRC_DIR) -name '*.c')

OBJECTS := $(SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
OBJECTS := $(subst tests/,,$(OBJECTS))
DEPS := $(OBJECTS:.o=.d)

# Every folder in ./src will need to be passed to GCC so that it can find header files
#INC_DIRS := $(shell find $(SRC_DIR) -type d)
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
#INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.
#CPPFLAGS := $(INC_FLAGS) -MMD -MP


# TARGET_BINARIES := test_tensor test_layer_dense

# Dependencies for the main application
deps_nn=obj/main.o obj/layer_dense.o obj/tensor.o obj/optimizer_sgd.o obj/loss_mse.o obj/activation_tanh.o obj/utils.o

# Dependencies for the tests
deps_test_tensor=obj/test_tensor.o obj/tensor.o
deps_test_layer_dense=obj/test_layer_dense.o obj/layer_dense.o obj/tensor.o
deps_test_activation_tanh=obj/test_activation_tanh.o obj/tensor.o obj/activation_tanh.o
deps_test_network=obj/test_network.o obj/layer_dense.o obj/activation_tanh.o obj/tensor.o obj/loss_mse.o
deps_test_loss_mse=obj/test_loss_mse.o obj/loss_mse.o obj/tensor.o
deps_test_optimizer_sgd=obj/test_optimizer_sgd.o obj/tensor.o obj/layer_dense.o obj/optimizer_sgd.o
