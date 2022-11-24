##########################################
#           Editable options             #
##########################################

# Compiler options
CXX = clang
CXXFLAGS += -std=c17 -pedantic -Wall -Wno-deprecated-declarations
LDFLAGS += -pedantic -Wall -lm
LDLIBS +=
#EXECUTABLE_NAME = $(SOURCE_FILES:%c=%)

# Folders
SRC = src
BIN = bin
OBJ = $(BIN)/obj

RM = rm -r

src = $(wildcard src/*.c)
obj = $(src:.c=.o)
bin = $(src:.c=.out) 

$(bin): $(obj)
	$(CXX) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(obj)
