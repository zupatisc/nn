# Project description
This is a small neural network "framework" implemented in pure C created as an assignment.

Implemented are:
- Dense layer
- Tanh and ReLU activation function
- MSE Loss function
- SGD optimizer

All these implementations use a custom Tensor object that handles all needed tensor operations like the matrix dot product, elementwise operations like addition, multiplication and raising by a power, and the matrix transpose. In addition there are convenience function like printing tensors, reading and writing tensors to and from CSV files as well as creating tensors with equal dimensions to an existing tensor and comparing two tensors on equality elementwise.

The main binary `nn` implements two networks, one that is trained to emulate the `sin()` function and another that is trained to emulate the `xor()` function.

# Building from source
## Dependencies
Generally a Linux system is assumed
- GNU Make
- Clang
- Python3
## Compiling
After setting the current working directory to the root directory of the project you may run `make` to compile the main binary `nn` which will be output to the `bin` folder. Running `make tests` will compile __and__ run all available tests. Since some changes in headers may not be directly picked up it may be neccessary to run `make clean` to delete any already build and linked objects in the `bin` or `obj` folders.

# Running
After the compilation any binary in `bin` may be run. The main `nn` binary will create both the synthetic data points the networks are trained on, as well as output the predictions of the networks after they are finished training to CSV files.
The included python script will spawn a plot of the test data for the sinus network.
