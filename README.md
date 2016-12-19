## Result
The final time is about 900ms.

## Implementation 
Unroll the input

Matrix Multiplication

Reunroll the output

Average Pool

Fully Connected


## CNN and MNIST

Read the book chapter and familiarize youself with the CNN algorithm.

Provided is a model that has been trained using 60,000 examples (training set images) and the provided test data is 10,000 batched queries (test set images). The expected accuracy of the CNN is `~97%` on the provided test dataset.

The data and model are in [HDF5](https://support.hdfgroup.org/HDF5/) format and we have provided the code to read the input model and the training dataset.


## System Requirements

The project requires a CUDA-supported operating system, C compiler, and the CUDA 8 Toolkit. The CUDA 8 Toolkit can be downloaded from the [CUDA Download](https://developer.nvidia.com/cuda-downloads) page. Instructions on how to install the CUDA Toolkit are available in the [Quick Start page](http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html). Installation guides and the list of supported C compilers for [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html), [Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and [OSX](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) are also found in the [CUDA Toolkit Documentation Page](http://docs.nvidia.com/cuda/index.html).

Aside from a C compiler and the CUDA 8 Toolkit, [CMake](https://cmake.org/) 3.1 or later is required to generate build scripts for your target IDE and compiler. On windows, we require Visual Studio 2015 (Service Pack 3) which you can download from the webstore. For other systems, a CUDA compatible compiler is required (e.g. for OSX the [clang compiler](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html#system-requirements) is the only one supported).

## How to Build

There are two options to build this project, the first is using the [Hunter] package manager and the other is using [Docker](https://www.docker.com/). We sugguest using CMake along with Hunter, but it is known not to work on all operating systems. In this case, we suggest that you either using Docker or install the libraries needed (mainly `HDF5`).

### Using Hunter Package Manager

By default, the compilation uses the [Hunter] --- a C package manager. This method requires that you have the CUDA toolkit installed on your machine.

Assuming that you have checked out the project into `$SRCDIR` do

~~~{.sh}
cd $SRCDIR
mkdir build
cd build
cmake $SRCDIR
~~~

This will download the required software needed for the project (see the [hunter docs][hunterdoc] for more information). You may see some warning while the system is compiling _HDF5_, which you can ignore. Once CMake has been run, a `Makefile` is generated so you can then perform `make` to buidl the project.

~~~{.sh}
make
~~~

If you do not plan on using `make`, examine the `cmake -G` option which allows you to generate XCode, Visual Studio, ... project configurations. You may also need to change the build type to enable/disable debugging and/or optimizations.

If you need to use another library, you need have to modify the [`CMakeLists.txt`](https://github.com/webgpu/ece408project/blob/master/CMakeLists.txt) and add the libraries to the `target_link_libraries` (and possibly the `include_directories`) section. Documentation on the CMake commands is found in the [documentation page][cmakedoc].



## How to Test

Test your implementation with small batch size frist to verify the correctness. You can parse the `data/test100.hdf5` into smaller chuncks using your preferred language(e.g. python). 2, 10 and 100 queries are provides in `data/test2.hdf5`, `data/test10.hdf5` and `data/test100.hdf5` in the data folder. Maker sure the data file you feed in has the same batch size as the `batch_size` you specify in the command line.

~~~{.sh}
./ece408 ../data/test10.hdf5 ../data/model.hdf5 10
~~~


### How to Time

In [`utils.hpp`][utilshpp] a function called `now()` which allows you to get the current time at a high resolution. To measure the overhead of a function `f(args...)`, the pattern to use is:

~~~{.cpp}
const auto tic = now();
f(args...);
const auto toc = now();
const auto elapsed = std::chrono::duration<double, std::milli>(toc - tic).count();;
std::cout << "Calling f(args...) took " << elapsed << "milliseconds\n";
~~~


### Range For Loops

Throughout the serial code, we use the [`range.hpp`][rangehpp] to make the code easier to understand. Essentially,


~~~{.cpp}
for (const auto ii : range(0, N)) {
    do_stuff(ii);
}
~~~

Is equivalent to

~~~{.cpp}
for (const auto ii = 0; ii < N; ii++) {
    do_stuff(ii);
}
~~~

### Checking Errors

To check for CUDA errors, specialize the `check_success` function in `utils.hpp` to also handle `cudaError_t`. For example:

~~~{.cpp}
template <>
bool check_success<cudaError_t>(const cudaError_t &err) {
  const auto res = err == cudaSuccess;
  if (res == true) {
    return res;
  }
  std::cout << "Failed in CUDA. Error = " << cudaGetErrorString(err) << std::endl;
  assert(res);
  return res;
}
~~~

`check_success` can then be used when calling CUDA functions:

~~~{.cpp}
check_success(cudaFree(deviceData));
~~~

