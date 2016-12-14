#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <sys/time.h>
#include <valarray>
#include <string>

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <hdf5.h>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10

#define TILE_WIDTH 2

static int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
static int xdims[] = {FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS};

// Model dimensions
static int conv1dims[] = {5, 5, 1, 32};
static int conv2dims[] = {5, 5, 32, 64};
static int fc1dims[]   = {1024, 128};
static int fc2dims[]   = {128, 10};

struct dims {
	int dim[4];
};

struct dims_2 {
	int dim[2];
};

__global__ void conv_forward_valid_relu(float *X, float *W, float *Y, dims x, dims w, dims y, int W_grid) {
	int filter_h = w.dim[0];
	int filter_w = w.dim[1];
	int filter_c = w.dim[2];
	int n, m, c, p, q;
	int xoffset, woffset, yoffset;
	n = blockIdx.x;
	m = blockIdx.y;
	int y_h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
	int y_w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
	float acc = 0;

	for (c = 0; c < filter_c; c++) {
		for (p = 0; p < filter_w; p++) {
			for (q = 0; q < filter_h; q++) {
				xoffset = ((n * x.dim[1] + (y_h + p)) * x.dim[2] + (y_w + q)) * x.dim[3] + c;
				woffset = ((p * filter_w + q) * filter_c + c) * m + m;
				acc += X[xoffset] * W[woffset];
			}
		}
	}

	yoffset = ((n * y.dim[1] + y_h) * y.dim[2] + y_w) * y.dim[3] + m;
	Y[yoffset] = (acc < 0) ? 0 : acc;
}

__global__ void average_pool_kernel(float *X, float *Y, dims x, dims y, int pool_size, int W_grid) {
	int n, m, p, q;
	int xoffset, yoffset;
	n = blockIdx.x;
	m = blockIdx.y;
	int y_h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
	int y_w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
	float acc = 0;

	for (p = 0; p < pool_size; p++) {
		for (q = 0; q < pool_size; q++) {
			xoffset = ((n * x.dim[1] + (pool_size * y_h + p)) * x.dim[2] + (pool_size * y_w + q)) * x.dim[3] + m;
			acc += X[xoffset] / (1.0f * pool_size * pool_size);
		}
	}

	yoffset = ((n * y.dim[1] + y_h) * y.dim[2] + y_w) * y.dim[3] + m;
	Y[yoffset] = acc;
}

__global__ void fully_forward_kernel(float *X, float *W, float *Y, dims_2 x, dims_2 w) {
	int i = blockIdx.x;
	int j = threadIdx.x;
	float sum = 0;

	for (int k = 0; k <= x.dim[1]; k++) {
		sum += X[i * x.dim[1] + k] * W[k * w.dim[1] + j];
	}

	Y[i * w.dim[1] + j] = sum;
}

__global__ void relu2_kernel(float *X) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	X[i] = (X[i] < 0) ? 0 : X[i];
}

// Choose the guess with largest score
static void argmax(const float *X, const int xdims[2], int *Y) {
	for (const auto i : range(0, xdims[0])) {
		auto max_idx = 0;
		auto max = X[i * xdims[1]];
		for (const auto j : range(0, xdims[1])) {
			const auto elem = X[(i * xdims[1]) + j];
			if (elem > max) {
				max_idx = j;
				max = elem;
			}
		}
		Y[i] = max_idx;
	}
}

void kernel_forward(float *x, float *conv1, float *conv2, float *fc1, float *fc2, int *out) {
	int pool_size = 2;

	int adims[] = { xdims[0], (xdims[1] - conv1dims[0] + 1), (xdims[2] - conv1dims[1] + 1), conv1dims[3] };
	int bdims[] = { adims[0], adims[1] / pool_size, adims[2] / pool_size, adims[3] };
	int cdims[] = { bdims[0], (bdims[1] - conv2dims[0] + 1), (bdims[2] - conv2dims[1] + 1), conv2dims[3] };
	int ddims[] = { cdims[0], cdims[1] / pool_size, cdims[2] / pool_size, cdims[3] };
	int edims[] = { ddims[0], fc1dims[1] };
	int fdims[] = { edims[0], fc2dims[1] };
	int ddim2[] = { ddims[0], ddims[1] * ddims[2] * ddims[3] };
	float *device_a, *device_b, *device_c, *device_d, *device_e, *device_f;
	float *device_x;
	float *device_w_1;
	float *device_w_2;
	float *device_fc1;
	float *device_fc2;

	dims a_d, b_d, c_d, d_d, x_d, w_1, w_2;
	for (int i = 0; i < 4; i++) {
		a_d.dim[i] = adims[i];
		b_d.dim[i] = bdims[i];
		c_d.dim[i] = cdims[i];
		d_d.dim[i] = ddims[i];
		x_d.dim[i] = xdims[i];
		w_1.dim[i] = conv1dims[i];
		w_2.dim[i] = conv2dims[i];
	}

	dims_2 e_d, d_2, fc1_d, fc2_d;
	for (int i = 0; i < 2; i++) {
		e_d.dim[i] = edims[i];
		d_2.dim[i] = ddim2[i];
		fc1_d.dim[i] = fc1dims[i];
		fc2_d.dim[i] = fc2dims[i];
	}

	int size_x = sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3];
	int size_a = sizeof(float) * adims[0] * adims[1] * adims[2] * adims[3];
	int size_b = sizeof(float) * bdims[0] * bdims[1] * bdims[2] * bdims[3];
	int size_c = sizeof(float) * cdims[0] * cdims[1] * cdims[2] * cdims[3];
	int size_d = sizeof(float) * ddims[0] * ddims[1] * ddims[2] * ddims[3];
	int size_e = sizeof(float) * edims[0] * edims[1];
	int size_f = sizeof(float) * fdims[0] * fdims[1];
	int size_w_1 = sizeof(float) * conv1dims[0] * conv1dims[1] * conv1dims[2] * conv1dims[3];
	int size_w_2 = sizeof(float) * conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3];
	int size_fc1 = sizeof(float) * fc1dims[0] * fc1dims[1];
	int size_fc2 = sizeof(float) * fc2dims[0] * fc2dims[1];

	int w_grid_a = adims[2] / TILE_WIDTH;
	int h_grid_a = adims[1] / TILE_WIDTH;
	int z_a = w_grid_a * h_grid_a;

	int w_grid_b = bdims[2] / TILE_WIDTH;
	int h_grid_b = bdims[1] / TILE_WIDTH;
	int z_b = w_grid_b * h_grid_b;

	int w_grid_c = cdims[2] / TILE_WIDTH;
	int h_grid_c = cdims[1] / TILE_WIDTH;
	int z_c = w_grid_c * h_grid_c;

	int w_grid_d = ddims[2] / TILE_WIDTH;
	int h_grid_d = ddims[1] / TILE_WIDTH;
	int z_d = w_grid_d * h_grid_d;

	float *f;
	f = (float *)malloc(size_f);

	cudaMalloc((void **)&device_x, size_x);
	cudaMalloc((void **)&device_a, size_a);
	cudaMalloc((void **)&device_b, size_b);
	cudaMalloc((void **)&device_c, size_c);
	cudaMalloc((void **)&device_d, size_d);
	cudaMalloc((void **)&device_e, size_e);
	cudaMalloc((void **)&device_f, size_f);
	cudaMalloc((void **)&device_w_1, size_w_1);
	cudaMalloc((void **)&device_w_2, size_w_2);
	cudaMalloc((void **)&device_fc1, size_fc1);
	cudaMalloc((void **)&device_fc2, size_fc2);

	cudaMemcpy(device_x, x, size_x, cudaMemcpyHostToDevice);
	cudaMemcpy(device_w_1, conv1, size_w_1, cudaMemcpyHostToDevice);
	cudaMemcpy(device_w_2, conv2, size_w_2, cudaMemcpyHostToDevice);
	cudaMemcpy(device_fc1, fc1, size_fc1, cudaMemcpyHostToDevice);
	cudaMemcpy(device_fc2, fc2, size_fc2, cudaMemcpyHostToDevice);

	dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 gridDim_a(adims[0], adims[3], z_a);
	dim3 gridDim_b(bdims[0], bdims[3], z_b);
	dim3 gridDim_c(cdims[0], cdims[3], z_c);
	dim3 gridDim_d(ddims[0], ddims[3], z_d);

	conv_forward_valid_relu <<<gridDim_a, blockDim>>> (device_x, device_w_1, device_a, x_d, w_1, a_d, w_grid_a);
	average_pool_kernel <<<gridDim_b, blockDim>>> (device_a, device_b, a_d, b_d, pool_size, w_grid_b);
	conv_forward_valid_relu <<<gridDim_c, blockDim>>> (device_b, device_w_2, device_c, b_d, w_2, c_d, w_grid_c);
	average_pool_kernel <<<gridDim_d, blockDim>>> (device_c, device_d, c_d, d_d, pool_size, w_grid_d);
	fully_forward_kernel <<<d_2.dim[0], fc1_d.dim[1]>>> (device_d, device_fc1, device_e, d_2, fc1_d);
	relu2_kernel <<<e_d.dim[0], e_d.dim[1]>>> (device_e);
	fully_forward_kernel <<<e_d.dim[0], fc2_d.dim[1]>>> (device_e, device_fc2, device_f, e_d, fc2_d);

	cudaDeviceSynchronize();

	cudaMemcpy(f, device_f, size_f, cudaMemcpyDeviceToHost);

	cudaFree(device_x);
	cudaFree(device_w_1);
	cudaFree(device_w_2);
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);
	cudaFree(device_d);
	cudaFree(device_e);
	cudaFree(device_f);
	cudaFree(device_fc1);
	cudaFree(device_fc2);

	argmax(f, fdims, out);
}

static int loadData(float *x, float *y) {
  // Open the data file
  const auto file_id =
      H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset x and y
  const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
  const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

  // Get the dataset x dimensions
  const auto xspace = H5Dget_space(x_id);
  const auto xndims = H5Sget_simple_extent_ndims(xspace);
  assert(xndims == 4);

  hsize_t input_dims[xndims];
  H5Sget_simple_extent_dims(xspace, input_dims, NULL);
  if (input_dims[0] != FLAGS_batch_size) {
    std::cout << "data size does not match batch size specified!\n";
    return 1; // return error
  }
  std::cout << "input dimensions = " << input_dims[0] << " x " << input_dims[1]
            << " x " << input_dims[2] << " x " << input_dims[3] << "\n";

  // Read the dataset x and y
  check_success(
      H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
  check_success(
      H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

  // Close the dataset x and y
  check_success(H5Dclose(x_id));
  check_success(H5Dclose(y_id));

  // Close the file
  check_success(H5Fclose(file_id));

  // return success
  return 0;
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
  // Open the model file
  const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset
  const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
  const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
  const auto fc1_id   = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
  const auto fc2_id   = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

  // Read the dataset
  check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv1));
  check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv2));
  check_success(
      H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
  check_success(
      H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

  // Close the dataset x and y
  check_success(H5Dclose(conv1_id));
  check_success(H5Dclose(conv2_id));
  check_success(H5Dclose(fc1_id));
  check_success(H5Dclose(fc2_id));

  // Close the file
  check_success(H5Fclose(file_id));
}

// From book chapter Figure 16.4
static void conv_forward_valid(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]) {
  const auto filter_h   = wdims[0];
  const auto filter_w   = wdims[1];
  const auto in_channel = wdims[2];

  for (const auto i : range(0, ydims[0])) {
    for (const auto m : range(0, ydims[3])) {
      for (const auto w : range(0, ydims[2])) {
        for (const auto h : range(0, ydims[1])) {
          for (const auto p : range(0, filter_h)) {
            for (const auto q : range(0, filter_w)) {
              for (const auto c : range(0, in_channel)) {
                const auto yoffset =
                    ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
                const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
                                     (h + p) * xdims[2] * xdims[3] +
                                     (w + q) * xdims[3] + c;
                const auto woffset = p * wdims[1] * wdims[2] * wdims[3] +
                                     q * wdims[2] * wdims[3] + c * wdims[3] + m;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Recified linear unit 4d
static void relu4(float *X, const int xdims[4]) {
  for (const auto i : range(0, xdims[0] * xdims[1] * xdims[2] * xdims[3])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// Recified linear unit 2d
static void relu2(float *X, const int xdims[2]) {
  for (const auto i : range(0, xdims[0] * xdims[1])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// From book chapter Figure 16.5
static void average_pool(const float *X, const int xdims[4],
                         const int pool_size, float *Y, const int ydims[4]) {
  for (const auto i : range(0, ydims[0])) {
    for (const auto m : range(0, ydims[3])) {
      for (const auto w : range(0, ydims[2])) {
        for (const auto h : range(0, ydims[1])) {
          for (const auto p : range(0, pool_size)) {
            for (const auto q : range(0, pool_size)) {
              const auto yoffset =
                  ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
              const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
                                   (pool_size * h + p) * xdims[2] * xdims[3] +
                                   (pool_size * w + q) * xdims[3] + m;
              Y[yoffset] += X[xoffset] / (1.0f * pool_size * pool_size);
            }
          }
        }
      }
    }
  }
}

static void fully_forward(const float *X, const int xdims[2], float *W,
                          const int wdims[2], float *Y, const int ydims[2]) {
  for (const auto i : range(0, xdims[0])) {
    for (const auto j : range(0, wdims[1])) {
      float sum = 0;
      for (const auto k : range(0, xdims[1])) {
        sum += X[i * xdims[1] + k] * W[k * wdims[1] + j];
      }
      Y[i * wdims[1] + j] = sum;
    }
  }
}


// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {
  // conv layer
  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1),
                       (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  auto a = zeros<float>(adims);
  conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);

  /// relu layer
  relu4(a, adims);

  // average pooling
  const int pool_size = 2;
  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size,
                       adims[3]};
  auto b = zeros<float>(bdims);
  average_pool(a, adims, pool_size, b, bdims);

  // conv layer
  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1),
                       (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  auto c = zeros<float>(cdims);
  conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);

  // relu
  relu4(c, cdims);

  // average pooling
  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size,
                       cdims[3]};
  auto d = zeros<float>(ddims);
  average_pool(c, cdims, pool_size, d, ddims);

  // reshape
  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};

  // matrix multiplication
  const int edims[] = {ddims[0], fc1dims[1]};
  auto e            = zeros<float>(edims);
  fully_forward(d, ddims2, fc1, fc1dims, e, edims);

  // relu
  relu2(e, edims);

  // matrix multiplication
  const int fdims[] = {edims[0], fc2dims[1]};
  auto f            = zeros<float>(fdims);
  fully_forward(e, edims, fc2, fc2dims, f, fdims);

  argmax(f, fdims, out);

  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  delete[] e;
  delete[] f;
}

int main(int argc, char **argv) {

  if (argc != 3 && argc != 4) {
    std::cerr << "\n"
              << "This program performs the forward opertion step for "
                 "Convolutional Neural Network(CNN).  "
                 "Sample usage: \n"
              << argv[0]
              << " [../data/test10.hdf5] [../data/model.hdf5] [10]\n";
    return -1;
  }
  FLAGS_testdata = std::string(argv[1]);
  FLAGS_model    = std::string(argv[2]);
  if (argc == 3) {
    const std::map<std::string, int> default_batch_sizes{
        {"../data/test2.hdf5", 2},
        {"../data/test10.hdf5", 10},
        {"../data/test100.hdf5", 100},
        {"../data/testfull.hdf5", 10000}};
    const auto batch_size_in_map = default_batch_sizes.find(FLAGS_testdata);
    if (batch_size_in_map == default_batch_sizes.end()) {
      std::cerr << "\nERROR:: Unrecognized file " << FLAGS_testdata << " batch_size must be specified.\n";
      return -1;
    }
    FLAGS_batch_size = batch_size_in_map->second;
  } else if (argc == 4) {
    FLAGS_batch_size = atoi(argv[3]);
  }
  xdims[0] = FLAGS_batch_size;
  rdims[0] = FLAGS_batch_size;

  // Load data into x and y
  float *x = allocate<float>(xdims);
  float *y = allocate<float>(rdims);
  loadData(x, y);

  // Load model
  float *conv1 = allocate<float>(conv1dims);
  float *conv2 = allocate<float>(conv2dims);
  float *fc1   = allocate<float>(fc1dims);
  float *fc2   = allocate<float>(fc2dims);
  loadModel(conv1, conv2, fc1, fc2);

  // Perform foward opertion
  int *out = zeros<int>(FLAGS_batch_size);

  // get start time
  const auto start = now();

  kernel_forward(x, conv1, conv2, fc1, fc2, out);
  //forward_operation(x, conv1, conv2, fc1, fc2, out);

  // get end time
  const auto end = now();

  // get elapsed time in milliseconds
  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Get reference
  int *ref = zeros<int>(FLAGS_batch_size);
  argmax(y, rdims, ref);

  // Calculate correctness
  int num_correct = 0;
  for (const auto i : range(0, FLAGS_batch_size)) {
    if (out[i] == ref[i]) {
      num_correct++;
    }
  }
  std::cout << "Done with " << FLAGS_batch_size << " queries in "
           << "elapsed = " << elapsed << " milliseconds. Correctness: "
           << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";

  delete[] x;
  delete[] y;
  delete[] conv1;
  delete[] conv2;
  delete[] fc1;
  delete[] fc2;
  delete[] out;
  delete[] ref;

  return 0;
}
