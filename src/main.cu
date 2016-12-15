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
#define BLOCK_SIZE 64

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

__global__ void conv_forward_valid_kernel(float *X, float *W, float *Y, dims x, dims w, dims y, int W_grid) {
	int filter_h = w.dim[0];
	int filter_w = w.dim[1];
	int in_channel = w.dim[2];
	int n, m, c, p, q, y_h, y_w;
	int xoffset, woffset, yoffset;
	n = blockIdx.x;
	m = blockIdx.y;
	y_h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
	y_w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
	float acc = 0;

	for (c = 0; c < in_channel; c++) {
		for (p = 0; p < filter_w; p++) {
			for (q = 0; q < filter_h; q++) {
				xoffset = ((n * x.dim[1] + (y_h + p)) * x.dim[2] + (y_w + q)) * x.dim[3] + c;
				woffset = ((p * w.dim[1] + q) * w.dim[2] + c) * w.dim[3] + m;
				acc += X[xoffset] * W[woffset];
			}
		}
	}

	yoffset = ((n * y.dim[1] + y_h) * y.dim[2] + y_w) * y.dim[3] + m;
	Y[yoffset] = (acc < 0) ? 0 : acc;
}

void conv_forward_valid_parallel(float *x, float *w, float *y, const int xdims[4], const int wdims[4], const int ydims[4]) {
	float *device_y, *device_w, *device_x;

	dims y_d, w_d, x_d;
	for (int i = 0; i < 4; i++) {
		y_d.dim[i] = ydims[i];
		x_d.dim[i] = xdims[i];
		w_d.dim[i] = wdims[i];
	}

	int size_x = sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3];
	int size_y = sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3];
	int size_w = sizeof(float) * wdims[0] * wdims[1] * wdims[2] * wdims[3];

	int w_grid = ydims[2] / TILE_WIDTH;
	int h_grid = ydims[1] / TILE_WIDTH;
	int z = w_grid * h_grid;

	cudaMalloc((void **)&device_x, size_x);
	cudaMalloc((void **)&device_y, size_y);
	cudaMalloc((void **)&device_w, size_w);

	cudaMemcpy(device_x, x, size_x, cudaMemcpyHostToDevice);
	cudaMemcpy(device_w, w, size_w, cudaMemcpyHostToDevice);

	dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 DimGrid(ydims[0], ydims[3], z);

	conv_forward_valid_kernel <<<DimGrid, DimBlock>>> (device_x, device_w, device_y, x_d, w_d, y_d, w_grid);

	cudaMemcpy(y, device_y, size_y, cudaMemcpyDeviceToHost);

	cudaFree(device_x);
	cudaFree(device_w);
	cudaFree(device_y);
}

__global__ void average_pool_kernel(float *X, float *Y, dims x, dims y, int pool_size, int W_grid) {
	int n, m, p, q, y_h, y_w;
	int xoffset, yoffset;
	n = blockIdx.x;
	m = blockIdx.y;
	y_h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
	y_w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
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

void average_pool_parallel(float *x, float *y, const int xdims[4], const int ydims[4], const int pool_size) {
	float *device_y, *device_x;

	dims y_d, x_d;
	for (int i = 0; i < 4; i++) {
		y_d.dim[i] = ydims[i];
		x_d.dim[i] = xdims[i];
	}

	int size_x = sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3];
	int size_y = sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3];

	int w_grid = ydims[2] / TILE_WIDTH;
	int h_grid = ydims[1] / TILE_WIDTH;
	int z = w_grid * h_grid;

	cudaMalloc((void **)&device_x, size_x);
	cudaMalloc((void **)&device_y, size_y);

	cudaMemcpy(device_x, x, size_x, cudaMemcpyHostToDevice);

	dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 DimGrid(ydims[0], ydims[3], z);

	average_pool_kernel <<<DimGrid, DimBlock>>> (device_x, device_y, x_d, y_d, pool_size, w_grid);

	cudaMemcpy(y, device_y, size_y, cudaMemcpyDeviceToHost);

	cudaFree(device_x);
	cudaFree(device_y);
}

__global__ void fully_forward_kernel(float *X, float *W, float *Y, int xdim, int wdim) {
	int i = blockIdx.x;
	int j = threadIdx.x;
	float sum = 0;

	for (int k = 0; k < xdim; k++) {
		sum += X[i * xdim + k] * W[k * wdim + j];
	}

	Y[i * wdim + j] = sum;
}

void fully_forward_parallel(float *x, float *w, float *y, const int xdims[2], const int wdims[2], const int ydims[2]) {
	float *device_x, *device_w, *device_y;

	int size_x = sizeof(float) * xdims[0] * xdims[1];
	int size_w = sizeof(float) * wdims[0] * wdims[1];
	int size_y = sizeof(float) * ydims[0] * ydims[1];

	cudaMalloc((void **)&device_x, size_x);
	cudaMalloc((void **)&device_w, size_w);
	cudaMalloc((void **)&device_y, size_y);

	cudaMemcpy(device_x, x, size_x, cudaMemcpyHostToDevice);
	cudaMemcpy(device_w, w, size_w, cudaMemcpyHostToDevice);

	fully_forward_kernel <<<xdims[0], wdims[1]>>> (device_x, device_w, device_y, xdims[1], wdims[1]);

	cudaMemcpy(y, device_y, size_y, cudaMemcpyDeviceToHost);

	cudaFree(device_x);
	cudaFree(device_w);
	cudaFree(device_y);
}

__global__ void unroll_x_kernel(float *X, float *X_unrolled, dims x, dims w, dims y) {
	int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q, xoffset;
	int t = blockDim.y * blockIdx.y + threadIdx.x;
	int n = blockIdx.x;
	int H_filter = w.dim[0];
	int W_filter = w.dim[1];
	int H_out = y.dim[1];
	int W_out = y.dim[2];
	int C = w.dim[2];
	int W_unroll = H_out * W_out;
	int H_unroll = C * H_filter * W_filter;

	if (t < C * W_unroll) {
		c = t / W_unroll;
		s = t % W_unroll;
		h_out = s / W_out;
		w_out = s % W_out;
		h_unroll = h_out * W_out + w_out;
		w_base = c * H_filter * W_filter;
		for (int p = 0; p < H_filter; p++) {
			for (int q = 0; q < W_filter; q++) {
				w_unroll = w_base + p * W_filter + q;
				xoffset = ((n * x.dim[1] + (h_out + p)) * x.dim[2] + (w_out + q)) * x.dim[3] + c;
				X_unrolled[(n * H_unroll + h_unroll) * W_unroll + w_unroll] = X[xoffset];
			}
		}
	}
}

void unroll_x(float *x, float *x_unroll, int xdims[4], int wdims[4], int ydims[4]) {
	float *device_x, *device_x_unroll;

	int H_out = ydims[1];
	int W_out = ydims[2];
	int C = wdims[2];

	dims y_d, w_d, x_d;
	for (int i = 0; i < 4; i++) {
		y_d.dim[i] = ydims[i];
		x_d.dim[i] = xdims[i];
		w_d.dim[i] = wdims[i];
	}

	int num_threads = C * H_out * W_out;
	int num_blocks = num_threads / 1024;
	if (num_threads % 1024 > 0) num_blocks++;

	int size_x = sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3];
	int size_x_unroll = size_x * wdims[0] * wdims[1] * wdims[2] * ydims[0] * ydims[1] * ydims[2];

	cudaMalloc((void **)&device_x, size_x);
	cudaMalloc((void **)&device_x_unroll, size_x_unroll);

	cudaMemcpy(device_x, x, size_x, cudaMemcpyHostToDevice);

	dim3 DimBlock(1024, 1, 1);
	dim3 DimGrid(xdims[0], num_blocks, 1);

	unroll_x_kernel <<<DimGrid, DimBlock>>> (device_x, device_x_unroll, x_d, w_d, y_d);

	cudaMemcpy(x_unroll, device_x_unroll, size_x_unroll, cudaMemcpyDeviceToHost);

	cudaFree(device_x);
	cudaFree(device_x_unroll);
}

__global__ void unroll_w_kernel(float *W, float *W_unroll, dims w, int size) {
	int row, col, w_unroll, idx_base, woffset;
	int t = blockDim.y * blockIdx.y + threadIdx.x;
	int H_filter = w.dim[0];
	int W_filter = w.dim[1];
	int C = w.dim[2];

	if (t < size) {
		row = t / C;
		col = t % C;
		w_unroll = H_filter * W_filter * C;
		idx_base = row * w_unroll + col * H_filter * W_filter;
		for (int p = 0; p < H_filter; p++) {
			for (int q = 0; q < W_filter; q++) {
				woffset = ((p * w.dim[1] + q) * w.dim[2] + col) * w.dim[3] + row;
				W_unroll[idx_base + p * W_filter + q] = W[woffset];
			}
		}
	}
}

void unroll_w(float *w, float *w_unroll, int wdims[4], int ydims[4]) {
	float *device_w, *device_w_unroll;

	int C = wdims[2];
	int M = ydims[3];

	dims w_d;
	for (int i = 0; i < 4; i++) {
		w_d.dim[i] = wdims[i];
	}

	int num_threads = C * M;
	int num_blocks = num_threads / BLOCK_SIZE;
	if (num_threads % BLOCK_SIZE > 0) num_blocks++;

	int size_w = sizeof(float) * wdims[0] * wdims[1] * wdims[2] * wdims[3];
	int size_w_unroll = size_w;

	cudaMalloc((void **)&device_w, size_w);
	cudaMalloc((void **)&device_w_unroll, size_w_unroll);

	cudaMemcpy(device_w, w, size_w, cudaMemcpyHostToDevice);

	dim3 DimBlock(BLOCK_SIZE, 1, 1);
	dim3 DimGrid(xdims[0], num_blocks, 1);

	unroll_w_kernel <<<DimGrid, DimBlock>>> (device_w, device_w_unroll, w_d, num_threads);

	cudaMemcpy(w_unroll, device_w_unroll, size_w_unroll, cudaMemcpyDeviceToHost);

	cudaFree(device_w);
	cudaFree(device_w_unroll);
}

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
	int numARows, int numAColumns,
	int numBRows, int numBColumns,
	int numCRows, int numCColumns) {
	//@@ Insert code to implement matrix multiplication here
	//@@ You have to use shared memory for this MP
	__shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
	__shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;
	float Cvalue = 0.0;
	int numOfTiles = numAColumns / TILE_WIDTH;
	if (numAColumns % TILE_WIDTH) numOfTiles++;

	for (int m = 0; m < numOfTiles; m++) {
		if ((m * TILE_WIDTH + tx < numAColumns) && (Row < numARows)) {
			subTileA[ty][tx] = A[Row * numAColumns + m * TILE_WIDTH + tx];
		}
		else {
			subTileA[ty][tx] = 0.0;
		}
		if ((m * TILE_WIDTH + ty < numBRows) && (Col < numBColumns)) {
			subTileB[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + Col];
		}
		else {
			subTileB[ty][tx] = 0.0;
		}

		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; k++) {
			Cvalue += subTileA[ty][k] * subTileB[k][tx];
		}
		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) {
		C[Row * numBColumns + Col] = Cvalue;
	}
}

// Choose the guess with largest score
static void argmax(const float *X, const int xdims[2], int *Y) {
  for (const auto i : range(0, xdims[0])) {
    auto max_idx = 0;
    auto max     = X[i * xdims[1]];
    for (const auto j : range(0, xdims[1])) {
      const auto elem = X[(i * xdims[1]) + j];
      if (elem > max) {
        max_idx = j;
        max     = elem;
      }
    }
    Y[i] = max_idx;
  }
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
