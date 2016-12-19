#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <time.h>
#include <valarray>
#include <string>

#include <hdf5.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10

#define MAX_THREADS 1024

static int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
static int xdims[] = { FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS };
static int rdims[] = { FLAGS_batch_size, NUM_DIGITS };

// Model dimensions
static int conv1dims[] = { 5, 5, 1, 32 };
static int conv2dims[] = { 5, 5, 32, 64 };
static int fc1dims[] = { 1024, 128 };
static int fc2dims[] = { 128, 10 };

struct dims {
	int dim[4];
};

__global__ void unroll_x_kernel(float *X, float *X_unroll, dims x, dims w, dims y) {
	int xoffset, uoffset;
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int n = blockIdx.y;
	int H_filter = w.dim[0];
	int W_filter = w.dim[1];
	int H_out = y.dim[1];
	int W_out = y.dim[2];
	int C = w.dim[2];
	int W_unroll = H_out * W_out;
	int H_unroll = C * H_filter * W_filter;
	int c = index / W_unroll;
	int s = index % W_unroll;
	int h_out = s / W_out;
	int w_out = s % W_out;

	if (index < C * W_unroll) {
		for (int p = 0; p < H_filter; p++) {
			for (int q = 0; q < W_filter; q++) {
        uoffset = (n * H_unroll + (c * H_filter * W_filter + p * W_filter + q)) * W_unroll + s;
				xoffset = ((n * x.dim[1] + (h_out + p)) * x.dim[2] + (w_out + q)) * x.dim[3] + c;
				X_unroll[uoffset] = X[xoffset];
			}
		}
	}
}

__global__ void reroll_y_kernel(float *Y, float *Y_roll, dims y) {
	int yoffset, roffset;
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int n = blockIdx.y;
	int y_roll_row = index / (y.dim[1] * y.dim[2]);
	int y_roll_col = index % (y.dim[1] * y.dim[2]);
	int y_row = y_roll_col / y.dim[2];
	int y_col = y_roll_col % y.dim[2];
	int y_width = y.dim[1] * y.dim[2];
	int y_height = y.dim[3];

	if (index < y.dim[1] * y.dim[2] * y.dim[3]) {
		roffset = ((n * y.dim[1] + y_row) * y.dim[2] + y_col) * y.dim[3] + y_roll_row;
		yoffset = (n * y_height + y_roll_row) * y_width + y_roll_col;
		Y_roll[roffset] = (Y[yoffset] < 0) ? 0 : Y[yoffset];
	}
}

__global__ void average_pool_kernel(float *X, float *Y, int pool_size, dims x, dims y) {
	int xoffset, yoffset;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y;
	int h = index / (y.dim[2] * y.dim[3]);
	int w = (index % (y.dim[2] * y.dim[3])) / y.dim[3];
	int m = (index % (y.dim[2] * y.dim[3])) % y.dim[3];
	float acc = 0;
	float size = (float)(pool_size * pool_size);

	if (index < y.dim[1] * y.dim[2] * y.dim[3]) {
		for (int p = 0; p < pool_size; p++) {
			for (int q = 0; q < pool_size; q++) {
				xoffset = ((n * x.dim[1] + (pool_size * h + p)) * x.dim[2] + (pool_size * w + q)) * x.dim[3] + m;
				acc += X[xoffset];
			}
		}

    yoffset = ((n * y.dim[1] + h) * y.dim[2] + w) * y.dim[3] + m;
  	Y[yoffset] = acc / size;
  }
}

__global__ void relu_kernel(float *X, int size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < size) {
		X[i] = (X[i] < 0) ? 0 : X[i];
	}
}

void unroll_weights(const float *W, float *W_unroll, dims w) {
	int c, m, row, col;
	int unroll_offset, offset;
	int filter_h = w.dim[0];
	int filter_w = w.dim[1];
	int C = w.dim[2];
	int M = w.dim[3];
	for (row = 0; row < filter_h; row++) {
		for (col = 0; col < filter_w; col++) {
			for (c = 0; c < C; c++) {
				for (m = 0; m < M; m++) {
					unroll_offset = ((m * C + c) * filter_h + row) * filter_w + col;
					offset = ((row * filter_w + col) * C + c) * M + m;
					W_unroll[unroll_offset] = W[offset];
				}
			}
		}
	}
}

void forward_operation_parallel(const float *x, const float *w, const float *w2, float *fc1, float *fc2, float *out, const int xdims[4], const int wdims[4], const int ydims[4], const int pooldims[4], const int w2dims[4], const int y2dims[4], const int pool2dims[4], const int w3dims[2], const int y3dims[2], const int w4dims[2], const int y4dims[2], int pool_size) {
	float *device_x, *device_y, *device_pool, *device_x_unroll, *device_w_unroll, *device_y_unroll;
	float *device_y_2, *device_pool_2, *device_x_unroll_2, *device_w_unroll_2, *device_y_unroll_2;

	float *device_fc1, *device_fc2, *device_fully, *device_out;

	float alpha = 1.0f;
	float beta = 0.0f;

	int numstream = 40;

	cudaStream_t stream[numstream];
	for (int i = 0; i < numstream; i++) cudaStreamCreate(&stream[i]);

	cublasHandle_t handle[numstream];
	for (int i = 0; i < numstream; i++) cublasCreate(&handle[i]);

	dims y_d, w_d, x_d, pool_d;
	dims y_d_2, w_d_2, pool_d_2;
	for (int i = 0; i < 4; i++) {
		x_d.dim[i] = xdims[i];
		y_d.dim[i] = ydims[i];
		w_d.dim[i] = wdims[i];
		pool_d.dim[i] = pooldims[i];

		y_d_2.dim[i] = y2dims[i];
		w_d_2.dim[i] = w2dims[i];
		pool_d_2.dim[i] = pool2dims[i];
	}

	int numAColumns = wdims[0] * wdims[1] * wdims[2], numARows = ydims[3];
	int numBColumns = ydims[1] * ydims[2];

	int numAColumns2 = w2dims[0] * w2dims[1] * w2dims[2], numARows2 = y2dims[3];
	int numBColumns2 = y2dims[1] * y2dims[2];

	int numARows3 = pool2dims[0], numAColumns3 = pool2dims[1] * pool2dims[2] * pool2dims[3];
	int numBColumns3 = w3dims[1];

	int numARows4 = y3dims[0], numAColumns4 = y3dims[1];
	int numBColumns4 = w4dims[1];

	int size_x = sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3];
	int size_y = sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3];
	int size_pool = sizeof(float) * pooldims[0] * pooldims[1] * pooldims[2] * pooldims[3];
	int size_x_unroll = sizeof(float) * xdims[0] * wdims[0] * wdims[1] * wdims[2] * ydims[1] * ydims[2];
	int size_w_unroll = sizeof(float) * wdims[0] * wdims[1] * wdims[2] * ydims[3];
	int size_y_unroll = sizeof(float) * xdims[0] * ydims[1] * ydims[2] * ydims[3];

	int size_y_2 = sizeof(float) * y2dims[0] * y2dims[1] * y2dims[2] * y2dims[3];
	int size_pool_2 = sizeof(float) * pool2dims[0] * pool2dims[1] * pool2dims[2] * pool2dims[3];
	int size_x_unroll_2 = sizeof(float) * pooldims[0] * w2dims[0] * w2dims[1] * w2dims[2] * y2dims[1] * y2dims[2];
	int size_w_unroll_2 = sizeof(float) * w2dims[0] * w2dims[1] * w2dims[2] * y2dims[3];
	int size_y_unroll_2 = sizeof(float) * pooldims[0] * y2dims[1] * y2dims[2] * y2dims[3];

	int size_w_3 = sizeof(float) * w3dims[0] * w3dims[1];
	int size_y_3 = sizeof(float) * y3dims[0] * y3dims[1];

	int size_w_4 = sizeof(float) * w4dims[0] * w4dims[1];
	int size_y_4 = sizeof(float) * y4dims[0] * y4dims[1];

	cudaMalloc((void **)&device_x, size_x);
	cudaMalloc((void **)&device_y, size_y);
	cudaMalloc((void **)&device_pool, size_pool);
	cudaMalloc((void **)&device_x_unroll, size_x_unroll);
	cudaMalloc((void **)&device_w_unroll, size_w_unroll);
	cudaMalloc((void **)&device_y_unroll, size_y_unroll);

	cudaMalloc((void **)&device_y_2, size_y_2);
	cudaMalloc((void **)&device_pool_2, size_pool_2);
	cudaMalloc((void **)&device_x_unroll_2, size_x_unroll_2);
	cudaMalloc((void **)&device_w_unroll_2, size_w_unroll_2);
	cudaMalloc((void **)&device_y_unroll_2, size_y_unroll_2);

	cudaMalloc((void **)&device_fc1, size_w_3);
  cudaMalloc((void **)&device_fully, size_y_3);
	cudaMalloc((void **)&device_fc2, size_w_4);
	cudaMalloc((void **)&device_out, size_y_4);

	float * w_unroll = (float *)malloc(size_w_unroll);
	unroll_weights(w, w_unroll, w_d);

	float * w_unroll_2 = (float *)malloc(size_w_unroll_2);
	unroll_weights(w2, w_unroll_2, w_d_2);

	cudaMemcpy(device_x, x, size_x, cudaMemcpyHostToDevice);
	cudaMemcpy(device_w_unroll, w_unroll, size_w_unroll, cudaMemcpyHostToDevice);
	cudaMemcpy(device_w_unroll_2, w_unroll_2, size_w_unroll_2, cudaMemcpyHostToDevice);
	cudaMemcpy(device_fc1, fc1, size_w_3, cudaMemcpyHostToDevice);
	cudaMemcpy(device_fc2, fc2, size_w_4, cudaMemcpyHostToDevice);

	dim3 DimBlock_unroll_x(MAX_THREADS, 1, 1);
	dim3 DimGrid_unroll_x(ceil((float)(wdims[2] * ydims[1] * ydims[2]) / MAX_THREADS), xdims[0], 1);

	dim3 DimBlock_reroll_y(MAX_THREADS, 1, 1);
	dim3 DimGrid_reroll_y(ceil((float)(ydims[1] * ydims[2] * ydims[3]) / MAX_THREADS), xdims[0], 1);

	dim3 DimBlock_pool(MAX_THREADS, 1, 1);
	dim3 DimGrid_pool(ceil((float)(pooldims[1] * pooldims[2] * pooldims[3]) / MAX_THREADS), xdims[0], 1);

	dim3 DimBlock_unroll_x_2(MAX_THREADS, 1, 1);
	dim3 DimGrid_unroll_x_2(ceil((float)(w2dims[2] * y2dims[1] * y2dims[2]) / MAX_THREADS), pooldims[0], 1);

	dim3 DimBlock_reroll_y_2(MAX_THREADS, 1, 1);
	dim3 DimGrid_reroll_y_2(ceil((float)(y2dims[1] * y2dims[2] * y2dims[3]) / MAX_THREADS), pooldims[0], 1);

	dim3 DimBlock_pool_2(MAX_THREADS, 1, 1);
	dim3 DimGrid_pool_2(ceil((float)(pool2dims[1] * pool2dims[2] * pool2dims[3]) / MAX_THREADS), pooldims[0], 1);

	int x_size = wdims[0] * wdims[1] * wdims[2] * ydims[1] * ydims[2];
	int y_size = ydims[1] * ydims[2] * ydims[3];
	int x_size_2 = w2dims[0] * w2dims[1] * w2dims[2] * y2dims[1] * y2dims[2];
	int y_size_2 = y2dims[1] * y2dims[2] * y2dims[3];

	unroll_x_kernel <<<DimGrid_unroll_x, DimBlock_unroll_x>>> (device_x, device_x_unroll, x_d, w_d, y_d);
	for (int iter = 0; iter < xdims[0]; iter += numstream) {
		for (int i = 0; (i + iter < xdims[0]) && i < numstream; i++) {
			cublasSetStream(handle[i], stream[i]);
			cublasSgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N, numBColumns, numARows, numAColumns, &alpha, device_x_unroll + (i + iter) * x_size, numBColumns, device_w_unroll, numAColumns, &beta, device_y_unroll + (i + iter) * y_size, numBColumns);
		}
	}
	reroll_y_kernel <<<DimGrid_reroll_y, DimBlock_reroll_y>>> (device_y_unroll, device_y, y_d);
	average_pool_kernel <<<DimGrid_pool, DimBlock_pool>>> (device_y, device_pool, pool_size, y_d, pool_d);

	unroll_x_kernel <<<DimGrid_unroll_x_2, DimBlock_unroll_x_2>>> (device_pool, device_x_unroll_2, pool_d, w_d_2, y_d_2);
	for (int iter = 0; iter < xdims[0]; iter += numstream) {
		for (int i = 0; (i + iter < xdims[0]) && i < numstream; i++) {
			cublasSetStream(handle[i], stream[i]);
			cublasSgemm(handle[i], CUBLAS_OP_N, CUBLAS_OP_N, numBColumns2, numARows2, numAColumns2, &alpha, device_x_unroll_2 + (i + iter) * x_size_2, numBColumns2, device_w_unroll_2, numAColumns2, &beta, device_y_unroll_2 + (i + iter) * y_size_2, numBColumns2);
		}
	}
	reroll_y_kernel <<<DimGrid_reroll_y_2, DimBlock_reroll_y_2>>> (device_y_unroll_2, device_y_2, y_d_2);
	average_pool_kernel <<<DimGrid_pool_2, DimBlock_pool_2>>> (device_y_2, device_pool_2, pool_size, y_d_2, pool_d_2);

	cublasSgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, numBColumns3, numARows3, numAColumns3, &alpha, device_fc1, numBColumns3, device_pool_2, numAColumns3, &beta, device_fully, numBColumns3);
	relu_kernel <<<ceil((float)(y3dims[0] * y3dims[1]) / MAX_THREADS), MAX_THREADS>>> (device_fully, y3dims[0] * y3dims[1]);
	cublasSgemm(handle[0], CUBLAS_OP_N, CUBLAS_OP_N, numBColumns4, numARows4, numAColumns4, &alpha, device_fc2, numBColumns4, device_fully, numAColumns4, &beta, device_out, numBColumns4);

	cudaMemcpy(out, device_out, size_y_4, cudaMemcpyDeviceToHost);

	for (int i = 0; i < numstream; i++) cublasDestroy(handle[i]);

	cudaFree(device_fc1);
  cudaFree(device_fc2);
	cudaFree(device_fully);
  cudaFree(device_out);

	cudaFree(device_y_2);
	cudaFree(device_pool_2);
	cudaFree(device_y_unroll_2);
	cudaFree(device_x_unroll_2);
	cudaFree(device_w_unroll_2);

	cudaFree(device_x);
	cudaFree(device_y);
	cudaFree(device_pool);
	cudaFree(device_y_unroll);
	cudaFree(device_x_unroll);
	cudaFree(device_w_unroll);
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

	hsize_t *input_dims = allocate<hsize_t>(xndims);
	//hsize_t input_dims[xndims];
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

	delete[] input_dims;

	// return success
	return 0;
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
	// Open the model file
	const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

	// Open the dataset
	const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
	const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
	const auto fc1_id = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
	const auto fc2_id = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

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

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {
	// conv layer
	const int pool_size = 2;
  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1), (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
	const int bdims[] = {adims[0], adims[1] / pool_size, adims[2] / pool_size, adims[3]};
	const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1), (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
	const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size, cdims[3]};
	const int edims[] = {ddims[0], fc1dims[1]};
	const int fdims[] = {edims[0], fc2dims[1]};
	auto f = zeros<float>(fdims);

	forward_operation_parallel(x, conv1, conv2, fc1, fc2, f, xdims, conv1dims, adims, bdims, conv2dims, cdims, ddims, fc1dims, edims, fc2dims, fdims, pool_size);

  argmax(f, fdims, out);

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
	FLAGS_model = std::string(argv[2]);
	if (argc == 3) {
		const std::map<std::string, int> default_batch_sizes{
			{ "../data/test2.hdf5", 2 },
			{ "../data/test10.hdf5", 10 },
			{ "../data/test100.hdf5", 100 },
			{ "../data/testfull.hdf5", 10000 } };
		const auto batch_size_in_map = default_batch_sizes.find(FLAGS_testdata);
		if (batch_size_in_map == default_batch_sizes.end()) {
			std::cerr << "\nERROR:: Unrecognized file " << FLAGS_testdata << " batch_size must be specified.\n";
			return -1;
		}
		FLAGS_batch_size = batch_size_in_map->second;
	}
	else if (argc == 4) {
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
	float *fc1 = allocate<float>(fc1dims);
	float *fc2 = allocate<float>(fc2dims);
	loadModel(conv1, conv2, fc1, fc2);

	// Perform foward opertion
	int *out = zeros<int>(FLAGS_batch_size);

	// get start time
	const auto start = now();

	forward_operation(x, conv1, conv2, fc1, fc2, out);

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
