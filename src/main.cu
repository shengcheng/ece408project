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

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10

#define TILE_WIDTH 16
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

// __global__ void unroll_x_kernel(float *X, float *X_unroll, dims x, dims w, dims y) {
// 	int c, s, h_out, w_out, h_unroll, w_unroll, xoffset, offset;
// 	int index = blockDim.x * blockIdx.x + threadIdx.x;
// 	int H_filter = w.dim[0];
// 	int W_filter = w.dim[1];
// 	int H_out = y.dim[1];
// 	int W_out = y.dim[2];
// 	int C = w.dim[2];
// 	int W_unroll = H_out * W_out;
//
// 	if (index < C * W_unroll) {
// 		c = index / W_unroll;
// 		s = index % W_unroll;
// 		h_out = s / W_out;
// 		w_out = s % W_out;
// 		for (int p = 0; p < H_filter; p++) {
// 			for (int q = 0; q < W_filter; q++) {
// 				w_unroll = s;
// 				h_unroll = c * H_filter * W_filter + p * W_filter + q;
//         offset = W_unroll * h_unroll + w_unroll;
// 				xoffset = ((h_out + p) * x.dim[2] + (w_out + q)) * x.dim[3] + c;
// 				X_unroll[offset] = X[xoffset];
// 			}
// 		}
// 	}
// }

__global__ void unroll_x_kernel(float *X, float *X_unroll, dims x, dims w, dims y) {
	int w_unroll, h_unroll, uoffset, xoffset;
	int H_filter = w.dim[0];
	int W_filter = w.dim[1];
	int H_out = y.dim[1];
	int W_out = y.dim[2];
	int W_unroll = H_out * W_out;
	int h_out = blockIdx.x;
	int w_out = blockIdx.y;
	int c = threadIdx.x;
	int p = threadIdx.y;
	int q = threadIdx.z;

	if (h_out < y.dim[1] && w_out < y.dim[2] && p < w.dim[0] && q < w.dim[1] && c < w.dim[2]) {
		w_unroll = h_out * W_out + w_out;
		h_unroll = c * H_filter * W_filter + p * W_filter + q;
		uoffset = h_unroll * W_unroll + w_unroll;
		xoffset = ((h_out + p) * x.dim[2] + (w_out + q)) * x.dim[3] + c;
		X_unroll[uoffset] = X[xoffset];
	}
}

__global__ void reroll_y_kernel(float *Y, float *Y_roll, dims y) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int y_roll_row = index / (y.dim[1] * y.dim[2]);
	int y_roll_col = index % (y.dim[1] * y.dim[2]);
	int y_row = y_roll_col / y.dim[2];
	int y_col = y_roll_col % y.dim[2];

	if (index < y.dim[1] * y.dim[2] * y.dim[3]) {
		int yroll_offset = y_row * y.dim[2] * y.dim[3] + y_col * y.dim[3] + y_roll_row;
		int y_offset = y_roll_row * y.dim[1] * y.dim[2] + y_roll_col;
		Y_roll[yroll_offset] = Y[y_offset];
	}
}

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
	int numARows, int numAColumns,
	int numBRows, int numBColumns,
	int numCRows, int numCColumns) {

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
		C[Row * numBColumns + Col] = (Cvalue < 0) ? 0 : Cvalue;
	}
}

__global__ void average_pool_kernel(float *X, float *Y, int pool_size, dims x, dims y) {
	int xoffset, yoffset;
	int n = blockIdx.x;
	int m = blockIdx.y;
	int h = threadIdx.x;
	int w = threadIdx.y;
	float acc = 0;

	for (int p = 0; p < pool_size; p++) {
		for (int q = 0; q < pool_size; q++) {
      if (n < y.dim[0] && m < y.dim[3] && w < y.dim[2] && h < y.dim[1]) {
        xoffset = ((n * x.dim[1] + (pool_size * h + p)) * x.dim[2] + (pool_size * w + q)) * x.dim[3] + m;
  			acc += X[xoffset] / (1.0f * pool_size * pool_size);
      }
		}
	}

  if (n < y.dim[0] && m < y.dim[3] && w < y.dim[2] && h < y.dim[1]) {
    yoffset = ((n * y.dim[1] + h) * y.dim[2] + w) * y.dim[3] + m;
  	Y[yoffset] = acc;
  }
}

void average_pool_parallel(const float *x, float *y, const int xdims[4], const int ydims[4], const int pool_size) {
	float *device_x, *device_y;

	dims y_d, x_d;
	for (int i = 0; i < 4; i++) {
		y_d.dim[i] = ydims[i];
		x_d.dim[i] = xdims[i];
	}

	int size_x = sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3];
	int size_y = sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3];

	cudaMalloc((void **)&device_x, size_x);
	cudaMalloc((void **)&device_y, size_y);

	cudaMemcpy(device_x, x, size_x, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

	dim3 DimGrid(ydims[0], ydims[3], 1);
	dim3 DimBlock(ydims[1], ydims[2], 1);

	average_pool_kernel <<<DimGrid, DimBlock>>> (device_x, device_y, pool_size, x_d, y_d);

  cudaDeviceSynchronize();

	cudaMemcpy(y, device_y, size_y, cudaMemcpyDeviceToHost);

	cudaFree(device_x);
	cudaFree(device_y);
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
void conv_forward_unroll(const float *x, const float *w, float *y, const int xdims[4], const int wdims[4], const int ydims[4]) {
	float *device_x, *device_y, *device_x_unroll, *device_w_unroll, *device_y_unroll;

	// std::cout<< "\nINPUT DIMENSIONS:\n";
	// std::cout<< "N: "<< xdims[0] << ", H: "<< xdims[1] << ", W: "<< xdims[2] << ", C: "<< xdims[3] << "\n";
	// std::cout<< "K1: "<< wdims[0] << ", K2: "<< wdims[1] << ", C: "<< wdims[2] << ", M: "<< wdims[3] << "\n";
	// std::cout<< "N: "<< ydims[0] << ", H_Out: "<< ydims[1] << ", W_Out: "<< ydims[2] << ", M: "<< ydims[3] << "\n\n";

	cudaStream_t stream0, stream1,stream2,stream3,stream4;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);

	dims y_d, w_d, x_d;
	for (int i = 0; i < 4; i++) {
		y_d.dim[i] = ydims[i];
		x_d.dim[i] = xdims[i];
		w_d.dim[i] = wdims[i];
	}

	int numAColumns = wdims[0] * wdims[1] * wdims[2], numARows = ydims[3];
	int numBColumns = ydims[1] * ydims[2], numBRows = wdims[0] * wdims[1] * wdims[2];
	int numCColumns = numBColumns, numCRows = numARows;

	int size_x = sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3];
	int size_y = sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3];
	int size_x_unroll = sizeof(float) * wdims[0] * wdims[1] * wdims[2] * ydims[1] * ydims[2];
	int size_w_unroll = sizeof(float) * wdims[0] * wdims[1] * wdims[2] * ydims[3];
	int size_y_unroll = sizeof(float) * ydims[1] * ydims[2] * ydims[3];

	int stripe_x = xdims[1] * xdims[2] * xdims[3];
	int stripe_y = ydims[1] * ydims[2] * ydims[3];
	int stripe_x_unroll = wdims[0] * wdims[1] * wdims[2] * ydims[1] * ydims[2];
	int stripe_y_unroll = ydims[1] * ydims[2] * ydims[3];

	cudaMalloc((void **)&device_x, size_x);
	cudaMalloc((void **)&device_y, size_y);
	cudaMalloc((void **)&device_x_unroll, size_x_unroll* xdims[0]);
	cudaMalloc((void **)&device_w_unroll, size_w_unroll);
	cudaMalloc((void **)&device_y_unroll, size_y_unroll* xdims[0]);

	float * w_unroll = (float *)malloc(size_w_unroll * sizeof(float));
	unroll_weights(w, w_unroll, w_d);

	
	
	cudaMemcpy(device_w_unroll, w_unroll, size_w_unroll, cudaMemcpyHostToDevice);

	// dim3 DimBlock_unroll_x(MAX_THREADS, 1, 1);
	// dim3 DimGrid_unroll_x(ceil((float)(wdims[2] * ydims[1] * ydims[2]) / MAX_THREADS), 1, 1);

	dim3 DimBlock_unroll_x(wdims[2], wdims[0], wdims[1]);
	dim3 DimGrid_unroll_x(ydims[1], ydims[2], 1);

	dim3 DimBlock_matmul(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 DimGrid_matmul(ceil((float)(ydims[1] * ydims[2]) / TILE_WIDTH), ceil((float)(ydims[3]) / TILE_WIDTH), 1);

	dim3 DimBlock_reroll_y(MAX_THREADS, 1, 1);
	dim3 DimGrid_reroll_y(ceil((float)(ydims[1] * ydims[2] * ydims[3]) / MAX_THREADS), 1, 1);
	
	cudaMemcpyAsync(device_x, x, stripe_x *sizeof(float), cudaMemcpyHostToDevice,stream0);
	cudaDeviceSynchronize();
	cudaMemcpyAsync(device_x + stripe_x, x + stripe_x, stripe_x *sizeof(float), cudaMemcpyHostToDevice,stream1);
	unroll_x_kernel <<<DimGrid_unroll_x, DimBlock_unroll_x,0,stream0>>> (device_x , device_x_unroll, x_d, w_d, y_d);
	cudaDeviceSynchronize();
	cudaMemcpyAsync(device_x + stripe_x *2, x + 2* stripe_x, stripe_x *sizeof(float), cudaMemcpyHostToDevice,stream2);
	unroll_x_kernel <<<DimGrid_unroll_x, DimBlock_unroll_x,0,stream1>>> (device_x +  stripe_x, device_x_unroll+ stripe_x_unroll, x_d, w_d, y_d);
	matrixMultiplyShared <<<DimGrid_matmul, DimBlock_matmul,0,stream0>>> (device_w_unroll, device_x_unroll, device_y_unroll,numARows, numAColumns,numBRows, numBColumns,numCRows, numCColumns);
	cudaDeviceSynchronize();
	cudaMemcpyAsync(device_x + stripe_x *3, x + 3* stripe_x, stripe_x *sizeof(float), cudaMemcpyHostToDevice,stream3);
	unroll_x_kernel <<<DimGrid_unroll_x, DimBlock_unroll_x,0,stream2>>> (device_x +  2 *stripe_x, device_x_unroll+ 2*stripe_x_unroll, x_d, w_d, y_d);
	matrixMultiplyShared <<<DimGrid_matmul, DimBlock_matmul,0,stream1>>> (device_w_unroll, device_x_unroll+stripe_x_unroll, device_y_unroll+stripe_y_unroll,numARows, numAColumns,numBRows, numBColumns,numCRows, numCColumns);
	reroll_y_kernel <<<DimGrid_reroll_y, DimBlock_reroll_y,0,stream0>>> (device_y_unroll , device_y , y_d);
	cudaDeviceSynchronize();



	for (int i = 0; i < xdims[0]-4; i++) {
		cudaMemcpyAsync(device_x+ (i+4) * stripe_x, x+ (i+4) * stripe_x, stripe_x *sizeof(float), cudaMemcpyHostToDevice,stream4);		
		unroll_x_kernel <<<DimGrid_unroll_x, DimBlock_unroll_x,0,stream3>>> (device_x + (i +3)* stripe_x, device_x_unroll+(i +3)* stripe_x_unroll, x_d, w_d, y_d);
		matrixMultiplyShared <<<DimGrid_matmul, DimBlock_matmul,0,stream2>>> (device_w_unroll , device_x_unroll+ (i +2)* stripe_x_unroll, device_y_unroll+ (i +2)* stripe_y_unroll,numARows, numAColumns,numBRows, numBColumns,numCRows, numCColumns);
		reroll_y_kernel <<<DimGrid_reroll_y, DimBlock_reroll_y,0,stream1>>> (device_y_unroll + (i+1) * stripe_y_unroll, device_y + (i+1) * stripe_y, y_d);
		cudaMemcpyAsync(y + i * stripe_y, device_y + i* stripe_y, stripe_y * sizeof(float), cudaMemcpyDeviceToHost,stream0);
		cudaDeviceSynchronize();
	}
	unroll_x_kernel <<<DimGrid_unroll_x, DimBlock_unroll_x,0,stream4>>> (device_x + (xdims[0] - 1)* stripe_x, device_x_unroll+(xdims[0] - 1)* stripe_x_unroll, x_d, w_d, y_d);
	matrixMultiplyShared <<<DimGrid_matmul, DimBlock_matmul,0,stream3>>> (device_w_unroll , device_x_unroll+ (xdims[0] - 2)* stripe_x_unroll, device_y_unroll+ (xdims[0] - 2)* stripe_y_unroll,numARows, numAColumns,numBRows, numBColumns,numCRows, numCColumns);
	reroll_y_kernel <<<DimGrid_reroll_y, DimBlock_reroll_y,0,stream2>>> (device_y_unroll + (xdims[0] - 3) * stripe_y_unroll, device_y + (xdims[0] - 3) * stripe_y, y_d);
	cudaMemcpyAsync(y + (xdims[0] - 4) * stripe_y, device_y +(xdims[0] - 4)* stripe_y, stripe_y * sizeof(float), cudaMemcpyDeviceToHost,stream1);
    cudaDeviceSynchronize();
    matrixMultiplyShared <<<DimGrid_matmul, DimBlock_matmul,0,stream4>>> (device_w_unroll , device_x_unroll+ (xdims[0] - 1)* stripe_x_unroll, device_y_unroll+ (xdims[0] - 1)* stripe_y_unroll,numARows, numAColumns,numBRows, numBColumns,numCRows, numCColumns);
	reroll_y_kernel <<<DimGrid_reroll_y, DimBlock_reroll_y,0,stream3>>> (device_y_unroll + (xdims[0] - 2) * stripe_y_unroll, device_y + (xdims[0] - 2)* stripe_y, y_d);
	cudaMemcpyAsync(y + (xdims[0] - 3) * stripe_y, device_y+(xdims[0] - 3) * stripe_y, stripe_y * sizeof(float), cudaMemcpyDeviceToHost,stream2);
	cudaDeviceSynchronize();
	reroll_y_kernel <<<DimGrid_reroll_y, DimBlock_reroll_y,0,stream4>>> (device_y_unroll + (xdims[0] - 1) * stripe_y_unroll, device_y + (xdims[0] - 1) * stripe_y, y_d);
	cudaMemcpyAsync(y + (xdims[0] - 2) * stripe_y, device_y+(xdims[0] - 2) * stripe_y, stripe_y * sizeof(float), cudaMemcpyDeviceToHost,stream3);
	cudaDeviceSynchronize();
	cudaMemcpyAsync(y + (xdims[0] - 1) * stripe_y, device_y +(xdims[0] - 1)* stripe_y, stripe_y * sizeof(float), cudaMemcpyDeviceToHost,stream4);
	cudaDeviceSynchronize();
	


	cudaFree(device_x);
	cudaFree(device_y);
	cudaFree(device_y_unroll);
	cudaFree(device_x_unroll);
	cudaFree(device_w_unroll);
}

// void conv_forward_unroll(const float *x, const float *w, float *y, const int xdims[4], const int wdims[4], const int ydims[4]) {
// 	float *device_x, *device_y, *device_x_unroll, *device_w_unroll, *device_y_unroll;

// 	std::cout<< "\nINPUT DIMENSIONS:\n";
// 	std::cout<< "N: "<< xdims[0] << ", H: "<< xdims[1] << ", W: "<< xdims[2] << ", C: "<< xdims[3] << "\n";
// 	std::cout<< "K1: "<< wdims[0] << ", K2: "<< wdims[1] << ", C: "<< wdims[2] << ", M: "<< wdims[3] << "\n";
// 	std::cout<< "N: "<< ydims[0] << ", H_Out: "<< ydims[1] << ", W_Out: "<< ydims[2] << ", M: "<< ydims[3] << "\n\n";

// 	dims y_d, w_d, x_d;
// 	for (int i = 0; i < 4; i++) {
// 		y_d.dim[i] = ydims[i];
// 		x_d.dim[i] = xdims[i];
// 		w_d.dim[i] = wdims[i];
// 	}

// 	int numAColumns = wdims[0] * wdims[1] * wdims[2], numARows = ydims[3];
// 	int numBColumns = ydims[1] * ydims[2], numBRows = wdims[0] * wdims[1] * wdims[2];
// 	int numCColumns = numBColumns, numCRows = numARows;

// 	int size_x = sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3];
// 	int size_y = sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3];
// 	int size_x_unroll = sizeof(float) * wdims[0] * wdims[1] * wdims[2] * ydims[1] * ydims[2];
// 	int size_w_unroll = sizeof(float) * wdims[0] * wdims[1] * wdims[2] * ydims[3];
// 	int size_y_unroll = sizeof(float) * ydims[1] * ydims[2] * ydims[3];

// 	int stripe_x = xdims[1] * xdims[2] * xdims[3];
// 	int stripe_y = ydims[1] * ydims[2] * ydims[3];

// 	cudaMalloc((void **)&device_x, size_x);
// 	cudaMalloc((void **)&device_y, size_y);
// 	cudaMalloc((void **)&device_x_unroll, size_x_unroll);
// 	cudaMalloc((void **)&device_w_unroll, size_w_unroll);
// 	cudaMalloc((void **)&device_y_unroll, size_y_unroll);

// 	float * w_unroll = (float *)malloc(size_w_unroll * sizeof(float));
// 	unroll_weights(w, w_unroll, w_d);

// 	cudaMemcpy(device_x, x, size_x, cudaMemcpyHostToDevice);
// 	cudaMemcpy(device_w_unroll, w_unroll, size_w_unroll, cudaMemcpyHostToDevice);

// 	// dim3 DimBlock_unroll_x(MAX_THREADS, 1, 1);
// 	// dim3 DimGrid_unroll_x(ceil((float)(wdims[2] * ydims[1] * ydims[2]) / MAX_THREADS), 1, 1);

// 	dim3 DimBlock_unroll_x(wdims[2], wdims[0], wdims[1]);
// 	dim3 DimGrid_unroll_x(ydims[1], ydims[2], 1);

// 	dim3 DimBlock_matmul(TILE_WIDTH, TILE_WIDTH, 1);
// 	dim3 DimGrid_matmul(ceil((float)(ydims[1] * ydims[2]) / TILE_WIDTH), ceil((float)(ydims[3]) / TILE_WIDTH), 1);

// 	dim3 DimBlock_reroll_y(MAX_THREADS, 1, 1);
// 	dim3 DimGrid_reroll_y(ceil((float)(ydims[1] * ydims[2] * ydims[3]) / MAX_THREADS), 1, 1);

// 	for (int i = 0; i < xdims[0]; i++) {
// 		unroll_x_kernel <<<DimGrid_unroll_x, DimBlock_unroll_x>>> (device_x + i * stripe_x, device_x_unroll, x_d, w_d, y_d);
// 		matrixMultiplyShared <<<DimGrid_matmul, DimBlock_matmul>>> (device_w_unroll, device_x_unroll, device_y_unroll,
// 			numARows, numAColumns,
// 			numBRows, numBColumns,
// 			numCRows, numCColumns);
// 		reroll_y_kernel <<<DimGrid_reroll_y, DimBlock_reroll_y>>> (device_y_unroll, device_y + i * stripe_y, y_d);
// 	}

// 	cudaMemcpy(y, device_y, size_y, cudaMemcpyDeviceToHost);

// 	cudaFree(device_x);
// 	cudaFree(device_y);
// 	cudaFree(device_y_unroll);
// 	cudaFree(device_x_unroll);
// 	cudaFree(device_w_unroll);
// }

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

// Recified linear unit 2d
static void relu2(float *X, const int xdims[2]) {
  for (const auto i : range(0, xdims[0] * xdims[1])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

__global__ void fully_forward_kernel(float *X, float *W, float *Y, int xdim0, int xdim1, int wdim1) {
  int i = blockIdx.x;
  int j = threadIdx.x;
  float sum = 0;

  for(int k = 0; k < xdim1; k++) {
    if (i < xdim0 && j < wdim1) {
      sum += X[i * xdim1 + k] * W[k * wdim1 + j];
    }
  }

  if (i < xdim0 && j < wdim1) {
    Y[i * wdim1 + j] = sum;
  }
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

  fully_forward_kernel <<<xdims[0], xdims[1]>>> (device_x, device_w, device_y, xdims[0], xdims[1], wdims[1]);

  cudaMemcpy(y, device_y, size_y, cudaMemcpyDeviceToHost);

  cudaFree(device_x);
  cudaFree(device_w);
  cudaFree(device_y);
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
  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1),
                       (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  auto a = zeros<float>(adims);
  // conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);

	const auto tic = now();

  conv_forward_unroll(x, conv1, a, xdims, conv1dims, adims);

	const auto toc = now();
	const auto elapsed = std::chrono::duration<double, std::milli>(toc - tic).count();;
	std::cout << "Calling f(args...) took " << elapsed << "milliseconds\n";

  // average pooling
  const int pool_size = 2;
  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size,
                         adims[3]};
  auto b = zeros<float>(bdims);

	const auto tic1 = now();

  // average_pool(a, adims, pool_size, b, bdims);
  average_pool_parallel(a, b, adims, bdims, pool_size);

	const auto toc1 = now();
	const auto elapsed1 = std::chrono::duration<double, std::milli>(toc1 - tic1).count();;
	std::cout << "Calling f(args...) took " << elapsed1 << "milliseconds\n";


  // conv layer
  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1),
                       (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  auto c = zeros<float>(cdims);

	const auto tic2 = now();

  // conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);
  conv_forward_unroll(b, conv2, c, bdims, conv2dims, cdims);

	const auto toc2 = now();
	const auto elapsed2 = std::chrono::duration<double, std::milli>(toc2 - tic2).count();;
	std::cout << "Calling f(args...) took " << elapsed2 << "milliseconds\n";

  // average pooling
  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size,
                       cdims[3]};
  auto d = zeros<float>(ddims);
  // average_pool(c, cdims, pool_size, d, ddims);

	const auto tic3 = now();

  average_pool_parallel(c, d, cdims, ddims, pool_size);

	const auto toc3 = now();
	const auto elapsed3 = std::chrono::duration<double, std::milli>(toc3 - tic3).count();;
	std::cout << "Calling f(args...) took " << elapsed3 << "milliseconds\n";

  // reshape
  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};

  // matrix multiplication
  const int edims[] = {ddims[0], fc1dims[1]};
  auto e            = zeros<float>(edims);
  //fully_forward(d, ddims2, fc1, fc1dims, e, edims);
  fully_forward_parallel(d, fc1, e, ddims2, fc1dims, edims);

  // relu
  relu2(e, edims);

  // matrix multiplication
  const int fdims[] = {edims[0], fc2dims[1]};
  auto f            = zeros<float>(fdims);
  // fully_forward(e, edims, fc2, fc2dims, f, fdims);
  fully_forward_parallel(e, fc2, f, edims, fc2dims, fdims);

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
