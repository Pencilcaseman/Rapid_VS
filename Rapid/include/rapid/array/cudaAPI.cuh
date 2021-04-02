#pragma once

#include "../internal.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256
#define TILE_WIDTH 32

namespace rapid
{
	namespace ndarray
	{
		namespace cuda
		{
			namespace kernel
			{
				__global__
					void printStuff_float(unsigned int size, const float *arr)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						printf("PrintStuff Information: %f\n", arr[i]);
					}
				}

				__global__
					void rowToColumnOrdering_float(unsigned int rows, unsigned int cols, const float *arr, float *res)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < rows * cols; i += stride)
					{
						unsigned int row = i / cols;
						unsigned int col = i % cols;

						res[row + col * rows] = arr[col + row * cols];
					}
				}

				__global__
					void rowToColumnOrdering_double(unsigned int rows, unsigned int cols, const double *arr, double *res)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < rows * cols; i += stride)
					{
						unsigned int row = i / cols;
						unsigned int col = i % cols;

						res[row + col * rows] = arr[col + row * cols];
					}
				}

				__global__
					void columnToRowOrdering_float(unsigned int rows, unsigned int cols, const float *arr, float *res)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < rows * cols; i += stride)
					{
						unsigned int row = i / cols;
						unsigned int col = i % cols;

						res[col + row * cols] = arr[row + col * rows];
					}
				}

				__global__
					void columnToRowOrdering_double(unsigned int rows, unsigned int cols, const double *arr, double *res)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < rows * cols; i += stride)
					{
						unsigned int row = i / cols;
						unsigned int col = i % cols;

						res[col + row * cols] = arr[row + col * rows];
					}
				}

				__global__
					void fill_float(unsigned int size, float *arr, const unsigned int M, float val)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						arr[i * M] = val;
					}
				}

				__global__
					void fill_double(unsigned int size, double *arr, const unsigned int M, double val)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						arr[i * M] = val;
					}
				}

				__global__
					void add_array_array_float(unsigned int size, const float *a, const unsigned int M, const float *b, const unsigned int N, float *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] + b[i * N];
					}
				}

				__global__
					void add_array_array_double(unsigned int size, const double *a, const unsigned int M, const double *b, const unsigned int N, double *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] + b[i * N];
					}
				}

				__global__
					void sub_array_array_float(unsigned int size, const float *a, const unsigned int M, const float *b, const unsigned int N, float *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] - b[i * N];
					}
				}

				__global__
					void sub_array_array_double(unsigned int size, const double *a, const unsigned int M, const double *b, const unsigned int N, double *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] - b[i * N];
					}
				}

				__global__
					void mul_array_array_float(unsigned int size, const float *a, const unsigned int M, const float *b, const unsigned int N, float *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] * b[i * N];
					}
				}

				__global__
					void mul_array_array_double(unsigned int size, const double *a, const unsigned int M, const double *b, const unsigned int N, double *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] * b[i * N];
					}
				}

				__global__
					void div_array_array_float(unsigned int size, const float *a, const unsigned int M, const float *b, const unsigned int N, float *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] / b[i * N];
					}
				}

				__global__
					void div_array_array_double(unsigned int size, const double *a, const unsigned int M, const double *b, const unsigned int N, double *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] / b[i * N];
					}
				}

				__global__
					void add_array_scalar_float(unsigned int size, const float *a, const unsigned int M, const float b, float *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] + b;
					}
				}

				__global__
					void add_array_scalar_double(unsigned int size, const double *a, const unsigned int M, const double b, double *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] + b;
					}
				}

				__global__
					void sub_array_scalar_float(unsigned int size, const float *a, const unsigned int M, const float b, float *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] - b;
					}
				}

				__global__
					void sub_array_scalar_double(unsigned int size, const double *a, const unsigned int M, const double b, double *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] - b;
					}
				}

				__global__
					void mul_array_scalar_float(unsigned int size, const float *a, const unsigned int M, const float b, float *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] * b;
					}
				}

				__global__
					void mul_array_scalar_double(unsigned int size, const double *a, const unsigned int M, const double b, double *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] * b;
					}
				}

				__global__
					void div_array_scalar_float(unsigned int size, const float *a, const unsigned int M, const float b, float *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] / b;
					}
				}

				__global__
					void div_array_scalar_double(unsigned int size, const double *a, const unsigned int M, const double b, double *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] / b;
					}
				}

				__global__
					void add_scalar_array_float(unsigned int size, const float a, const float *b, const unsigned int N, float *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a + b[i * N];
					}
				}

				__global__
					void add_scalar_array_double(unsigned int size, const double a, const double *b, const unsigned int N, double *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a + b[i * N];
					}
				}

				__global__
					void sub_scalar_array_float(unsigned int size, const float a, const float *b, const unsigned int N, float *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a - b[i * N];
					}
				}

				__global__
					void sub_scalar_array_double(unsigned int size, const double a, const double *b, const unsigned int N, double *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a - b[i * N];
					}
				}

				__global__
					void mul_scalar_array_float(unsigned int size, const float a, const float *b, const unsigned int N, float *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a * b[i * N];
					}
				}

				__global__
					void mul_scalar_array_double(unsigned int size, const double a, const double *b, const unsigned int N, double *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a * b[i * N];
					}
				}

				__global__
					void div_scalar_array_float(unsigned int size, const float a, const float *b, const unsigned int N, float *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a / b[i * N];
					}
				}

				__global__
					void div_scalar_array_double(unsigned int size, const double a, const double *b, const unsigned int N, double *res, const unsigned int K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a / b[i * N];
					}
				}

				template<typename matrixType>
				__global__
					void matrixProduct(const matrixType *A_d, const matrixType *B_d, matrixType *C_d, unsigned int m, unsigned int k, unsigned int n)
				{
					__shared__ matrixType ds_A[TILE_WIDTH][TILE_WIDTH];
					__shared__ matrixType ds_B[TILE_WIDTH][TILE_WIDTH];
					unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
					unsigned int tx = threadIdx.x;
					unsigned int ty = threadIdx.y;
					matrixType sum = 0;

					for (int t = 0; t < (n - 1) / TILE_WIDTH + 1; t++)
					{
						if (row < m && t * TILE_WIDTH + tx < n)
							ds_A[ty][tx] = A_d[row * n + t * TILE_WIDTH + tx];
						else
							ds_A[ty][tx] = 0.0;
						if (t * TILE_WIDTH + ty < n && col < k)
							ds_B[ty][tx] = B_d[(t * TILE_WIDTH + ty) * k + col];
						else
							ds_B[ty][tx] = 0.0;
						__syncthreads();
						for (int i = 0; i < TILE_WIDTH; i++)
							sum += ds_A[ty][i] * ds_B[i][tx];
						__syncthreads();
					}
					if (row < m && col < k)
						C_d[col + row * k] = sum;
				}

				__global__
					void array_minimum_float(unsigned int size, const float *arr, const unsigned int M, const float val, float *res, const unsigned int N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] < val ? val : arr[i * M];
					}
				}

				__global__
					void array_minimum_double(unsigned int size, const double *arr, const unsigned int M, const double val, double *res, const unsigned int N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] < val ? val : arr[i * M];
					}
				}

				__global__
					void array_maximum_float(unsigned int size, const float *arr, const unsigned int M, const float val, float *res, const unsigned int N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] > val ? val : arr[i * M];
					}
				}

				__global__
					void array_maximum_double(unsigned int size, const double *arr, const unsigned int M, const double val, double *res, const unsigned int N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] > val ? val : arr[i * M];
					}
				}

				__global__
					void array_less_float(unsigned int size, const float *arr, const unsigned int M, const float val, float *res, const unsigned int N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] < val ? 1 : 0;
					}
				}

				__global__
					void array_less_double(unsigned int size, const double *arr, const unsigned int M, const double val, double *res, const unsigned int N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] < val ? 1 : 0;
					}
				}

				__global__
					void array_greater_float(unsigned int size, const float *arr, const unsigned int M, const float val, float *res, const unsigned int N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] > val ? 1 : 0;
					}
				}

				__global__
					void array_greater_double(unsigned int size, const double *arr, const unsigned int M, const double val, double *res, const unsigned int N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] > val ? 1 : 0;
					}
				}

				__global__
					void array_exp_float(unsigned int size, const float *arr, const unsigned int M, float *res, const unsigned int N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = std::exp(arr[i * M]);
					}
				}

				__global__
					void array_exp_double(unsigned int size, const double *arr, const unsigned int M, double *res, const unsigned int N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = std::exp(arr[i * M]);
					}
				}

				__global__
					void array_square_float(unsigned int size, const float *arr, const unsigned int M, float *res, const unsigned int N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] * arr[i * M];
					}
				}

				__global__
					void array_square_double(unsigned int size, const double *arr, const unsigned int M, double *res, const unsigned int N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] * arr[i * M];
					}
				}

				__global__
					void array_pow_float(unsigned int size, const float *arr, const unsigned int M, const float n, float *res, const unsigned int N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = std::pow(arr[i * M], n);
					}
				}

				__global__
					void array_pow_double(unsigned int size, const double *arr, const unsigned int M, const double n, double *res, const unsigned int N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = std::pow(arr[i * M], n);
					}
				}
			}

			inline void printStuff(unsigned int size, const float *arr, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::printStuff_float << <numBlocks, blockSize >> > (size, arr);
			}

			inline void rowToColumnOrdering(unsigned int rows, unsigned int cols, float *arr, float *res, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (rows * cols + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::rowToColumnOrdering_float << <numBlocks, blockSize >> > (rows, cols, arr, res);
			}

			inline void rowToColumnOrdering(unsigned int rows, unsigned int cols, double *arr, double *res, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (rows * cols + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::rowToColumnOrdering_double << <numBlocks, blockSize >> > (rows, cols, arr, res);
			}

			inline void columnToRowOrdering(unsigned int rows, unsigned int cols, float *arr, float *res, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (rows * cols + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::columnToRowOrdering_float << <numBlocks, blockSize >> > (rows, cols, arr, res);
			}

			inline void columnToRowOrdering(unsigned int rows, unsigned int cols, double *arr, double *res, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (rows * cols + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::columnToRowOrdering_double << <numBlocks, blockSize >> > (rows, cols, arr, res);
			}

			inline void fill(unsigned int size, float *arr, const unsigned int M, float val, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::fill_float << <numBlocks, blockSize >> > (size, arr, M, val);
			}

			inline void fill(unsigned int size, double *arr, const unsigned int M, double val, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::fill_double << <numBlocks, blockSize >> > (size, arr, M, val);
			}

			inline void add_array_array(unsigned int size, const float *a, const unsigned int M, const float *b, const unsigned int N, float *res, const unsigned int K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::add_array_array_float << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void add_array_array(unsigned int size, const double *a, const unsigned int M, const double *b, const unsigned int N, double *res, const unsigned int K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::add_array_array_double << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void sub_array_array(unsigned int size, const float *a, const unsigned int M, const float *b, const unsigned int N, float *res, const unsigned int K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::sub_array_array_float << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void sub_array_array(unsigned int size, const double *a, const unsigned int M, const double *b, const unsigned int N, double *res, const unsigned int K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::sub_array_array_double << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void mul_array_array(unsigned int size, const float *a, const unsigned int M, const float *b, const unsigned int N, float *res, const unsigned int K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::mul_array_array_float << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void mul_array_array(unsigned int size, const double *a, const unsigned int M, const double *b, const unsigned int N, double *res, const unsigned int K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::mul_array_array_double << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void div_array_array(unsigned int size, const float *a, const unsigned int M, const float *b, const unsigned int N, float *res, const unsigned int K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::div_array_array_float << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void div_array_array(unsigned int size, const double *a, const unsigned int M, const double *b, const unsigned int N, double *res, const unsigned int K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::div_array_array_double << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void add_array_scalar(unsigned int size, const float *a, const unsigned int M, const float b, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::add_array_scalar_float << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void add_array_scalar(unsigned int size, const double *a, const unsigned int M, const double b, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::add_array_scalar_double << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void sub_array_scalar(unsigned int size, const float *a, const unsigned int M, const float b, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::sub_array_scalar_float << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void sub_array_scalar(unsigned int size, const double *a, const unsigned int M, const double b, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::sub_array_scalar_double << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void mul_array_scalar(unsigned int size, const float *a, const unsigned int M, const float b, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::mul_array_scalar_float << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void mul_array_scalar(unsigned int size, const double *a, const unsigned int M, const double b, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::mul_array_scalar_double << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void div_array_scalar(unsigned int size, const float *a, const unsigned int M, const float b, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::div_array_scalar_float << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void div_array_scalar(unsigned int size, const double *a, const unsigned int M, const double b, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::div_array_scalar_double << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void add_scalar_array(unsigned int size, const float a, const float *b, const unsigned int M, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::add_scalar_array_float << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void add_scalar_array(unsigned int size, const double a, const double *b, const unsigned int M, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::add_scalar_array_double << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void sub_scalar_array(unsigned int size, const float a, const float *b, const unsigned int M, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::sub_scalar_array_float << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void sub_scalar_array(unsigned int size, const double a, const double *b, const unsigned int M, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::sub_scalar_array_double << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void mul_scalar_array(unsigned int size, const float a, const float *b, const unsigned int M, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::mul_scalar_array_float << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void mul_scalar_array(unsigned int size, const double a, const double *b, const unsigned int M, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::mul_scalar_array_double << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void div_scalar_array(unsigned int size, const float a, const float *b, const unsigned int M, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::div_scalar_array_float << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void div_scalar_array(unsigned int size, const double a, const double *b, const unsigned int M, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::div_scalar_array_double << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void gemm(cublasHandle_t handle,
							 cublasOperation_t transa, cublasOperation_t transb,
							 unsigned int m, unsigned int n, unsigned int k,
							 const float *alpha,
							 const float *A, unsigned int lda,
							 const float *B, unsigned int ldb,
							 const float *beta,
							 float *C, unsigned int ldc)
			{
				cublasSafeCall(cublasSgemm(handle, transa, transb,
							   m, n, k,
							   alpha,
							   A, lda,
							   B, ldb,
							   beta,
							   C, ldc));
			}

			inline void gemm(cublasHandle_t handle,
							 cublasOperation_t transa, cublasOperation_t transb,
							 unsigned int m, unsigned int n, unsigned int k,
							 const double *alpha,
							 const double *A, unsigned int lda,
							 const double *B, unsigned int ldb,
							 const double *beta,
							 double *C, unsigned int ldc)
			{
				cublasSafeCall(cublasDgemm(handle, transa, transb,
							   m, n, k,
							   alpha,
							   A, lda,
							   B, ldb,
							   beta,
							   C, ldc));
			}

			inline void geam(cublasHandle_t handle,
							 cublasOperation_t transa, cublasOperation_t transb,
							 unsigned int m, unsigned int n,
							 const float *alpha,
							 const float *A, unsigned int lda,
							 const float *beta,
							 const float *B, unsigned int ldb,
							 float *C, unsigned int ldc)
			{
				cublasSafeCall(cublasSgeam(handle,
							   transa, transb,
							   m, n,
							   alpha,
							   A, lda,
							   beta,
							   B, ldb,
							   C, ldc));
			}

			inline void geam(cublasHandle_t handle,
							 cublasOperation_t transa, cublasOperation_t transb,
							 unsigned int m, unsigned int n,
							 const double *alpha,
							 const double *A, unsigned int lda,
							 const double *beta,
							 const double *B, unsigned int ldb,
							 double *C, unsigned int ldc)
			{
				cublasSafeCall(cublasDgeam(handle,
							   transa, transb,
							   m, n,
							   alpha,
							   A, lda,
							   beta,
							   B, ldb,
							   C, ldc));
			}

			template<typename t>
			inline void dot(unsigned int m, unsigned int n, unsigned int k, const t *a, const t *b, t *res)
			{
				dim3 dimGrid((k - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1, 1);
				dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
				kernel::matrixProduct << <dimGrid, dimBlock >> > (a, b, res, m, k, n);
			}

			inline void array_minimum(unsigned int size, const float *arr, const unsigned int M, const float val, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_minimum_float << <numBlocks, blockSize >> > (size, arr, M, val, res, N);
			}

			inline void array_minimum(unsigned int size, const double *arr, const unsigned int M, const double val, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_minimum_double << <numBlocks, blockSize >> > (size, arr, M, val, res, N);
			}

			inline void array_maximum(unsigned int size, const float *arr, const unsigned int M, const float val, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_maximum_float << <numBlocks, blockSize >> > (size, arr, M, val, res, N);
			}

			inline void array_maximum(unsigned int size, const double *arr, const unsigned int M, const double val, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_maximum_double << <numBlocks, blockSize >> > (size, arr, M, val, res, N);
			}

			inline void array_less(unsigned int size, const float *arr, const unsigned int M, const float val, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_less_float << <numBlocks, blockSize >> > (size, arr, M, val, res, N);
			}

			inline void array_less(unsigned int size, const double *arr, const unsigned int M, const double val, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_less_double << <numBlocks, blockSize >> > (size, arr, M, val, res, N);
			}

			inline void array_greater(unsigned int size, const float *arr, const unsigned int M, const float val, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_greater_float << <numBlocks, blockSize >> > (size, arr, M, val, res, N);
			}

			inline void array_greater(unsigned int size, const double *arr, const unsigned int M, const double val, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_greater_double << <numBlocks, blockSize >> > (size, arr, M, val, res, N);
			}

			inline void array_exp(unsigned int size, const float *arr, const unsigned int M, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_exp_float << <numBlocks, blockSize >> > (size, arr, M, res, N);
			}

			inline void array_exp(unsigned int size, const double *arr, const unsigned int M, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_exp_double << <numBlocks, blockSize >> > (size, arr, M, res, N);
			}

			inline void array_square(unsigned int size, const float *arr, const unsigned int M, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_square_float << <numBlocks, blockSize >> > (size, arr, M, res, N);
			}

			inline void array_square(unsigned int size, const double *arr, const unsigned int M, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_square_double << <numBlocks, blockSize >> > (size, arr, M, res, N);
			}

			inline void array_pow(unsigned int size, const float *arr, const unsigned int M, const float p, float *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_pow_float << <numBlocks, blockSize >> > (size, arr, M, p, res, N);
			}

			inline void array_pow(unsigned int size, const double *arr, const unsigned int M, const float p, double *res, const unsigned int N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_pow_double << <numBlocks, blockSize >> > (size, arr, M, p, res, N);
			}
		}
	}
}
