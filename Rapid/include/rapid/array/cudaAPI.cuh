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
					void printStuff_float(uint64_t size, const float *arr)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						printf("PrintStuff Information: %f\n", arr[i]);
					}
				}

				__global__
					void rowToColumnOrdering_float(uint64_t rows, uint64_t cols, const float *arr, float *res)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < rows * cols; i += stride)
					{
						uint64_t row = i / cols;
						uint64_t col = i % cols;

						res[row + col * rows] = arr[col + row * cols];
					}
				}

				__global__
					void rowToColumnOrdering_double(uint64_t rows, uint64_t cols, const double *arr, double *res)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < rows * cols; i += stride)
					{
						uint64_t row = i / cols;
						uint64_t col = i % cols;

						res[row + col * rows] = arr[col + row * cols];
					}
				}

				__global__
					void columnToRowOrdering_float(uint64_t rows, uint64_t cols, const float *arr, float *res)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < rows * cols; i += stride)
					{
						uint64_t row = i / cols;
						uint64_t col = i % cols;

						res[col + row * cols] = arr[row + col * rows];
					}
				}

				__global__
					void columnToRowOrdering_double(uint64_t rows, uint64_t cols, const double *arr, double *res)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < rows * cols; i += stride)
					{
						uint64_t row = i / cols;
						uint64_t col = i % cols;

						res[col + row * cols] = arr[row + col * rows];
					}
				}

				__global__
					void fill_float(uint64_t size, float *arr, const uint64_t M, float val)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						arr[i * M] = val;
					}
				}

				__global__
					void fill_double(uint64_t size, double *arr, const uint64_t M, double val)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						arr[i * M] = val;
					}
				}

				__global__
					void add_array_array_float(uint64_t size, const float *a, const uint64_t M, const float *b, const uint64_t N, float *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] + b[i * N];
					}
				}

				__global__
					void add_array_array_double(uint64_t size, const double *a, const uint64_t M, const double *b, const uint64_t N, double *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] + b[i * N];
					}
				}

				__global__
					void sub_array_array_float(uint64_t size, const float *a, const uint64_t M, const float *b, const uint64_t N, float *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] - b[i * N];
					}
				}

				__global__
					void sub_array_array_double(uint64_t size, const double *a, const uint64_t M, const double *b, const uint64_t N, double *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] - b[i * N];
					}
				}

				__global__
					void mul_array_array_float(uint64_t size, const float *a, const uint64_t M, const float *b, const uint64_t N, float *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] * b[i * N];
					}
				}

				__global__
					void mul_array_array_double(uint64_t size, const double *a, const uint64_t M, const double *b, const uint64_t N, double *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] * b[i * N];
					}
				}

				__global__
					void div_array_array_float(uint64_t size, const float *a, const uint64_t M, const float *b, const uint64_t N, float *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] / b[i * N];
					}
				}

				__global__
					void div_array_array_double(uint64_t size, const double *a, const uint64_t M, const double *b, const uint64_t N, double *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] / b[i * N];
					}
				}

				__global__
					void add_array_scalar_float(uint64_t size, const float *a, const uint64_t M, const float b, float *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] + b;
					}
				}

				__global__
					void add_array_scalar_double(uint64_t size, const double *a, const uint64_t M, const double b, double *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] + b;
					}
				}

				__global__
					void sub_array_scalar_float(uint64_t size, const float *a, const uint64_t M, const float b, float *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] - b;
					}
				}

				__global__
					void sub_array_scalar_double(uint64_t size, const double *a, const uint64_t M, const double b, double *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] - b;
					}
				}

				__global__
					void mul_array_scalar_float(uint64_t size, const float *a, const uint64_t M, const float b, float *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] * b;
					}
				}

				__global__
					void mul_array_scalar_double(uint64_t size, const double *a, const uint64_t M, const double b, double *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] * b;
					}
				}

				__global__
					void div_array_scalar_float(uint64_t size, const float *a, const uint64_t M, const float b, float *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] / b;
					}
				}

				__global__
					void div_array_scalar_double(uint64_t size, const double *a, const uint64_t M, const double b, double *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a[i * M] / b;
					}
				}

				__global__
					void add_scalar_array_float(uint64_t size, const float a, const float *b, const uint64_t N, float *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a + b[i * N];
					}
				}

				__global__
					void add_scalar_array_double(uint64_t size, const double a, const double *b, const uint64_t N, double *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a + b[i * N];
					}
				}

				__global__
					void sub_scalar_array_float(uint64_t size, const float a, const float *b, const uint64_t N, float *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a - b[i * N];
					}
				}

				__global__
					void sub_scalar_array_double(uint64_t size, const double a, const double *b, const uint64_t N, double *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a - b[i * N];
					}
				}

				__global__
					void mul_scalar_array_float(uint64_t size, const float a, const float *b, const uint64_t N, float *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a * b[i * N];
					}
				}

				__global__
					void mul_scalar_array_double(uint64_t size, const double a, const double *b, const uint64_t N, double *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a * b[i * N];
					}
				}

				__global__
					void div_scalar_array_float(uint64_t size, const float a, const float *b, const uint64_t N, float *res, const uint64_t K)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * K] = a / b[i * N];
					}
				}

				__global__
					void div_scalar_array_double(uint64_t size, const double a, const double *b, const uint64_t N, double *res, const uint64_t K)
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
					void matrixProduct(const matrixType *A_d, const matrixType *B_d, matrixType *C_d, uint64_t m, uint64_t k, uint64_t n)
				{
					__shared__ matrixType ds_A[TILE_WIDTH][TILE_WIDTH];
					__shared__ matrixType ds_B[TILE_WIDTH][TILE_WIDTH];
					uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
					uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
					uint64_t tx = threadIdx.x;
					uint64_t ty = threadIdx.y;
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
					void array_minimum_float(uint64_t size, const float *arr, const uint64_t M, const float val, float *res, const uint64_t N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] < val ? val : arr[i * M];
					}
				}

				__global__
					void array_minimum_double(uint64_t size, const double *arr, const uint64_t M, const double val, double *res, const uint64_t N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] < val ? val : arr[i * M];
					}
				}

				__global__
					void array_maximum_float(uint64_t size, const float *arr, const uint64_t M, const float val, float *res, const uint64_t N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] > val ? val : arr[i * M];
					}
				}

				__global__
					void array_maximum_double(uint64_t size, const double *arr, const uint64_t M, const double val, double *res, const uint64_t N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] > val ? val : arr[i * M];
					}
				}

























				__global__
					void array_exp_float(uint64_t size, const float *arr, const uint64_t M, float *res, const uint64_t N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = std::exp(arr[i * M]);
					}
				}

				__global__
					void array_exp_double(uint64_t size, const double *arr, const uint64_t M, double *res, const uint64_t N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = std::exp(arr[i * M]);
					}
				}

				__global__
					void array_square_float(uint64_t size, const float *arr, const uint64_t M, float *res, const uint64_t N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] * arr[i * M];
					}
				}

				__global__
					void array_square_double(uint64_t size, const double *arr, const uint64_t M, double *res, const uint64_t N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = arr[i * M] * arr[i * M];
					}
				}

				__global__
					void array_pow_float(uint64_t size, const float *arr, const uint64_t M, const float n, float *res, const uint64_t N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = std::pow(arr[i * M], n);
					}
				}

				__global__
					void array_pow_double(uint64_t size, const double *arr, const uint64_t M, const double n, double *res, const uint64_t N)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i * N] = std::pow(arr[i * M], n);
					}
				}
			}

			inline void printStuff(uint64_t size, const float *arr, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::printStuff_float << <numBlocks, blockSize >> > (size, arr);
			}

			inline void rowToColumnOrdering(uint64_t rows, uint64_t cols, float *arr, float *res, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (rows * cols + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::rowToColumnOrdering_float << <numBlocks, blockSize >> > (rows, cols, arr, res);
			}

			inline void rowToColumnOrdering(uint64_t rows, uint64_t cols, double *arr, double *res, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (rows * cols + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::rowToColumnOrdering_double << <numBlocks, blockSize >> > (rows, cols, arr, res);
			}

			inline void columnToRowOrdering(uint64_t rows, uint64_t cols, float *arr, float *res, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (rows * cols + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::columnToRowOrdering_float << <numBlocks, blockSize >> > (rows, cols, arr, res);
			}

			inline void columnToRowOrdering(uint64_t rows, uint64_t cols, double *arr, double *res, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (rows * cols + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::columnToRowOrdering_double << <numBlocks, blockSize >> > (rows, cols, arr, res);
			}

			inline void fill(uint64_t size, float *arr, const uint64_t M, float val, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::fill_float << <numBlocks, blockSize >> > (size, arr, M, val);
			}

			inline void fill(uint64_t size, double *arr, const uint64_t M, double val, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::fill_double << <numBlocks, blockSize >> > (size, arr, M, val);
			}

			inline void add_array_array(uint64_t size, const float *a, const uint64_t M, const float *b, const uint64_t N, float *res, const uint64_t K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::add_array_array_float << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void add_array_array(uint64_t size, const double *a, const uint64_t M, const double *b, const uint64_t N, double *res, const uint64_t K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::add_array_array_double << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void sub_array_array(uint64_t size, const float *a, const uint64_t M, const float *b, const uint64_t N, float *res, const uint64_t K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::sub_array_array_float << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void sub_array_array(uint64_t size, const double *a, const uint64_t M, const double *b, const uint64_t N, double *res, const uint64_t K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::sub_array_array_double << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void mul_array_array(uint64_t size, const float *a, const uint64_t M, const float *b, const uint64_t N, float *res, const uint64_t K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::mul_array_array_float << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void mul_array_array(uint64_t size, const double *a, const uint64_t M, const double *b, const uint64_t N, double *res, const uint64_t K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::mul_array_array_double << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void div_array_array(uint64_t size, const float *a, const uint64_t M, const float *b, const uint64_t N, float *res, const uint64_t K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::div_array_array_float << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void div_array_array(uint64_t size, const double *a, const uint64_t M, const double *b, const uint64_t N, double *res, const uint64_t K, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::div_array_array_double << <numBlocks, blockSize >> > (size, a, M, b, N, res, K);
			}

			inline void add_array_scalar(uint64_t size, const float *a, const uint64_t M, const float b, float *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::add_array_scalar_float << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void add_array_scalar(uint64_t size, const double *a, const uint64_t M, const double b, double *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::add_array_scalar_double << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void sub_array_scalar(uint64_t size, const float *a, const uint64_t M, const float b, float *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::sub_array_scalar_float << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void sub_array_scalar(uint64_t size, const double *a, const uint64_t M, const double b, double *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::sub_array_scalar_double << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void mul_array_scalar(uint64_t size, const float *a, const uint64_t M, const float b, float *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::mul_array_scalar_float << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void mul_array_scalar(uint64_t size, const double *a, const uint64_t M, const double b, double *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::mul_array_scalar_double << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void div_array_scalar(uint64_t size, const float *a, const uint64_t M, const float b, float *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::div_array_scalar_float << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void div_array_scalar(uint64_t size, const double *a, const uint64_t M, const double b, double *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::div_array_scalar_double << <numBlocks, blockSize >> > (size, a, M, b, res, N);
			}

			inline void add_scalar_array(uint64_t size, const float a, const float *b, const uint64_t M, float *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::add_scalar_array_float << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void add_scalar_array(uint64_t size, const double a, const double *b, const uint64_t M, double *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::add_scalar_array_double << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void sub_scalar_array(uint64_t size, const float a, const float *b, const uint64_t M, float *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::sub_scalar_array_float << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void sub_scalar_array(uint64_t size, const double a, const double *b, const uint64_t M, double *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::sub_scalar_array_double << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void mul_scalar_array(uint64_t size, const float a, const float *b, const uint64_t M, float *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::mul_scalar_array_float << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void mul_scalar_array(uint64_t size, const double a, const double *b, const uint64_t M, double *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::mul_scalar_array_double << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void div_scalar_array(uint64_t size, const float a, const float *b, const uint64_t M, float *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::div_scalar_array_float << <numBlocks, blockSize >> > (size, a, b, M, res, N);
			}

			inline void div_scalar_array(uint64_t size, const double a, const double *b, const uint64_t M, double *res, const uint64_t N, int sync = 1)
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
							 uint64_t m, uint64_t n, uint64_t k,
							 const float *alpha,
							 const float *A, uint64_t lda,
							 const float *B, uint64_t ldb,
							 const float *beta,
							 float *C, uint64_t ldc)
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
							 uint64_t m, uint64_t n, uint64_t k,
							 const double *alpha,
							 const double *A, uint64_t lda,
							 const double *B, uint64_t ldb,
							 const double *beta,
							 double *C, uint64_t ldc)
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
							 uint64_t m, uint64_t n,
							 const float *alpha,
							 const float *A, uint64_t lda,
							 const float *beta,
							 const float *B, uint64_t ldb,
							 float *C, uint64_t ldc)
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
							 uint64_t m, uint64_t n,
							 const double *alpha,
							 const double *A, uint64_t lda,
							 const double *beta,
							 const double *B, uint64_t ldb,
							 double *C, uint64_t ldc)
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
			inline void dot(uint64_t m, uint64_t n, uint64_t k, const t *a, const t *b, t *res)
			{
				dim3 dimGrid((k - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1, 1);
				dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
				kernel::matrixProduct << <dimGrid, dimBlock >> > (a, b, res, m, k, n);
			}

			inline void array_minimum(uint64_t size, const float *arr, const uint64_t M, const float val, float *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_minimum_float << <numBlocks, blockSize >> > (size, arr, M, val, res, N);
			}

			inline void array_minimum(uint64_t size, const double *arr, const uint64_t M, const double val, double *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_minimum_double << <numBlocks, blockSize >> > (size, arr, M, val, res, N);
			}

			inline void array_maximum(uint64_t size, const float *arr, const uint64_t M, const float val, float *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_maximum_float << <numBlocks, blockSize >> > (size, arr, M, val, res, N);
			}

			inline void array_maximum(uint64_t size, const double *arr, const uint64_t M, const double val, double *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_maximum_double << <numBlocks, blockSize >> > (size, arr, M, val, res, N);
			}

			inline void array_exp(uint64_t size, const float *arr, const uint64_t M, float *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_exp_float << <numBlocks, blockSize >> > (size, arr, M, res, N);
			}

			inline void array_exp(uint64_t size, const double *arr, const uint64_t M, double *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_exp_double << <numBlocks, blockSize >> > (size, arr, M, res, N);
			}

			inline void array_square(uint64_t size, const float *arr, const uint64_t M, float *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_square_float << <numBlocks, blockSize >> > (size, arr, M, res, N);
			}

			inline void array_square(uint64_t size, const double *arr, const uint64_t M, double *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_square_double << <numBlocks, blockSize >> > (size, arr, M, res, N);
			}

			inline void array_pow(uint64_t size, const float *arr, const uint64_t M, const float p, float *res, const uint64_t N, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::array_pow_float << <numBlocks, blockSize >> > (size, arr, M, p, res, N);
			}

			inline void array_pow(uint64_t size, const double *arr, const uint64_t M, const float p, double *res, const uint64_t N, int sync = 1)
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
