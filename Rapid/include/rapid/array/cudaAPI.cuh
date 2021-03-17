#pragma once

#include "../internal.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256

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
						printf("Info: %f\n", arr[i]);
					}
				}
				
				__global__
					void fill_float(uint64_t size, float *arr, float val)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						arr[i] = val;
					}
				}

				__global__
					void fill_double(uint64_t size, double *arr, double val)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						arr[i] = val;
					}
				}

				__global__
					void add_array_array_float(uint64_t size, const float *a, const float *b, float *res)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i] = a[i] + b[i];
					}
				}

				__global__
					void add_array_array_double(uint64_t size, const double *a, const double *b, double *res)
				{
					unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
					unsigned int stride = blockDim.x * gridDim.x;

					for (unsigned int i = index; i < size; i += stride)
					{
						res[i] = a[i] + b[i];
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
			
			inline void fill(uint64_t size, float *arr, float val, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::fill_float << <numBlocks, blockSize >> > (size, arr, val);
			}

			inline void fill(uint64_t size, double *arr, double val, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::fill_double << <numBlocks, blockSize >> > (size, arr, val);
			}

			inline void add_array_array(uint64_t size, const float *a, const float *b, float *res, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::add_array_array_float << <numBlocks, blockSize >> > (size, a, b, res);
			}

			inline void add_array_array(uint64_t size, const double *a, const double *b, double *res, int sync = 1)
			{
				// Perform calculation
				unsigned int blockSize = BLOCK_SIZE;
				unsigned int numBlocks = (size + blockSize - 1) / blockSize;

				if (sync)
					cudaSafeCall(cudaDeviceSynchronize());

				kernel::add_array_array_double << <numBlocks, blockSize >> > (size, a, b, res);
			}
		}
	}
}
