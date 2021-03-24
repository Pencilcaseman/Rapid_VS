#pragma once

#include "../internal.h"
#include "../math.h"

#include "fromData.h"

#ifndef RAPID_NO_BLAS
#include "cblasAPI.h"
#endif

#ifdef RAPID_CUDA
#include "cudaAPI.cuh"
#endif

namespace rapid
{
	namespace ndarray
	{
	#ifdef RAPID_CUDA
		namespace handle
		{
			static cublasHandle_t handle;
			bool handleInitialized = false;

			inline void createHandle()
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
				handleInitialized = true;
			}
		}
	#endif

		namespace utils
		{
			struct strContainer
			{
				std::string str;
				size_t decimalPoint;
			};

			/// <summary>
			/// Format a numerical value and return it as a string
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="val"></param>
			/// <returns></returns>
			template<typename t>
			strContainer formatNumerical(const t &val)
			{
				std::stringstream stream;
				stream << val;

				auto lastDecimal = stream.str().find_last_of('.');

				if (std::is_floating_point<t>::value && lastDecimal == std::string::npos)
				{
					stream << ".";
					lastDecimal = stream.str().length() - 1;
				}

				auto lastZero = stream.str().find_last_of('0');

				// Value is integral
				if (lastDecimal == std::string::npos)
					return {stream.str(), stream.str().length() - 1};

				return {stream.str(), lastDecimal};
			}

			/// <summary>
			/// Convert an index and a given array shape and return the memory
			/// offset for the given position in contiguous memory
			/// </summary>
			/// <typeparam name="indexT"></typeparam>
			/// <typeparam name="shapeT"></typeparam>
			/// <param name="index"></param>
			/// <param name="shape"></param>
			/// <returns></returns>
			template<typename indexT, typename shapeT>
			inline indexT ndToScalar(const std::vector<indexT> &index,
									 const std::vector<shapeT> &shape)
			{
				indexT sig = 1;
				indexT pos = 0;

				for (indexT i = shape.size(); i > 0; i--)
				{
					pos += (i - 1 < index.size() ? index[i - 1] : 0) * sig;
					sig *= shape[i - 1];
				}

				return pos;
			}

			/// <summary>
			/// Convert an index and a given array shape and return the memory
			/// offset for the given position in contiguous memory
			/// </summary>
			/// <typeparam name="indexT"></typeparam>
			/// <typeparam name="shapeT"></typeparam>
			/// <param name="index"></param>
			/// <param name="shape"></param>
			/// <returns></returns>
			template<typename indexT, typename shapeT>
			inline indexT ndToScalar(const std::initializer_list<indexT> &index,
									 const std::vector<shapeT> &shape)
			{
				indexT sig = 1;
				indexT pos = 0;
				uint64_t off;

				for (indexT i = shape.size(); i > 0; i--)
				{
					off = i - 1;
					pos += (i - 1 < index.size() ? (*(index.begin() + off)) : 0) * sig;
					sig *= shape[off];
				}

				return pos;
			}
		}

		namespace imp
		{
			/// <summary>
			/// Convert a set of dimensions into a memory location. Intended for internal use only
			/// </summary>
			/// <param name="dims"></param>
			/// <param name="pos"></param>
			/// <returns></returns>
			uint64_t dimsToIndex(const std::vector<uint64_t> &dims, const std::vector<uint64_t> &pos)
			{
				uint64_t index = 0;
				for (long int i = 0; i < dims.size(); i++)
				{
					uint64_t sub = pos[i];
					for (uint64_t j = i; j < dims.size() - 1; j++)
						sub *= dims[j + 1];
					index += sub;
				}
				return index;
			}
		}

		enum class ExecutionType
		{
			SERIAL = 0b0001,
			PARALLEL = 0b0010,
			MASSIVE = 0b0100
		};

	#ifdef RAPID_CUDA
		enum ArrayLocation
		{
			CPU,
			GPU
		};
	#else
		enum ArrayLocation
		{
			CPU
		};
	#endif

		/// <summary>
		/// A powerful and fast ndarray type, supporting a wide variety
		/// of optimized functions and routines. It also supports different
		/// arrayTypes, allowing for greater flexibility.
		/// </summary>
		/// <typeparam name="arrayType"></typeparam>
		template<typename arrayType, ArrayLocation location = CPU>
		class Array
		{
		public:
			std::vector<size_t> shape;
			arrayType *dataOrigin;
			arrayType *dataStart;
			size_t *originCount;
			bool isZeroDim;

		#ifdef RAPID_CUDA
			bool useMatrixData = false;
			uint64_t matrixRows = 0;
			uint64_t matrixAccess = 0;
		#endif

			/// <summary>
			/// Apply a lambda function to two arrays, storing the result in a third.
			/// Both arrays must be the same size, but this in not checked when running,
			/// so it is therefore the responsibility of the user to ensure this function
			/// is called safely
			/// </summary>
			/// <typeparam name="Lambda"></typeparam>
			/// <param name="a"></param>
			/// <param name="b"></param>
			/// <param name="c"></param>
			/// <param name="mode"></param>
			/// <param name="func"></param>
			template<typename Lambda, ArrayLocation loc>
			inline static void binaryOpArrayArray(const Array<arrayType, loc> &a, const Array<arrayType, loc> &b,
												  Array<arrayType, loc> &c, ExecutionType mode, Lambda func)
			{
				size_t size = math::prod(a.shape);

				if (mode == ExecutionType::SERIAL)
				{
					// Serial execution on CPU
					size_t index = 0;

					if (size > 3)
					{
						for (index = 0; index < size - 3; index += 4)
						{
							c.dataStart[index + 0] = func(a.dataStart[index + 0], b.dataStart[index + 0]);
							c.dataStart[index + 1] = func(a.dataStart[index + 1], b.dataStart[index + 1]);
							c.dataStart[index + 2] = func(a.dataStart[index + 2], b.dataStart[index + 2]);
							c.dataStart[index + 3] = func(a.dataStart[index + 3], b.dataStart[index + 3]);
						}
					}

					for (; index < size; index++)
						c.dataStart[index] = func(a.dataStart[index], b.dataStart[index]);
				}
				else if (mode == ExecutionType::PARALLEL)
				{
					// Parallel execution on CPU
					long index = 0;

				#pragma omp parallel for shared(size, a, b, c) private(index) default(none)
					for (index = 0; index < size; ++index)
						c.dataStart[index] = func(a.dataStart[index], b.dataStart[index]);
				}
				else
				{
					message::RapidError("Mode Error", "Invalid mode for binary mapping. Must be SERIAL or PARALLEL").display();
				}
			}

			/// <summary>
			/// Apply a lambda function to an array in the format
			/// func(array, scalar) and store the result. Both arrays
			/// must be the same size, but this in not checked when running,
			/// so it is therefore the responsibility of the user to ensure
			/// this function is called safely
			/// </summary>
			/// <typeparam name="Lambda"></typeparam>
			/// <param name="a"></param>
			/// <param name="b"></param>
			/// <param name="c"></param>
			/// <param name="mode"></param>
			/// <param name="func"></param>
			template<typename Lambda, ArrayLocation loc>
			inline static void binaryOpArrayScalar(const Array<arrayType, loc> &a, const arrayType &b,
												   Array<arrayType, loc> &c, ExecutionType mode, Lambda func)
			{
				size_t size = math::prod(a.shape);

				if (mode == ExecutionType::SERIAL)
				{
					// Serial execution on CPU
					size_t index = 0;

					if (size > 3)
					{
						for (index = 0; index < size - 3; index += 4)
						{
							c.dataStart[index + 0] = func(a.dataStart[index + 0], b);
							c.dataStart[index + 1] = func(a.dataStart[index + 1], b);
							c.dataStart[index + 2] = func(a.dataStart[index + 2], b);
							c.dataStart[index + 3] = func(a.dataStart[index + 3], b);
						}
					}

					for (; index < size; index++)
						c.dataStart[index] = func(a.dataStart[index], b);
				}
				else if (mode == ExecutionType::PARALLEL)
				{
					// Parallel execution on CPU
					long index = 0;

				#pragma omp parallel for shared(size, a, b, c) private(index) default(none)
					for (index = 0; index < size; ++index)
						c.dataStart[index] = func(a.dataStart[index], b);
				}
				else
				{
					message::RapidError("Mode Error", "Invalid mode for binary mapping. Must be SERIAL or PARALLEL").display();
				}
			}

			/// <summary>
			/// Apply a lambda function to a scalar and an array in the format
			/// func(scalar, array) and store the result. Both arrays
			/// must be the same size, but this in not checked when running,
			/// so it is therefore the responsibility of the user to ensure
			/// this function is called safely
			/// </summary>
			/// <typeparam name="Lambda"></typeparam>
			/// <param name="a"></param>
			/// <param name="b"></param>
			/// <param name="c"></param>
			/// <param name="mode"></param>
			/// <param name="func"></param>
			template<typename Lambda, ArrayLocation loc>
			inline static void binaryOpScalarArray(const arrayType &a, const Array<arrayType, loc> &b,
												   Array<arrayType, loc> &c, ExecutionType mode, Lambda func)
			{
				size_t size = math::prod(b.shape);

				if (mode == ExecutionType::SERIAL)
				{
					// Serial execution on CPU
					size_t index = 0;

					if (size > 3)
					{
						for (index = 0; index < size - 3; index += 4)
						{
							c.dataStart[index + 0] = func(a, b.dataStart[index + 0]);
							c.dataStart[index + 1] = func(a, b.dataStart[index + 1]);
							c.dataStart[index + 2] = func(a, b.dataStart[index + 2]);
							c.dataStart[index + 3] = func(a, b.dataStart[index + 3]);
						}
					}

					for (; index < size; index++)
						c.dataStart[index] = func(a, b.dataStart[index]);
				}
				else if (mode == ExecutionType::PARALLEL)
				{
					// Parallel execution on CPU
					long index = 0;

				#pragma omp parallel for shared(size, a, b, c) private(index) default(none)
					for (index = 0; index < size; ++index)
						c.dataStart[index] = func(a, b.dataStart[index]);
				}
				else
				{
					message::RapidError("Mode Error", "Invalid mode for binary mapping. Must be SERIAL or PARALLEL").display();
				}
			}

			/// <summary>
			/// Apply a lambda function to an array in the format
			/// func(array) and store the result
			/// </summary>
			/// <typeparam name="Lambda"></typeparam>
			/// <param name="a"></param>
			/// <param name="b"></param>
			/// <param name="mode"></param>
			/// <param name="func"></param>
			template<typename Lambda, ArrayLocation loc>
			inline static void unaryOpArray(const Array<arrayType, loc> &a, Array<arrayType, loc> &b,
											ExecutionType mode, Lambda func)
			{
				size_t size = math::prod(a.shape);

				if (mode == ExecutionType::SERIAL)
				{
					// Serial execution on CPU
					size_t index = 0;

					if (size > 3)
					{
						for (index = 0; index < size - 3; index += 4)
						{
							b.dataStart[index + 0] = func(a.dataStart[index + 0]);
							b.dataStart[index + 1] = func(a.dataStart[index + 1]);
							b.dataStart[index + 2] = func(a.dataStart[index + 2]);
							b.dataStart[index + 3] = func(a.dataStart[index + 3]);
						}
					}

					for (; index < size; index++)
						b.dataStart[index] = func(a.dataStart[index]);
				}
				else if (mode == ExecutionType::PARALLEL)
				{
					// Parallel execution on CPU
					long index = 0;

				#pragma omp parallel for shared(size, a, b) private(index) default(none)
					for (index = 0; index < size; ++index)
						b.dataStart[index] = func(a.dataStart[index]);
				}
				else
				{
					message::RapidError("Mode Error", "Invalid mode for binary mapping. Must be SERIAL or PARALLEL").display();
				}
			}

			/// <summary>
			/// Resize an array to different dimensions and return the result.
			/// The data stored in the array is copied, so an update in the
			/// result array will not trigger an update in the parent.
			/// </summary>
			/// <param name="newShape"></param>
			/// <returns></returns>
			inline Array<arrayType, location> internal_resized(const std::vector<uint64_t> &newShape) const
			{
				rapidAssert(newShape.size() == 2, "Resizing currently only supports 2D array");

				Array<arrayType, location> res(newShape);
				auto resData = res.dataStart;
				auto thisData = dataStart;

				if (location == CPU)
				{
					for (size_t i = 0; i < rapid::math::min(shape[0], newShape[0]); i++)
						memcpy(resData + i * newShape[1], thisData + i * shape[1],
							   sizeof(arrayType) * rapid::math::min(shape[1], newShape[1]));
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					for (size_t i = 0; i < rapid::math::min(shape[0], newShape[0]); i++)
						cudaSafeCall(cudaMemcpy(resData + i * newShape[1], thisData + i * shape[1],
									 sizeof(arrayType) * rapid::math::min(shape[1], newShape[1]), cudaMemcpyHostToDevice));
				}
			#endif

				return res;
			}

			/// <summary>
			/// Resize an array to different dimensions and return the result.
			/// The data stored in the array is copied, so an update in the
			/// result array will not trigger an update in the parent.
			/// </summary>
			/// <param name="newShape"></param>
			/// <returns></returns>
			inline void internal_resize(const std::vector<uint64_t> &newShape)
			{
				auto newThis = internal_resized(newShape);

				freeSelf();

				originCount = newThis.originCount;
				(*originCount)++;

				dataOrigin = newThis.dataOrigin;
				dataStart = newThis.dataStart;

				shape = newShape;
			}

		public:

			/// <summary>
			/// Default constructor
			/// </summary>
			Array()
			{
			#ifdef RAPID_CUDA
				if (!handle::handleInitialized)
					handle::createHandle();
			#endif

				isZeroDim = true;
				shape = {0};

				if (location == CPU)
					dataStart = new arrayType[1];
			#ifdef RAPID_CUDA
				else if (location == GPU)
					cudaSafeCall(cudaMalloc(&dataStart, sizeof(arrayType)));
			#endif

				dataOrigin = dataStart;

				originCount = new uint64_t;
				*originCount = 1;

			#ifdef RAPID_CUDA
				useMatrixData = false;
				matrixRows = 0;
				matrixAccess = 0;
			#endif
			}

			inline void set(const Array<arrayType, location> &other)
			{
				// Only delete data if originCount becomes zero
				freeSelf();

				isZeroDim = other.isZeroDim;
				shape = other.shape;

				dataStart = other.dataStart;
				dataOrigin = other.dataOrigin;

			#ifdef RAPID_CUDA
				useMatrixData = other.useMatrixData;
				matrixRows = other.matrixRows;
				matrixAccess = other.matrixAccess;
			#endif

				originCount = other.originCount;
				(*originCount)++;
			}

			/// <summary>
			/// Create a new array from a given shape. This allocates entirely
			/// new data, and no existing arrays are modified in any way.
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <typeparam name="type"></typeparam>
			/// <param name="arrShape"></param>
			Array(const std::vector<size_t> &arrShape)
			{
			#ifdef RAPID_CUDA
				if (!handle::handleInitialized)
					handle::createHandle();
			#endif

			#ifdef RAPID_DEBUG
				for (const auto &val : arrShape)
					if (val <= 0)
						rapidAssert(false, "Dimensions must be positive");
			#endif

				if (arrShape.empty() || math::prod(arrShape) == 0)
				{
					isZeroDim = true;
					shape = {1};

					if (location == CPU)
						dataStart = new arrayType[1];
				#ifdef RAPID_CUDA
					else
						cudaSafeCall(cudaMalloc(&dataStart, sizeof(arrayType)));
				#endif

					dataOrigin = dataStart;
					originCount = new size_t;
					*originCount = 1;
				}
				else
				{
					isZeroDim = false;
					shape = arrShape;

					if (location == CPU)
						dataStart = new arrayType[math::prod(arrShape)];
				#ifdef RAPID_CUDA
					else
						cudaSafeCall(cudaMalloc(&dataStart, sizeof(arrayType) * math::prod(arrShape)));
				#endif

					dataOrigin = dataStart;
					originCount = new size_t;
					*originCount = 1;
				}

			#ifdef RAPID_CUDA
				useMatrixData = false;
				matrixRows = 0;
				matrixAccess = 0;
			#endif
			}

			/// <summary>
			/// Create an array from an existing array. The array that is created
			/// will inherit the same data as the array it is created from, so an
			/// update in one will cause an update in the other.
			/// </summary>
			/// <param name="other"></param>
			Array(const Array<arrayType, location> &other)
			{
				isZeroDim = other.isZeroDim;
				shape = other.shape;
				dataOrigin = other.dataOrigin;
				dataStart = other.dataStart;
				originCount = other.originCount;
				(*originCount)++;

			#ifdef RAPID_CUDA
				useMatrixData = other.useMatrixData;
				matrixRows = other.matrixRows;
				matrixAccess = other.matrixAccess;
			#endif
			}

			/// <summary>
			/// Set one array equal to another and copy the memory.
			/// This means an update in one array will not trigger
			/// an update in the other
			/// </summary>
			/// <param name="other"></param>
			/// <returns></returns>
			Array<arrayType, location> &operator=(const Array<arrayType, location> &other)
			{
				rapidAssert(shape == other.shape, "Invalid shape for array setting");
				if (location == CPU)
					memcpy(dataStart, other.dataStart, math::prod(shape) * sizeof(arrayType));
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					if (useMatrixData)
					{
						for (uint64_t i = 0; i < other.shape[0]; i++)
							cudaSafeCall(cudaMemcpy(dataStart + matrixAccess + i * matrixRows, other.dataStart + i, sizeof(arrayType), cudaMemcpyDeviceToDevice));
					}
					else
					{
						cudaSafeCall(cudaMemcpy(dataStart, other.dataStart, math::prod(shape) * sizeof(arrayType), cudaMemcpyDeviceToDevice));
					}
				}
			#endif

			#ifdef RAPID_CUDA
				useMatrixData = other.useMatrixData;
				matrixRows = other.matrixRows;
				matrixAccess = other.matrixAccess;
			#endif

				return *this;
			}

			/// <summary>
			/// Set an array equal to a scalar value. This fills
			/// the array with the value.
			/// </summary>
			/// <param name="other"></param>
			/// <returns></returns>
			Array<arrayType, location> &operator=(const arrayType &other)
			{
				fill(other);
				return *this;
			}

			/// <summary>
			/// Create an array from the provided data, without creating a
			/// temporary one first. This fixes memory leaks and is intended
			/// for internal use only.
			/// </summary>
			/// <param name="arrDims"></param>
			/// <param name="newDataOrigin"></param>
			/// <param name="dataStart"></param>
			/// <param name="originCount"></param>
			/// <param name="isZeroDim"></param>
			/// <returns></returns>
			static inline Array<arrayType, location> fromData(const std::vector<size_t> &arrDims,
															  arrayType *newDataOrigin, arrayType *dataStart,
															  size_t *originCount, bool isZeroDim)
			{
			#ifdef RAPID_CUDA
				if (!handle::handleInitialized)
					handle::createHandle();
			#endif

				Array<arrayType, location> res;
				res.isZeroDim = isZeroDim;
				res.shape = arrDims;
				res.dataOrigin = newDataOrigin;
				res.dataStart = dataStart;
				res.originCount = originCount;
				return res;
			}

			/// <summary>
			/// Create a new array from an initializer_list. This supports creating
			/// arrays of up to 20-dimensions via nested initializer lists
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="data"></param>
			/// <returns></returns>
			template<typename t>
			static inline Array<arrayType, location> fromData(const std::initializer_list<t> &data)
			{
			#ifdef RAPID_CUDA
				if (!handle::handleInitialized)
					handle::createHandle();
			#endif

				auto res = Array<arrayType, location>({data.size()});
				std::vector<arrayType> values;
				for (const auto &val : data)
					values.emplace_back(val);

				if (location == CPU)
					memcpy(res.dataStart, values.data(), sizeof(arrayType) * data.size());
			#ifdef RAPID_CUDA
				else if (location == GPU)
					cudaSafeCall(cudaMemcpy(res.dataStart, values.data(), sizeof(arrayType) * data.size(), cudaMemcpyHostToDevice));
			#endif

				return res;
			}


		#define imp_temp template<typename t>
		#define imp_func_def(x) static inline Array<arrayType, location> fromData(const x &data)
		#define imp_func_body	auto res = Array<arrayType, location>(imp::extractShape(data)); \
							uint64_t index = 0; \
							for (const auto &val : data) res[index++] = Array<arrayType, location>::fromData(val); \
							return res;
		#define L std::initializer_list

			// Up to 20-dimensional array setting from data
			imp_temp imp_func_def(L<L<t>>)
			{
				auto res = Array<arrayType, location>(imp::extractShape(data));

				if (location == CPU)
				{
					uint64_t index = 0;
					for (const auto &val : data)
						res[index++] = Array<arrayType, location>::fromData(val);
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					uint64_t index = 0;
					for (const auto &val : data)
						res.pseudoBrackets(index++) = Array<arrayType, location>::fromData(val);

					arrayType *resData;
					cudaSafeCall(cudaMalloc(&resData, sizeof(arrayType) * math::prod(res.shape)));

					cuda::rowToColumnOrdering(res.shape[0], res.shape[1], res.dataStart, resData);
					cudaSafeCall(cudaDeviceSynchronize());
					cudaSafeCall(cudaMemcpy(res.dataStart, resData, sizeof(arrayType) * math::prod(res.shape), cudaMemcpyDeviceToDevice));
					cudaSafeCall(cudaFree(resData));
				}
			#endif

				return res;
			}

			imp_temp imp_func_def(L<L<L<t>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<t>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<t>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<t>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<t>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<t>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<t>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>>>>)
			{
				imp_func_body
			}

		#undef imp_temp
		#undef imp_func_def
		#undef imp_func_body
		#undef L

			~Array()
			{
				freeSelf();
			}

			inline void freeSelf()
			{
				// Only delete data if originCount becomes zero
				(*originCount)--;

				if ((*originCount) == 0)
				{
					if (location == CPU)
						delete[] dataOrigin;
				#ifdef RAPID_CUDA
					else
						cudaSafeCall(cudaFree(dataOrigin));
				#endif
					delete originCount;
				}
			}

			/// <summary>
			/// Cast a zero-dimensional array to a scalar value
			/// </summary>
			/// <typeparam name="t"></typeparam>
			template<typename t>
			inline operator t() const
			{
				if (!isZeroDim)
					rapidAssert(isZeroDim, "Cannot cast multidimensional array to scalar value");
				if (location == CPU)
					return (t) (dataStart[0]);

			#ifdef RAPID_CUDA
				if (location == GPU)
				{
					auto resPtr = new arrayType;
					cudaSafeCall(cudaMemcpy(resPtr, dataStart, sizeof(arrayType), cudaMemcpyDeviceToHost));
					auto res = (t) (*resPtr);
					delete resPtr;
					return res;
				}
			#endif
			}

			Array<arrayType, location> pseudoBrackets(const size_t &index) const
			{
				rapidAssert(index < shape[0], "Index out of range for array subscript");

				(*originCount)++;

				if (shape.size() == 1)
				{
					return Array<arrayType, location>::fromData({1}, dataOrigin, dataStart + utils::ndToScalar({index}, shape),
																originCount, true);
				}

				if (location == CPU)
				{
					std::vector<size_t> resShape(shape.begin() + 1, shape.end());
					return Array<arrayType, location>::fromData(resShape, dataOrigin, dataStart + utils::ndToScalar({index}, shape),
																originCount, isZeroDim);
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					std::vector<size_t> resShape(shape.begin() + 1, shape.end());
					return Array<arrayType, location>::fromData(resShape, dataOrigin, dataStart + utils::ndToScalar({index}, shape),
																originCount, isZeroDim);
				}
			#endif
			}

			/// <summary>
			/// Access a subarray or value of an array. The result is linked
			/// to the parent array, so an update in one will trigger an update
			/// in the other.
			/// </summary>
			/// <param name="index"></param>
			/// <returns></returns>
			Array<arrayType, location> operator[](const size_t &index) const
			{
				rapidAssert(index < shape[0], "Index out of range for array subscript");

				(*originCount)++;

				if (shape.size() == 1)
				{
				#ifdef RAPID_CUDA
					if (useMatrixData)
					{
						return Array<arrayType, location>::fromData({1}, dataOrigin, dataStart + matrixAccess + index * matrixRows,
																	originCount, true);
					}
					else
					{
						return Array<arrayType, location>::fromData({1}, dataOrigin, dataStart + utils::ndToScalar({index}, shape),
																	originCount, true);
					}
				#endif

					return Array<arrayType, location>::fromData({1}, dataOrigin, dataStart + utils::ndToScalar({index}, shape),
																originCount, true);
				}

				if (location == CPU)
				{
					std::vector<size_t> resShape(shape.begin() + 1, shape.end());
					return Array<arrayType, location>::fromData(resShape, dataOrigin, dataStart + utils::ndToScalar({index}, shape),
																originCount, isZeroDim);
				}
			#ifdef RAPID_CUDA
				else if (shape.size() == 2 && location == GPU)
				{
					std::vector<size_t> resShape(shape.begin() + 1, shape.end());
					auto res = Array<arrayType, location>::fromData(resShape, dataOrigin, dataStart, // + utils::ndToScalar({index}, shape), // dataStart,
																	originCount, isZeroDim);

					res.useMatrixData = true;
					res.matrixRows = shape[0];
					res.matrixAccess = index;

					return res;
				}
				else
				{
					std::vector<size_t> resShape(shape.begin() + 1, shape.end());
					return Array<arrayType, location>::fromData(resShape, dataOrigin, dataStart + utils::ndToScalar({index}, shape),
																originCount, isZeroDim);
				}
			#endif
			}

			/// <summary>
			/// Directly access an individual value in an array. This does
			/// not allow for changing the value, but is much faster than
			/// accessing it via repeated subscript operations
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="index"></param>
			/// <returns></returns>
			template<typename t, ArrayLocation loc>
			inline arrayType accessVal(const std::initializer_list<t> &index) const
			{
				rapidAssert(index.size() == shape.size(), "Invalid number of dimensions to access");
			#ifdef RAPID_DEBUG
				for (size_t i = 0; i < index.size(); i++)
				{
					if (*(index.begin() + i) < 0 || *(index.begin() + i) >= shape[i])
						message::RapidError("Index Error", "Index out of range or negative").display();
				}
			#endif

				return dataStart[utils::ndToScalar(index, shape)];
			}

			/// <summary>
			/// Set a scalar value in an array from a given
			/// index location
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="index"></param>
			/// <param name="val"></param>
			template<typename t>
			inline void setVal(const std::initializer_list<t> &index, const arrayType &val) const
			{
				rapidAssert(index.size() == shape.size(), "Invalid number of dimensions to access");
			#ifdef RAPID_DEBUG
				for (size_t i = 0; i < index.size(); i++)
				{
					if (*(index.begin() + i) < 0 || *(index.begin() + i) >= shape[i])
						message::RapidError("Index Error", "Index out of range or negative");
				}
			#endif

				if (location == CPU)
					dataStart[utils::ndToScalar(index, shape)] = val;
			#ifdef RAPID_CUDA
				else if (location == GPU)
					cudaSafeCall(cudaMemcpy(dataStart + utils::ndToScalar(index, shape), &val, sizeof(arrayType), cudaMemcpyHostToDevice));
			#endif
			}

			inline Array<arrayType, location> operator-() const
			{
				auto res = Array<arrayType, location>(shape);
				Array<arrayType, location>::unaryOpArray(*this, res,
														 math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
														 [](arrayType x)
				{
					return -x;
				});
				return res;
			}

			/// <summary>
			/// Array add Array
			/// </summary>
			/// <param name="other"></param>
			/// <returns></returns>
			inline Array<arrayType, location> operator+(const Array<arrayType, location> &other) const
			{
				rapidAssert(shape == other.shape, "Shapes must be equal to perform array addition");

				if (location == CPU)
				{
					auto res = Array<arrayType, location>(shape);
					Array<arrayType, location>::binaryOpArrayArray(*this, other, res,
																   math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																   [](arrayType x, arrayType y)
					{
						return x + y;
					});
					return res;
				}

			#ifdef RAPID_CUDA
				auto res = Array<arrayType, location>(shape);
				cuda::add_array_array(math::prod(shape), dataStart, other.dataStart, res.dataStart);
				return res;
			#endif
			}

			/// <summary>
			/// Array sub Array
			/// </summary>
			/// <param name="other"></param>
			/// <returns></returns>
			inline Array<arrayType, location> operator-(const Array<arrayType, location> &other) const
			{
				rapidAssert(shape == other.shape, "Shapes must be equal to perform array addition");

				if (location == CPU)
				{
					auto res = Array<arrayType, location>(shape);
					Array<arrayType, location>::binaryOpArrayArray(*this, other, res,
																   math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																   [](arrayType x, arrayType y)
					{
						return x - y;
					});
					return res;
				}

			#ifdef RAPID_CUDA
				auto res = Array<arrayType, location>(shape);
				cuda::sub_array_array(math::prod(shape), dataStart, other.dataStart, res.dataStart);
				return res;
			#endif
			}

			/// <summary>
			/// Array mul Array
			/// </summary>
			/// <param name="other"></param>
			/// <returns></returns>
			inline Array<arrayType, location> operator*(const Array<arrayType, location> &other) const
			{
				rapidAssert(shape == other.shape, "Shapes must be equal to perform array addition");

				if (location == CPU)
				{
					auto res = Array<arrayType, location>(shape);
					Array<arrayType, location>::binaryOpArrayArray(*this, other, res,
																   math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																   [](arrayType x, arrayType y)
					{
						return x * y;
					});
					return res;
				}

			#ifdef RAPID_CUDA
				auto res = Array<arrayType, location>(shape);
				cuda::mul_array_array(math::prod(shape), dataStart, other.dataStart, res.dataStart);
				return res;
			#endif
			}

			/// <summary>
			/// Array div Array
			/// </summary>
			/// <param name="other"></param>
			/// <returns></returns>
			inline Array<arrayType, location> operator/(const Array<arrayType, location> &other) const
			{
				rapidAssert(shape == other.shape, "Shapes must be equal to perform array addition");

				if (location == CPU)
				{
					auto res = Array<arrayType, location>(shape);
					Array<arrayType, location>::binaryOpArrayArray(*this, other, res,
																   math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																   [](arrayType x, arrayType y)
					{
						return x / y;
					});
					return res;
				}

			#ifdef RAPID_CUDA
				auto res = Array<arrayType, location>(shape);
				cuda::div_array_array(math::prod(shape), dataStart, other.dataStart, res.dataStart);
				return res;
			#endif
			}

			/// <summary>
			/// Array add Scalar
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="other"></param>
			/// <returns></returns>
			template<typename t>
			inline Array<arrayType, location> operator+(const t &other) const
			{
				if (location == CPU)
				{
					auto res = Array<arrayType, location>(shape);
					Array <arrayType, location> ::binaryOpArrayScalar(*this, (arrayType) other,
																	  res, math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	  [](arrayType x, arrayType y)
					{
						return x + y;
					});
					return res;
				}

			#ifdef RAPID_CUDA
				auto res = Array<arrayType, location>(shape);
				cuda::add_array_scalar(math::prod(shape), dataStart, (arrayType) other, res.dataStart);
				return res;
			#endif
			}

			/// <summary>
			/// Array sub Scalar
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="other"></param>
			/// <returns></returns>
			template<typename t>
			inline Array<arrayType, location> operator-(const t &other) const
			{
				if (location == CPU)
				{
					auto res = Array<arrayType, location>(shape);
					Array<arrayType, location>::binaryOpArrayScalar(*this, (arrayType) other, res,
																	math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	[](arrayType x, arrayType y)
					{
						return x - y;
					});
					return res;
				}

			#ifdef RAPID_CUDA
				auto res = Array<arrayType, location>(shape);
				cuda::sub_array_scalar(math::prod(shape), dataStart, (arrayType) other, res.dataStart);
				return res;
			#endif
			}

			/// <summary>
			/// Array mul Scalar
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="other"></param>
			/// <returns></returns>
			template<typename t>
			inline Array<arrayType, location> operator*(const t &other) const
			{
				if (location == CPU)
				{
					auto res = Array<arrayType, location>(shape);
					Array<arrayType, location>::binaryOpArrayScalar(*this, (arrayType) other, res,
																	math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	[](arrayType x, arrayType y)
					{
						return x * y;
					});
					return res;
				}

			#ifdef RAPID_CUDA
				auto res = Array<arrayType, location>(shape);
				cuda::mul_array_scalar(math::prod(shape), dataStart, (arrayType) other, res.dataStart);
				return res;
			#endif
			}

			/// <summary>
			/// Array div Scalar
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="other"></param>
			/// <returns></returns>
			template<typename t>
			inline Array<arrayType, location> operator/(const t &other) const
			{
				if (location == CPU)
				{
					auto res = Array<arrayType, location>(shape);
					Array<arrayType, location>::binaryOpArrayScalar(*this, (arrayType) other, res,
																	math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	[](arrayType x, arrayType y)
					{
						return x / y;
					});
					return res;
				}

			#ifdef RAPID_CUDA
				auto res = Array<arrayType, location>(shape);
				cuda::div_array_scalar(math::prod(shape), dataStart, (arrayType) other, res.dataStart);
				return res;
			#endif
			}

			inline Array<arrayType, location> &operator+=(const Array<arrayType, location> &other)
			{
				rapidAssert(shape == other.shape, "Shapes must be equal to perform array addition");

				if (location == CPU)
				{
					Array<arrayType, location>::binaryOpArrayArray(*this, other, *this,
																   math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																   [](arrayType x, arrayType y)
					{
						return x + y;
					});
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cuda::add_array_array(math::prod(shape), dataStart, other.dataStart, dataStart);
				}
			#endif

				return *this;
			}

			inline Array<arrayType, location> &operator-=(const Array<arrayType, location> &other)
			{
				rapidAssert(shape == other.shape, "Shapes must be equal to perform array addition");

				if (location == CPU)
				{
					Array<arrayType, location>::binaryOpArrayArray(*this, other, *this,
																   math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																   [](arrayType x, arrayType y)
					{
						return x - y;
					});
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cuda::sub_array_array(math::prod(shape), dataStart, other.dataStart, dataStart);
				}
			#endif

				return *this;
			}

			inline Array<arrayType, location> &operator*=(const Array<arrayType, location> &other)
			{
				rapidAssert(shape == other.shape, "Shapes must be equal to perform array addition");

				if (location == CPU)
				{
					Array<arrayType, location>::binaryOpArrayArray(*this, other, *this,
																   math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																   [](arrayType x, arrayType y)
					{
						return x * y;
					});
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cuda::mul_array_array(math::prod(shape), dataStart, other.dataStart, dataStart);
				}
			#endif

				return *this;
			}

			inline Array<arrayType, location> &operator/=(const Array<arrayType, location> &other)
			{
				rapidAssert(shape == other.shape, "Shapes must be equal to perform array addition");

				if (location == CPU)
				{
					Array<arrayType, location>::binaryOpArrayArray(*this, other, *this,
																   math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																   [](arrayType x, arrayType y)
					{
						return x / y;
					});
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cuda::div_array_array(math::prod(shape), dataStart, other.dataStart, dataStart);
				}
			#endif

				return *this;
			}

			inline Array<arrayType, location> &operator+=(const arrayType &other)
			{
				if (location == CPU)
				{
					Array<arrayType, location>::binaryOpArrayScalar(*this, other, *this,
																	math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	[](arrayType x, arrayType y)
					{
						return x + y;
					});
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cuda::add_array_scalar(math::prod(shape), dataStart, other, dataStart);
				}
			#endif

				return *this;
			}

			inline Array<arrayType, location> &operator-=(const arrayType &other)
			{
				if (location == CPU)
				{
					Array<arrayType, location>::binaryOpArrayScalar(*this, other, *this,
																	math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	[](arrayType x, arrayType y)
					{
						return x - y;
					});
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cuda::sub_array_scalar(math::prod(shape), dataStart, other, dataStart);
				}
			#endif

				return *this;
			}

			inline Array<arrayType, location> &operator*=(const arrayType &other)
			{
				if (location == CPU)
				{
					Array<arrayType, location>::binaryOpArrayScalar(*this, other, *this,
																	math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	[](arrayType x, arrayType y)
					{
						return x * y;
					});
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cuda::mul_array_scalar(math::prod(shape), dataStart, other, dataStart);
				}
			#endif

				return *this;
			}

			inline Array<arrayType, location> &operator/=(const arrayType &other)
			{
				if (location == CPU)
				{
					Array<arrayType, location>::binaryOpArrayScalar(*this, other, *this,
																	math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	[](arrayType x, arrayType y)
					{
						return x / y;
					});
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cuda::div_array_scalar(math::prod(shape), dataStart, other, dataStart);
				}
			#endif

				return *this;
			}

			/// <summary>
			/// Fill an array with a scalar value
			/// </summary>
			/// <param name="val"></param>
			inline void fill(const arrayType &val)
			{
				if (location == CPU)
				{
					Array<arrayType, location>::unaryOpArray(*this, *this,
															 math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
															 [=](arrayType x)
					{
						return val;
					});
				}
			#ifdef RAPID_CUDA
				else
				{
					cuda::fill(math::prod(shape), dataStart, val);
				}
			#endif
			}

			/// <summary>
			/// Calculate the dot math::product with another array. If the
			/// arrays are single-dimensional vectors, the vector math::product
			/// is used and a scalar value is returned. If the arrays are
			/// matrices, the matrix math::product is calculated. Otherwise, the
			/// dot math::product of the final two dimensions of the array are
			/// calculated.
			/// </summary>
			/// <param name="other"></param>
			/// <returns></returns>
			inline Array<arrayType, location> dot(const Array<arrayType, location> &other) const
			{
				rapidAssert(shape.size() == other.shape.size(), "Invalid number of dimensions for array dot math::product");
				uint64_t dims = shape.size();

				if (location == CPU)
				{
				#ifndef RAPID_NO_BLAS
					switch (dims)
					{
						case 1:
							{
								rapidAssert(shape[0] == other.shape[0], "Invalid shape for array math::product");
								rapidAssert(isZeroDim == other.isZeroDim, "Invalid value for array math::product");

								Array<arrayType, location> res(shape);
								res.isZeroDim = true;
								res.dataStart[0] = imp::rapid_dot(shape[0], dataStart, other.dataStart);

								return res;
							}
						case 2:
							{
								rapidAssert(shape[1] == other.shape[0], "Columns of A must match rows of B for dot math::product");

								Array<arrayType, location> res({shape[0], other.shape[1]});

								const size_t M = shape[0];
								const size_t N = shape[1];
								const size_t K = other.shape[1];

								const arrayType *__restrict a = dataStart;
								const arrayType *__restrict b = other.dataStart;
								arrayType *__restrict c = res.dataStart;

								imp::rapid_gemm(M, N, K, a, b, c);

								return res;
							}
						default:
							{
								std::vector<uint64_t> resShape = shape;
								resShape[resShape.size() - 2] = shape[shape.size() - 2];
								resShape[resShape.size() - 1] = other.shape[other.shape.size() - 1];
								Array<arrayType, location> res(resShape);

								for (uint64_t i = 0; i < shape[0]; i++)
								{
									res[i] = (operator[](i).dot(other[i]));
								}

								return res;
							}
					}
				#else
					switch (dims)
					{
						case 1:
							{
								rapidAssert(shape[0] == other.shape[0], "Invalid shape for array math::product");
								rapidAssert(isZeroDim == other.isZeroDim, "Invalid value for array math::product");

								Array<arrayType, location> res({1});
								res.isZeroDim = true;
								res.dataStart[0] = 0;

								for (uint64_t i = 0; i < shape[0]; i++)
									res.dataStart[0] += dataStart[i] * other.dataStart[i];

								return res;
							}
						case 2:
							{
								rapidAssert(shape[1] == other.shape[0], "Columns of A must match rows of B for dot math::product");
								uint64_t mode;
								uint64_t size = shape[0] * shape[1] * other.shape[1];

								if (size < 8000) mode = 0;
								else if (size < 64000000) mode = 1;
							#ifndef RAPID_NO_AMP
								else mode = 2;
							#else
								else mode = 1;
							#endif

								Array<arrayType, location> res({shape[0], other.shape[1]});

								if (mode == 0)
								{
									// Serial

									size_t M = shape[0];
									size_t N = shape[1];
									size_t K = other.shape[1];

									const arrayType *__restrict a = dataStart;
									const arrayType *__restrict b = other.dataStart;
									arrayType *__restrict c = res.dataStart;

									size_t i, j, k;
									arrayType tmp;

									for (i = 0; i < M; ++i)
									{
										for (j = 0; j < K; ++j)
										{
											tmp = 0;

											for (k = 0; k < N; ++k)
												tmp += a[k + i * N] * b[j + k * K];

											c[j + i * K] = tmp;
										}
									}
								}
								else if (mode == 1)
								{
									// Parallel

									auto M = (long long) shape[0];
									auto N = (long long) shape[1];
									auto K = (long long) other.shape[1];

									const arrayType *__restrict a = dataStart;
									const arrayType *__restrict b = other.dataStart;
									arrayType *__restrict c = res.dataStart;

									long long i, j, k;
									arrayType tmp;

								#pragma omp parallel for shared(M, N, K, a, b, c) private(i, j, k, tmp) default(none)
									for (i = 0; i < M; ++i)
									{
										for (j = 0; j < K; ++j)
										{
											tmp = 0;

											for (k = 0; k < N; ++k)
												tmp += a[k + i * N] * b[j + k * K];

											c[j + i * K] = tmp;
										}
									}
								}
							#ifndef RAPID_NO_AMP
								else if (mode == 2)
								{
									// Massive parallel

									// Tile size
									static const int TS = 32;

									const auto resizedThis = internal_resized({rapid::roundUp(shape[0], (size_t) TS),
																			  rapid::roundUp(shape[1], (size_t) TS)});
									const auto resizedOther = internal_resized({rapid::roundUp(other.shape[0], (size_t) TS),
																			   rapid::roundUp(other.shape[1], (size_t) TS)});
									res.internal_resize({rapid::roundUp(shape[0], (size_t) TS),
														rapid::roundUp(other.shape[1], (size_t) TS)});

									auto M = (unsigned int) resizedThis.shape[0];
									auto N = (unsigned int) resizedThis.shape[1];
									auto K = (unsigned int) res.shape[1];

									array_view<const arrayType, 2> a(M, N, resizedThis.dataStart);
									array_view<const arrayType, 2> b(N, K, resizedOther.dataStart);
									array_view<arrayType, 2> math::product(M, K, res.dataStart);

									parallel_for_each(math::product.extent.tile<TS, TS>(), [=](tiled_index<TS, TS> t_idx) restrict(amp)
									{
										// Get the location of the thread relative to the tile (row, col)
										// and the entire array_view (rowGlobal, colGlobal).
										const int row = t_idx.local[0];
										const int col = t_idx.local[1];
										const int rowGlobal = t_idx.global[0];
										const int colGlobal = t_idx.global[1];
										arrayType sum = 0;

										for (int i = 0; i < M; i += TS)
										{
											tile_static arrayType locA[TS][TS];
											tile_static arrayType locB[TS][TS];
											locA[row][col] = a(rowGlobal, col + i);
											locB[row][col] = b(row + i, colGlobal);

											t_idx.barrier.wait();

											for (int k = 0; k < TS; k++)
												sum += locA[row][k] * locB[k][col];

											t_idx.barrier.wait();
										}

										math::product[t_idx.global] = sum;
									});

									math::product.synchronize();

									res.internal_resize({shape[0], other.shape[1]});
								}
							#endif

								return res;
							}
						default:
							{
								std::vector<uint64_t> resShape = shape;
								resShape[resShape.size() - 2] = shape[shape.size() - 2];
								resShape[resShape.size() - 1] = other.shape[other.shape.size() - 1];
								Array<arrayType, location> res(resShape);

								for (uint64_t i = 0; i < shape[0]; i++)
								{
									res[i] = (operator[](i).dot(other[i]));
								}

								return res;
							}
					}
				#endif
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					switch (dims)
					{
						case 1:
							{
								rapidAssert(shape[0] == other.shape[0], "Invalid shape for array math::product");
								rapidAssert(isZeroDim == other.isZeroDim, "Invalid value for array math::product");

								Array<arrayType, location> res({1});
								res.isZeroDim = true;

								cuda::dot(1, shape[0], other.shape[0], dataStart, other.dataStart, res.dataStart);

								return res;
							}
						case 2:
							{
								rapidAssert(shape[1] == other.shape[0], "Columns of A must match rows of B for dot math::product");
								uint64_t size = shape[0] * shape[1] * other.shape[1];

								Array<arrayType, location> res({shape[0], other.shape[1]});

								uint64_t m = shape[0];
								uint64_t n = shape[1];
								uint64_t k = other.shape[1];

								arrayType dotAlpha = 1;
								arrayType dotBeta = 0;

								cudaSafeCall(cudaDeviceSynchronize());
								cuda::gemm(handle::handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &dotAlpha, dataStart, m, other.dataStart, n, &dotBeta, res.dataStart, m);

								return res;
							}
						default:
							{
								std::vector<uint64_t> resShape = shape;
								resShape[resShape.size() - 2] = shape[shape.size() - 2];
								resShape[resShape.size() - 1] = other.shape[other.shape.size() - 1];
								Array<arrayType, location> res(resShape);

								for (uint64_t i = 0; i < shape[0]; i++)
									res[i] = (operator[](i).dot(other[i]));

								return res;
							}
					}
				}
			#endif
			}

			/// <summary>
			/// Transpose an array and return the result. If the
			/// array is one dimensional, a vector is returned. The
			/// order in which the transpose occurs can be set with
			/// the "axes" parameter
			/// </summary>
			/// <param name="axes"></param>
			/// <returns></returns>
			inline Array<arrayType, location> transposed(const std::vector<uint64_t> &axes = std::vector<uint64_t>(), bool dataOnly = false) const
			{
				if (!axes.empty())
				{
					if (axes.size() != shape.size())
						message::RapidError("Transpose Error", "Invalid number of axes for array transpose").display();
					for (uint64_t i = 0; i < axes.size(); i++)
						if (std::count(axes.begin(), axes.end(), i) != 1)
							message::RapidError("Transpose Error", "Dimension does not appear only once").display();
				}

				std::vector<uint64_t> newDims;

				if (dataOnly)
				{
					newDims = std::vector<uint64_t>(shape.begin(), shape.end());
				}
				else
				{
					newDims = std::vector<uint64_t>(shape.size());
					if (axes.empty())
						for (uint64_t i = 0; i < shape.size(); i++)
							newDims[i] = shape[shape.size() - i - 1];
					else
						for (uint64_t i = 0; i < shape.size(); i++)
							newDims[i] = shape[axes[i]];
				}

				if (location == CPU)
				{
					// Edge case for 1D array
					if (shape.size() == 1 || (axes.size() == 1 && axes[0] == 0))
					{
						auto res = Array<arrayType, location>(newDims);
						memcpy(res.dataStart, dataStart, sizeof(arrayType) * math::prod(newDims));
						return res;
					}

					if (shape.size() == 2)
					{
						auto res = Array<arrayType, location>(newDims);

						uint64_t rows = shape[0];
						uint64_t cols = shape[1];

						if (rows * cols < 62500)
						{
							for (uint64_t i = 0; i < rows; i++)
							{
								for (uint64_t j = 0; j < cols; j++)
									res.dataStart[i + j * rows] = dataStart[j + i * cols];
							}
						}
						else
						{
							int64_t i = 0, j = 0;
							const arrayType *__restrict thisData = dataStart;
							arrayType *__restrict resData = res.dataStart;
							auto minCols = rapid::math::max(cols, 3) - 3;

						#pragma omp parallel for private(i, j) shared(resData, thisData, minCols) default(none) num_threads(8)
							for (i = 0; i < rows; i++)
							{
								for (j = 0; j < minCols; j++)
								{
									int64_t p1 = i + j * rows;
									int64_t p2 = j + i * cols;

									resData[p1 + 0] = thisData[p2 + 0];
									resData[p1 + 1] = thisData[p2 + 1];
									resData[p1 + 2] = thisData[p2 + 2];
									resData[p1 + 3] = thisData[p2 + 3];
								}

								for (; j < cols; j++)
									resData[i + j * rows] = thisData[+i * cols];
							}
						}

						return res;
					}

					auto res = Array<arrayType, location>(newDims);

					std::vector<uint64_t> indices(shape.size(), 0);
					std::vector<uint64_t> indicesRes(shape.size(), 0);

					if (math::prod(shape) < 62000)
					{
						for (uint64_t i = 0; i < math::prod(shape); i++)
						{
							if (axes.empty())
								for (uint64_t j = 0; j < shape.size(); j++)
									indicesRes[j] = indices[shape.size() - j - 1];
							else
								for (uint64_t j = 0; j < shape.size(); j++)
									indicesRes[j] = indices[axes[j]];

							res.dataStart[imp::dimsToIndex(newDims, indicesRes)] = dataStart[imp::dimsToIndex(shape, indices)];

							indices[shape.size() - 1]++;
							uint64_t index = shape.size() - 1;

							while (indices[index] >= shape[index] && index > 0)
							{
								indices[index] = 0;
								index--;
								indices[index]++;
							}
						}
					}
					else
					{
						auto tmpShape = shape;
						for (int64_t i = 0; i < math::prod(tmpShape); i++)
						{
							if (axes.empty())
								for (int64_t j = 0; j < tmpShape.size(); j++)
									indicesRes[j] = indices[tmpShape.size() - j - 1];
							else
								for (int64_t j = 0; j < tmpShape.size(); j++)
									indicesRes[j] = indices[axes[j]];

							res.dataStart[imp::dimsToIndex(newDims, indicesRes)] = dataStart[imp::dimsToIndex(tmpShape, indices)];

							indices[tmpShape.size() - 1]++;
							int64_t index = tmpShape.size() - 1;

							while (indices[index] >= tmpShape[index] && index > 0)
							{
								indices[index] = 0;
								index--;
								indices[index]++;
							}
						}
					}

					return res;
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					if (shape.size() == 2)
					{
						uint64_t m = shape[1];
						uint64_t n = shape[0];

						auto res = Array<arrayType, location>(newDims);

						arrayType alpha = 1;
						arrayType beta = 0;

						cudaSafeCall(cudaDeviceSynchronize());
						cublasSafeCall(cublasSgeam(handle::handle,
									   CUBLAS_OP_T, CUBLAS_OP_T,
									   m, n,
									   &alpha,
									   dataStart, n,
									   &beta,
									   dataStart, n,
									   res.dataStart, m));

						return res;
					}
				}
			#endif

				return Array<arrayType, location>({0, 0});
			}

		#define AUTO ((uint64_t) -1)

			/// <summary>
			/// Resize an array and return the result. The resulting data
			/// is linked to the parent data, so updating values will
			/// trigger an update in the parent/child array
			/// </summary>
			/// <param name="newShape"></param>
			/// <returns></returns>
			inline Array<arrayType> resized(const std::vector<uint64_t> &newShape) const
			{
				auto tmpNewShape = std::vector<uint64_t>(newShape.size(), 1);
				uint64_t undefined = (uint64_t) -1;

				for (uint64_t i = 0; i < newShape.size(); i++)
				{
					if (newShape[i] == AUTO)
					{
						if (undefined != AUTO)
							message::RapidError("Resize Error", "Only one AUTO dimension is allowed when resizing").display();
						else
							undefined = i;
					}
					else
					{
						tmpNewShape[i] = newShape[i];
					}
				}

				if (undefined != AUTO)
					tmpNewShape[undefined] = math::prod(shape) / math::prod(tmpNewShape);

				if (math::prod(tmpNewShape) != math::prod(shape))
					message::RapidError("Invalid Shape", "Invalid reshape size. Number of elements differ").display();

				bool zeroDim = false;

				if (isZeroDim && tmpNewShape.size() == 1)
					zeroDim = true;
				else
					zeroDim = false;

				(*originCount)++;
				auto res = Array<arrayType>::fromData(tmpNewShape, dataOrigin, dataStart, originCount, zeroDim);

				return res;
			}

			/// <summary>
			/// Resize an array inplace
			/// </summary>
			/// <param name="newShape"></param>
			inline void resize(const std::vector<uint64_t> &newShape)
			{
				auto tmpNewShape = std::vector<uint64_t>(newShape.size(), 1);
				uint64_t undefined = (uint64_t) -1;

				for (uint64_t i = 0; i < newShape.size(); i++)
				{
					if (newShape[i] == AUTO)
					{
						if (undefined != AUTO)
							message::RapidError("Resize Error", "Only one AUTO dimension is allowed when resizing").display();
						else
							undefined = i;
					}
					else
					{
						tmpNewShape[i] = newShape[i];
					}
				}

				if (undefined != AUTO)
					tmpNewShape[undefined] = math::prod(shape) / math::prod(tmpNewShape);

				if (math::prod(tmpNewShape) != math::prod(shape))
					message::RapidError("Invalid Shape", "Invalid reshape size. Number of elements differ").display();

				if (isZeroDim && tmpNewShape.size() == 1)
					isZeroDim = true;
				else
					isZeroDim = false;

				shape = tmpNewShape;
			}

			template<typename Lambda>
			inline Array<arrayType, location> mapped(Lambda func) const
			{
				auto res = Array<arrayType, location>(shape);
				auto size = math::prod(shape);
				auto mode = ExecutionType::SERIAL;

				if (size > 10000) mode = ExecutionType::PARALLEL;

				unaryOpArray(*this, res, mode, func);
				return res;
			}

			/// <summary>
			/// Create an exact copy of an array. The resulting array
			/// is not linked to the parent in any way, so an 
			/// </summary>
			/// <returns></returns>
			inline Array<arrayType, location> copy() const
			{
				Array<arrayType, location> res;
				res.isZeroDim = isZeroDim;
				res.shape = shape;
				res.originCount = new size_t;
				*(res.originCount) = 1;

				if (location == CPU)
				{
					res.dataStart = new arrayType[math::prod(shape)];
					memcpy(res.dataStart, dataStart, sizeof(arrayType) * math::prod(shape));
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cudaSafeCall(cudaMalloc(&(res.dataStart), sizeof(arrayType) * math::prod(shape)));
					cudaMemcpy(res.dataStart, dataStart, sizeof(arrayType) * math::prod(shape), cudaMemcpyDeviceToDevice);
				}
			#endif
				res.dataOrigin = res.dataStart;

				return res;
			}

			/// <summary>
			/// Get a string representation of an array
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <returns></returns>
			std::string toString(uint64_t startDepth = 0) const;
		};

		template<typename t, ArrayLocation loc>
		std::ostream &operator<<(std::ostream &os, const Array<t, loc> &arr)
		{
			return os << arr.toString();
		}

		/// <summary>
		/// Create a new array of the same size and dimensions as
		/// another array, but fill it with zeros.
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> zerosLike(const Array<t, loc> &other)
		{
			auto res = Array<t, loc>(other.shape);
			res.fill((t) 0);
			return res;
		}

		/// <summary>
		/// Create a new array of the same size and dimensions as
		/// another array, but fill it with ones
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> onesLike(const Array<t, loc> &other)
		{
			auto res = Array<t, loc>(other.shape);
			res.fill((t) 1);
			return res;
		}

		/// <summary>
		/// Reverse addition
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="val"></param>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> operator+(t val, const Array<t, loc> &other)
		{
			auto res = Array<t, loc>(other.shape);

			if (loc == CPU)
			{
				Array<t, loc>::binaryOpScalarArray(val, other, res,
												   math::prod(other.shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
												   [](t x, t y)
				{
					return x + y;
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto res = Array<t, loc>(other.shape);
				cuda::add_scalar_array(math::prod(other.shape), val, other.dataStart, res.dataStart);
				return res;
			}
		#endif
			return res;
		}

		/// <summary>
		/// Reverse subtraction
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="val"></param>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> operator-(t val, const Array<t, loc> &other)
		{
			auto res = Array<t, loc>(other.shape);

			if (loc == CPU)
			{
				Array<t, loc>::binaryOpScalarArray(val, other, res,
												   math::prod(other.shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
												   [](t x, t y)
				{
					return x - y;
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto res = Array<t, loc>(other.shape);
				cuda::sub_scalar_array(math::prod(other.shape), val, other.dataStart, res.dataStart);
				return res;
			}
		#endif

			return res;
		}

		/// <summary>
		/// Reverse multiplication
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="val"></param>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> operator*(t val, const Array<t, loc> &other)
		{
			auto res = Array<t, loc>(other.shape);

			if (loc == CPU)
			{
				Array<t, loc>::binaryOpScalarArray(val, other, res,
												   math::prod(other.shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
												   [](t x, t y)
				{
					return x * y;
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto res = Array<t, loc>(other.shape);
				cuda::mul_scalar_array(math::prod(other.shape), val, other.dataStart, res.dataStart);
				return res;
			}
		#endif
			return res;
		}

		/// <summary>
		/// Reverse division
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="val"></param>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> operator/(t val, const Array<t, loc> &other)
		{
			auto res = Array<t, loc>(other.shape);

			if (loc == CPU)
			{
				Array<t, loc>::binaryOpScalarArray(val, other, res,
												   math::prod(other.shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
												   [](t x, t y)
				{
					return x / y;
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto res = Array<t, loc>(other.shape);
				cuda::div_scalar_array(math::prod(other.shape), val, other.dataStart, res.dataStart);
				return res;
			}
		#endif
			return res;
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> minimum(const Array<t, loc> &arr, t x)
		{
			if (loc == CPU)
			{
				return arr.mapped([&](t val)
				{
					return val < x ? val : x;
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto res = Array<t, loc>(arr.shape);
				cuda::array_minimum(math::prod(arr.shape), arr.dataStart, x, res.dataStart);
				return res;
			}
		#endif
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> maximum(const Array<t, loc> &arr, t x)
		{
			if (loc == CPU)
			{
				return arr.mapped([&](t val)
				{
					return val > x ? val : x;
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto res = Array<t, loc>(arr.shape);
				cuda::array_maximum(math::prod(arr.shape), arr.dataStart, x, res.dataStart);
				return res;
			}
		#endif
		}

		/// <summary>
		/// Sum all of the elements of an array
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="arr"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline t sum(const Array<t, loc> &arr)
		{
			if (loc == CPU)
			{
				t res = 0;

				for (size_t i = 0; i < math::prod(arr.shape); i++)
					res += arr.dataStart[i];
				return res;
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto host = new t[math::prod(arr.shape)];
				cudaSafeCall(cudaDeviceSynchronize());
				cudaSafeCall(cudaMemcpy(host, arr.dataStart, sizeof(t) * math::prod(arr.shape), cudaMemcpyDeviceToHost));
				t res = 0;

				for (size_t i = 0; i < math::prod(arr.shape); i++)
					res += host[i];

				delete[] host;
				return res;
			}
		#endif
		}

		/// <summary>
		/// Calculate the exponent of every value
		/// in an array, and return the result
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="arr"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> exp(const Array<t, loc> &arr)
		{
			Array<t, loc> result(arr.shape);

			if (loc == CPU)
			{
				ExecutionType mode;
				if (math::prod(arr.shape) > 10000)
					mode = ExecutionType::PARALLEL;
				else
					mode = ExecutionType::SERIAL;

				Array<t, loc>::unaryOpArray(arr, result, mode, [](t x)
				{
					return std::exp(x);
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				cuda::array_exp(math::prod(arr.shape), arr.dataStart, result.dataStart);
			}
		#endif

			return result;
		}

		/// <summary>
		/// Square every element in an array and return
		/// the result
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="arr"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> square(const Array<t, loc> &arr)
		{
			Array<t, loc> result(arr.shape);

			if (loc == CPU)
			{
				ExecutionType mode;
				if (math::prod(arr.shape) > 10000)
					mode = ExecutionType::PARALLEL;
				else
					mode = ExecutionType::SERIAL;

				Array<t, loc>::unaryOpArray(arr, result, mode, [](t x)
				{
					return x * x;
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				cuda::array_exp(math::prod(arr.shape), arr.dataStart, result.dataStart);
			}
		#endif

			return result;
		}

		/// <summary>
		/// Square root every element in an array
		/// and return the result
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="arr"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> sqrt(const Array<t, loc> &arr)
		{
			Array<t, loc> result(arr.shape);

			if (loc == CPU)
			{
				ExecutionType mode;
				if (math::prod(arr.shape) > 10000)
					mode = ExecutionType::PARALLEL;
				else
					mode = ExecutionType::SERIAL;

				Array<t, loc>::unaryOpArray(arr, result, mode, [](t x)
				{
					return std::sqrt(x);
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				cuda::array_square(math::prod(arr.shape), arr.dataStart, result.dataStart);
			}
		#endif

			return result;
		}

		/// <summary>
		/// Raise an array to a power
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="arr"></param>
		/// <param name="power"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> pow(const Array<t, loc> &arr, t power)
		{
			Array<t, loc> result(arr.shape);

			if (loc == CPU)
			{
				ExecutionType mode;
				if (math::prod(arr.shape) > 10000)
					mode = ExecutionType::PARALLEL;
				else
					mode = ExecutionType::SERIAL;

				Array<t, loc>::unaryOpArray(arr, result, mode, [=](t x)
				{
					return std::pow(x, power);
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				cuda::array_pow(math::prod(arr.shape), arr.dataStart, power, result.dataStart);
			}
		#endif

			return result;
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> sin(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::sin(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> cos(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::cos(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> tan(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::tan(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> asin(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::asin(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> acos(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::acos(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> atan(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::atan(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> sinh(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::sinh(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> cosh(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::cosh(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> tanh(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::tanh(val);
			});
		}

		/// <summary>
		/// Create a vector of a given length where the first element
		/// is "start" and the final element is "end", increasing in
		/// regular increments
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="start"></param>
		/// <param name="end"></param>
		/// <param name="len"></param>
		/// <returns></returns>
		template<typename s, typename e>
		inline Array<typename std::common_type<s, e>::type> linspace(s start, e end, size_t len)
		{
			using ct = typename std::common_type<s, e>::type;

			Array<ct> result({len});
			result.isZeroDim = len <= 1;

			if (len == 0)
				return result;

			if (len == 1)
			{
				result.dataStart[0] = start;
				return result;
			}

			ct inc = ((ct) end - (ct) start) / (ct) (len - 1);
			for (size_t i = 0; i < len; i++)
				result.dataStart[i] = (ct) start + (ct) i * inc;

			return result;
		}

		/// <summary>
		/// Create a vector of a specified type, where the values
		/// increase/decrease linearly between a start and end
		/// point by an specified
		/// </summary>
		/// <typeparam name="s"></typeparam>
		/// <typeparam name="e"></typeparam>
		/// <typeparam name="t"></typeparam>
		/// <param name="start"></param>
		/// <param name="end"></param>
		/// <param name="inc"></param>
		/// <returns></returns>
		template<typename s, typename e, typename t>
		inline Array<typename std::common_type<s, e, t>::type> arange(s start, e end, t inc = 1)
		{
			using ct = typename std::common_type<s, e, t>::type;

			auto len = (uint64_t) ceil(abs((ct) end - (ct) start) / (ct) inc);
			auto res = Array<t, CPU>({len});
			for (uint64_t i = 0; i < len; i++)
				res[i] = (ct) start + (ct) inc * (ct) i;
			return res;
		}

		/// <summary>
		/// Create a vector of a specified type, where the values
		/// increase/decrease linearly between a start and end
		/// point by an specified
		/// </summary>
		/// <typeparam name="e"></typeparam>
		/// <param name="end"></param>
		/// <returns></returns>
		template<typename e>
		inline Array<e, CPU> arange(e end)
		{
			return arange((e) 0, end, (e) 1);
		}

		/// <summary>
		/// Create a 3D array from two vectors, where the first element
		/// is vector A in row format, and the second element is vector
		/// B in column format.
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> meshgrid(const Array<t, loc> &a, const Array<t, loc> &b)
		{
			rapidAssert(a.shape.size() == 1 && b.shape.size() == 1, "Invalid size for meshgrid. Must be a 1D array");
			Array<t, loc> result({2, b.shape[0], a.shape[0]});

			if (math::prod(result.shape) < 10000)
			{
				for (int64_t i = 0; i < b.shape[0]; i++)
					for (int64_t j = 0; j < a.shape[0]; j++)
						result.setVal({(int64_t) 0, i, j}, a.accessVal({j}));

				for (int64_t i = 0; i < b.shape[0]; i++)
					for (int64_t j = 0; j < a.shape[0]; j++)
						result.setVal({(int64_t) 1, i, j}, b.accessVal({i}));
			}
			else
			{
			#pragma omp parallel for
				for (int64_t i = 0; i < b.shape[0]; i++)
					for (int64_t j = 0; j < a.shape[0]; j++)
						result.setVal({(int64_t) 0, i, j}, a.accessVal({j}));

			#pragma omp parallel for
				for (int64_t i = 0; i < b.shape[0]; i++)
					for (int64_t j = 0; j < a.shape[0]; j++)
						result.setVal({(int64_t) 1, i, j}, b.accessVal({i}));
			}

			return result;
		}

		/// <summary>
		/// Return a gaussian matrix with the given rows, columns and
		/// standard deviation.
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="r"></param>
		/// <param name="c"></param>
		/// <param name="sigma"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> gaussian(size_t r, size_t c, t sigma)
		{
			t rows = (t) r;
			t cols = (t) c;

			auto ax = linspace<t>(-(rows - 1) / 2., (rows - 1) / 2., r);
			auto ay = linspace<t>(-(cols - 1) / 2., (cols - 1) / 2., c);
			auto mesh = meshgrid(ay, ax);
			auto xx = mesh[0];
			auto yy = mesh[1];

			auto kernel = exp(-0.5 * (square(xx) + square(yy)) / (sigma * sigma));
			return kernel / sum(kernel);
		}

		/// <summary>
		/// Cast an array from one type to another. This makes a copy of the array,
		/// and therefore altering a value in one will not cause an update in the
		/// other.
		/// </summary>
		/// <typeparam name="res"></typeparam>
		/// <typeparam name="src"></typeparam>
		/// <param name="src"></param>
		/// <returns></returns>
		template<typename resT, typename srcT>
		inline Array<resT> cast(const Array<srcT> &src)
		{
			Array<resT> res(src.shape);

			if (math::prod(src.shape) < 10000)
			{
				for (int64_t i = 0; i < math::prod(src.shape); i++)
					res.dataStart[i] = (resT) src.dataStart[i];
			}
			else
			{
			#pragma omp parallel for
				for (int64_t i = 0; i < math::prod(src.shape); i++)
					res.dataStart[i] = (resT) src.dataStart[i];
			}

			return res;
		}
	}
}
