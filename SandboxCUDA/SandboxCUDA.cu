// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP
// #define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#define RAPID_CUDA

#include <iostream>
#include <rapid/rapid.h>

int main()
{
	auto arr = rapid::ndarray::Array<float, rapid::ndarray::GPU>({10, 10});
	arr.fill(1);
	
	std::cout << arr << "\n\n";
	std::cout << (arr + arr) << "\n\n";

	auto cpuArr = rapid::ndarray::Array<float, rapid::ndarray::CPU>({20000, 20000});
	cpuArr.fill(1);

	std::cout << "CPU Iterations: ";
	uint64_t cpuIters;
	std::cin >> cpuIters;

	START_TIMER(0, cpuIters);
	auto res = cpuArr + cpuArr;
	END_TIMER(0);

	auto gpuArr = rapid::ndarray::Array<float, rapid::ndarray::GPU>({20000, 20000});
	gpuArr.fill(1);

	std::cout << "GPU Iterations: ";
	uint64_t gpuIters;
	std::cin >> gpuIters;

	START_TIMER(1, gpuIters);
	auto res = gpuArr + gpuArr;
	END_TIMER(1);

	return 0;
}
