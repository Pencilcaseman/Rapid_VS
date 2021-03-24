// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP
// #define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#define RAPID_CUDA

#include <iostream>
#include <rapid/rapid.h>

int main()
{
	using rapid::ndarray::CPU;
	using rapid::ndarray::GPU;

	auto arr = rapid::ndarray::Array<float, GPU>::fromData({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
	arr[1][1][0] = 99999;
	std::cout << arr << "\n\n";
	std::cout << arr.transposed({0, 2, 1}) << "\n\n";

	return 0;
}
