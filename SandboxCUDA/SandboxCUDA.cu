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

	auto arr = rapid::ndarray::Array<float, CPU>::fromData({{1, 2}, {3, 4}});
	arr[0] = rapid::ndarray::Array<float, CPU>::fromData({123, 456});
	
	std::cout << arr << "\n\n";
	std::cout << arr[0] << "\n\n";
	std::cout << arr[0][0] << "\n\n";
	std::cout << (int) arr[0][0] << "\n\n";

	return 0;
}
