#define RAPID_NO_BLAS
#define RAPID_NO_AMP
#define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#define RAPID_CUDA

#include <iostream>
#include <rapid/rapid.h>

int main()
{
	// FUNCTIONS MUST TAKE INTO ACCOUNT "MATRIX DATA"
	// Addition, etc.
	// To string
	// Others???

	/*
	
	Test everything -- every possible combination of values!?

	Maximum and minimum for GPU arrays

	Reverse grid addition

	*/

	using rapid::ndarray::CPU;
	using rapid::ndarray::GPU;

	auto arr = rapid::ndarray::Array<float, GPU>::fromData({1, 2, 3, 4, 5, 6, 7, 8});
	auto matrix = arr.reshaped({2, 4});
	auto d3 = arr.reshaped({2, 2, 2});

	std::cout << arr << "\n\n";
	
	std::cout << "To matrix\n";
	std::cout << matrix << "\n\n";
	
	std::cout << "To 3D\n";
	std::cout << d3 << "\n\n";
	
	std::cout << "Setting matrix\n";
	matrix[0][0] = 5;
	std::cout << arr << "\n\n";
	std::cout << matrix << "\n\n";
	
	std::cout << "Setting 3D\n";
	d3[1][1][1] = 123456;
	std::cout << arr << "\n\n";
	std::cout << d3 << "\n\n";

	return 0;
}
