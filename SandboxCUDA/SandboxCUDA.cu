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

	auto dotTest = rapid::ndarray::Array<float, GPU>({1000, 1000});
	START_TIMER(0, 1000);
	auto dotRes = dotTest.dot(dotTest);
	END_TIMER(0);

	auto dotTest2 = rapid::ndarray::Array<float, CPU>({1000, 1000});
	START_TIMER(1, 100);
	auto dotRes = dotTest2.dot(dotTest2);
	END_TIMER(1);

	auto dotTest3 = rapid::ndarray::Array<double, GPU>({1000, 1000});
	START_TIMER(2, 1000);
	auto dotRes = dotTest3.dot(dotTest3);
	END_TIMER(2);

	auto dotTest4 = rapid::ndarray::Array<double, CPU>({1000, 1000});
	START_TIMER(3, 100);
	auto dotRes = dotTest4.dot(dotTest4);
	END_TIMER(3);

	std::cout << rapid::ndarray::Array<float, GPU>::fromData({1, 2, 3, 4, 5, 6}).reshaped({2, 3}).dot(rapid::ndarray::Array<float, GPU>::fromData({1, 2, 3}).reshaped({3, 1})) << "\n\n";

	return 0;
}
