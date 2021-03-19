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

	auto arr = rapid::ndarray::Array<float, GPU>::fromData({{1, 2}, {3, 4}});
	
	std::cout << arr << "\n\n";
	std::cout << arr + arr << "\n\n";
	std::cout << arr - arr << "\n\n";
	std::cout << arr * arr << "\n\n";
	std::cout << arr / arr << "\n\n";
	
	std::cout << "\n\n";

	std::cout << arr + 10 << "\n\n";
	std::cout << arr - 10 << "\n\n";
	std::cout << arr * 10 << "\n\n";
	std::cout << arr / 10 << "\n\n";

	std::cout << "\n\n";

	std::cout << arr << "\n\n";
	arr += rapid::ndarray::Array<float, GPU>::fromData({{1, 2}, {3, 4}});
	std::cout << arr << "\n\n";
	arr -= rapid::ndarray::Array<float, GPU>::fromData({{1, 2}, {3, 4}});
	std::cout << arr << "\n\n";
	arr *= rapid::ndarray::Array<float, GPU>::fromData({{1, 2}, {3, 4}});
	std::cout << arr << "\n\n";
	arr /= rapid::ndarray::Array<float, GPU>::fromData({{1, 2}, {3, 4}});
	std::cout << arr << "\n\n";

	std::cout << "\n\n";

	std::cout << arr << "\n\n";
	arr += 10;
	std::cout << arr << "\n\n";
	arr -= 10;
	std::cout << arr << "\n\n";
	arr *= 10;
	std::cout << arr << "\n\n";
	arr /= 10;
	std::cout << arr << "\n\n";

	std::cout << "\n\n";

	auto lhs = rapid::ndarray::Array<float, GPU>::fromData({{1, 2, 3}, {4, 5, 6}});
	auto rhs = rapid::ndarray::Array<float, GPU>::fromData({{1, 2}, {3, 4}, {5, 6}});
	std::cout << lhs.dot(rhs) << "\n\n";

	// 6.904483 ms
	auto speedTest = rapid::ndarray::Array<double, CPU>({1000, 1000});

	START_TIMER(0, 1000);
	auto res = speedTest.dot(speedTest);
	END_TIMER(0);

	return 0;
}
