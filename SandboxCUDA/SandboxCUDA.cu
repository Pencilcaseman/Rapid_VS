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

	// auto arr = rapid::ndarray::Array<float, GPU>::fromData({{1, 2}, {3, 4}});
	// 
	// std::cout << arr << "\n\n";
	// std::cout << arr + arr << "\n\n";
	// std::cout << arr - arr << "\n\n";
	// std::cout << arr * arr << "\n\n";
	// std::cout << arr / arr << "\n\n";
	// 
	// std::cout << "\n\n";
	// 
	// std::cout << arr + 10 << "\n\n";
	// std::cout << arr - 10 << "\n\n";
	// std::cout << arr * 10 << "\n\n";
	// std::cout << arr / 10 << "\n\n";
	// 
	// std::cout << "\n\n";
	// 
	// std::cout << arr << "\n\n";
	// arr += rapid::ndarray::Array<float, GPU>::fromData({{1, 2}, {3, 4}});
	// std::cout << arr << "\n\n";
	// arr -= rapid::ndarray::Array<float, GPU>::fromData({{1, 2}, {3, 4}});
	// std::cout << arr << "\n\n";
	// arr *= rapid::ndarray::Array<float, GPU>::fromData({{1, 2}, {3, 4}});
	// std::cout << arr << "\n\n";
	// arr /= rapid::ndarray::Array<float, GPU>::fromData({{1, 2}, {3, 4}});
	// std::cout << arr << "\n\n";
	// 
	// std::cout << "\n\n";
	// 
	// std::cout << arr << "\n\n";
	// arr += 10;
	// std::cout << arr << "\n\n";
	// arr -= 10;
	// std::cout << arr << "\n\n";
	// arr *= 10;
	// std::cout << arr << "\n\n";
	// arr /= 10;
	// std::cout << arr << "\n\n";
	// 
	// std::cout << "\n\n\n\n";

	auto lhs = rapid::ndarray::Array<float, GPU>::fromData({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
	auto rhs = rapid::ndarray::Array<float, GPU>::fromData({{1, 2}, {3, 4}, {5, 6}});

	std::cout << lhs << "\n\n";
	std::cout << rhs << "\n\n";
	std::cout << lhs.dot(rhs) << "\n\n";

	rhs[1][1] = 12345;

	std::cout << rhs << "\n\n";
	std::cout << "Test: " << rhs[1][0] << "\n";

	{
		std::cout << "Timing GPU<float>\n";
		auto speedTestGPU = rapid::ndarray::Array<float, GPU>({1000, 1000});

		START_TIMER(0, 10000);
		auto res = speedTestGPU.dot(speedTestGPU);
		END_TIMER(0);

		std::cout << "Timing CPU<float>\n";
		auto speedTestCPU = rapid::ndarray::Array<float, CPU>({1000, 1000});

		START_TIMER(1, 100);
		auto res = speedTestCPU.dot(speedTestCPU);
		END_TIMER(1);
	}

	{
		std::cout << "Timing GPU<double>\n";
		auto speedTestGPU = rapid::ndarray::Array<double, GPU>({1000, 1000});

		START_TIMER(0, 10000);
		auto res = speedTestGPU.dot(speedTestGPU);
		END_TIMER(0);

		std::cout << "Timing CPU<double>\n";
		auto speedTestCPU = rapid::ndarray::Array<double, CPU>({1000, 1000});

		START_TIMER(1, 1000);
		auto res = speedTestCPU.dot(speedTestCPU);
		END_TIMER(1);
	}

	return 0;
}
