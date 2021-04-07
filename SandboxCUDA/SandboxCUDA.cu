#define RAPID_NO_BLAS
#define RAPID_NO_AMP
#define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#define RAPID_CUDA

#include <iostream>
#include <rapid/rapid.h>

int main()
{
	using rapid::ndarray::CPU;
	using rapid::ndarray::GPU;

	using networkType = float;
	const rapid::ndarray::ArrayLocation networkLocation = CPU;

	// auto x = rapid::ndarray::Array<networkType, networkLocation>({5, 3, 3, 3});
	// auto w = rapid::ndarray::Array<networkType, networkLocation>({3 * 3 * 3, 5});
	// auto b = rapid::ndarray::Array<networkType, networkLocation>({5});
	// 
	// x.fill(0.5);
	// w.fill(-0.5);
	// b.fill(0.75);
	// 
	// auto res = rapid::network::affineForward(x, w, b);
	// 
	// auto dOut = rapid::ndarray::zerosLike(res.out);
	// dOut.fill(0.2);
	// 
	// auto backwardRes = rapid::network::affineBackward(dOut, res.cache);
	// 
	// std::cout << backwardRes.delta.x << "\n\n";
	// std::cout << backwardRes.delta.w << "\n\n";
	// std::cout << backwardRes.delta.b << "\n\n";

	// {
	// 	auto arr = rapid::ndarray::Array<float, CPU>({1000, 1000});
	// 
	// 	START_TIMER(0, 10);
	// 	auto res = rapid::ndarray::var(arr);
	// 	END_TIMER(0);
	// 
	// 	START_TIMER(1, 10);
	// 	auto res = rapid::ndarray::var(arr, 0);
	// 	END_TIMER(1);
	// 
	// 	START_TIMER(2, 10);
	// 	auto res = rapid::ndarray::var(arr, 1);
	// 	END_TIMER(2);
	// }
	// 
	// {
	// 	auto arr = rapid::ndarray::Array<float, GPU>({1000, 1000});
	// 
	// 	START_TIMER(0, 10);
	// 	auto res = rapid::ndarray::var(arr);
	// 	END_TIMER(0);
	// 
	// 	START_TIMER(1, 10);
	// 	auto res = rapid::ndarray::var(arr, 0);
	// 	END_TIMER(1);
	// 
	// 	START_TIMER(2, 10);
	// 	auto res = rapid::ndarray::var(arr, 1);
	// 	END_TIMER(2);
	// }

	auto arr = rapid::ndarray::arange<GPU>(1.f, 9.f).reshaped({2, 2, 2});
	std::cout << arr << "\n\n";
	std::cout << "========================================\n";
	std::cout << arr.transposed({0, 1, 2}) << "\n\n";
	std::cout << "========================================\n";
	std::cout << arr.transposed({0, 2, 1}) << "\n\n";
	std::cout << "========================================\n";
	std::cout << arr.transposed({2, 0, 1}) << "\n\n";
	std::cout << "========================================\n";
	std::cout << arr.transposed({2, 1, 0}) << "\n\n";
	std::cout << "========================================\n";
	std::cout << arr.transposed() << "\n\n";

	{
		auto arr = rapid::ndarray::Array<float, GPU>({2000, 2000});

		START_TIMER(0, 1000);
		auto res = arr.dot(arr);
		END_TIMER(0);
	}

	{
		auto arr = rapid::ndarray::Array<float, GPU>({2000, 2000});

		START_TIMER(0, 1000);
		auto res = arr.transposed();
		END_TIMER(0);
	}

	std::cout << "\n\n\n\n\n\n\n";

	auto test = rapid::ndarray::Array<float, CPU>::fromData({1, 2, 3, 4, 5, 6, 7, 8}).reshaped({2, 2, 2});
	std::cout << test << "\n\n";

	std::cout << test.transposed() << "\n\n";
	std::cout << test.transposed().reshaped({AUTO}) << "\n\n";

	return 0;
}
