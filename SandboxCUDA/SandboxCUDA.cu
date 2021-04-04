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

	auto arr = rapid::ndarray::Array<float, CPU>::fromData({{1, 2, 3}, {4, 5, 6}});
	std::cout << arr << "\n\n";
	std::cout << arr - rapid::ndarray::mean(arr, 0) << "\n\n";
	std::cout << rapid::ndarray::var(arr) << "\n\n";
	std::cout << rapid::ndarray::var(arr, 0) << "\n\n";
	std::cout << rapid::ndarray::var(arr, 1) << "\n\n";

	return 0;
}
