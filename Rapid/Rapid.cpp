// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP
#define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#include <iostream>
#include "include/rapid/rapid.h"

int main()
{
	auto weight = rapid::ndarray::Array<double>({5, 5});
	weight.fill(0.5);

	auto dw = rapid::ndarray::onesLike(weight);

	auto config = rapid::network::optim::newConfig<double>();
	auto res = rapid::network::optim::adam(weight, dw, config);

	for (int i = 0; i < 5; i++)
	{
		auto newRes = rapid::network::optim::adam(weight, dw, config);
		std::cout << newRes.weight.toString() << "\n\n";
	}

	auto testArr = rapid::ndarray::Array<double>::fromData({1, 2, 3, 4, 5, 6});
	std::cout << testArr.toString() << "\n";
	std::cout << testArr.resized({6, AUTO}).toString() << "\n";

	return 0;
}
