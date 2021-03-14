// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP
#define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#include <iostream>
#include "include/rapid/rapid.h"

int main()
{
// 	def test(n) :
// 		... : weight = np.ndarray(shape = (5, 5))
// 		... : weight.fill(0.5)
// 		... :
// 		... : dw = np.ones_like(weight)
// 		... :
// 		... : first, config = optim.sgd(weight, dw)
// 		... :
// 		... : for i in range(n) :
// 		... : first = optim.sgd(weight, dw, config)
// 		... :
// 		... : print(first, "\n\n")

	auto weight = rapid::Array<double>({5, 5});
	weight.fill(0.5);

	auto dw = rapid::onesLike(weight);

	auto config = rapid::optim::newConfig<double>();
	auto res = rapid::optim::rmsprop(weight, dw, config);

	for (int i = 0; i < 5; i++)
	{
		auto newRes = rapid::optim::rmsprop(weight, dw, config);
		std::cout << newRes.weight.toString() << "\n\n";
	}

	return 0;
}
