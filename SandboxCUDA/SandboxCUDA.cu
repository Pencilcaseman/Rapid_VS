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
	const rapid::ndarray::ArrayLocation networkLocation = GPU;

	auto w = rapid::ndarray::Array<networkType, networkLocation>({5, 5});
	auto dw = rapid::ndarray::onesLike(w);
	w.fill(0.5);
	auto optimizer = std::make_shared<rapid::network::optim::adam<networkType, networkLocation>>();

	auto res = optimizer->apply(w, dw);
	w.set(res.weight);
	rapid::network::optim::Config<networkType, networkLocation> config = res.config;

	std::cout << w.toString() << "\n\n";
	for (int i = 0; i < 5; i++)
	{
		res = optimizer->apply(w, dw, config);
		w.set(res.weight);
		config = res.config;
		std::cout << w.toString() << "\n\n";
	}

	{
		auto w = rapid::ndarray::Array<networkType, networkLocation>({1000, 1000});
		auto dw = rapid::ndarray::onesLike(w);
		w.fill(0.5);
		auto optimizer = std::make_shared<rapid::network::optim::adam<networkType, networkLocation>>();
	
		auto res = optimizer->apply(w, dw);
		w.set(res.weight);
		auto config = res.config;
	
		std::cout << "Timing:\n";
		START_TIMER(0, 100);
		res = optimizer->apply(w, dw, config);
		w.set(res.weight);
		config = res.config;
		END_TIMER(0);
	}

	std::cout << rapid::ndarray::minimum(rapid::ndarray::Array<double, CPU>::fromData({1, 2, 3, 4, 5}), 3.) << "\n";

	return 0;
}
