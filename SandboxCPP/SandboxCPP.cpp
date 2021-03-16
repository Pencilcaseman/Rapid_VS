// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP
// #define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#include <iostream>
#include <rapid/rapid.h>

int main()
{
	auto w = rapid::ndarray::Array<double>({5, 5});
	auto dw = rapid::ndarray::onesLike(w);
	w.fill(0.5);
	auto optimizer = std::make_shared<rapid::network::optim::adam<double>>();

	auto res = optimizer->apply(w, dw);
	w.set(res.weight);
	rapid::network::optim::Config<double> config = res.config;

	std::cout << w.toString() << "\n\n";
	for (int i = 0; i < 5; i++)
	{
		res = optimizer->apply(w, dw, config);
		w.set(res.weight);
		config = res.config;
		std::cout << w.toString() << "\n\n";
	}

	{
		auto w = rapid::ndarray::Array<double>({1000, 1000});
		auto dw = rapid::ndarray::onesLike(w);
		w.fill(0.5);
		auto optimizer = std::make_shared<rapid::network::optim::adam<double>>();

		auto res = optimizer->apply(w, dw);
		w.set(res.weight);
		rapid::network::optim::Config<double> config = res.config;

		START_TIMER(0, 10);
		res = optimizer->apply(w, dw, config);
		w.set(res.weight);
		config = res.config;
		END_TIMER(0);
	}

	auto chonkey = rapid::ndarray::Array<float>({1000, 1000});

	START_TIMER(1, 1000);
	auto res = chonkey.dot(chonkey);
	END_TIMER(1);

	return 0;
}
