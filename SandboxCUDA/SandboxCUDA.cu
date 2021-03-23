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

	// ADAM
	// > CPU ==> Block finished in: 23.239650 ms
	// > GPU ==> Block finished in: 9.897372 ms

	// > CPU ==> Block finished in: 136.322890 ms
	// > GPU ==> Block finished in: 38.881290 ms

	// SGD
	// > CPU ==> Block finished in: 10.753736 ms
	// > GPU ==> Block finished in: 3.629665 ms

	using networkType = float;
	const rapid::ndarray::ArrayLocation networkLocation = GPU;

	// auto w = rapid::ndarray::Array<networkType, networkLocation>({5, 5});
	// auto dw = rapid::ndarray::onesLike(w);
	// w.fill(0.5);
	// auto optimizer = std::make_shared<rapid::network::optim::adam<networkType, networkLocation>>();
	// 
	// auto res = optimizer->apply(w, dw);
	// w.set(res.weight);
	// rapid::network::optim::Config<networkType, networkLocation> config = res.config;
	// 
	// std::cout << w.toString() << "\n\n";
	// for (int i = 0; i < 5; i++)
	// {
	// 	res = optimizer->apply(w, dw, config);
	// 	w.set(res.weight);
	// 	config = res.config;
	// 	std::cout << w.toString() << "\n\n";
	// }
	
	{
		auto w = rapid::ndarray::Array<networkType, networkLocation>({3000, 3000});
		auto dw = rapid::ndarray::onesLike(w);
		w.fill(0.5);
		auto optimizer = std::make_shared<rapid::network::optim::sgd<networkType, networkLocation>>();
	
		auto res = optimizer->apply(w, dw);
		w.set(res.weight);
		auto config = res.config;
	
		std::cout << "Timing:\n";
		START_TIMER(0, 5000);
		res = optimizer->apply(w, dw, config);
		w.set(res.weight);
		config = res.config;
		END_TIMER(0);
	}

	return 0;
}
