// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP
// #define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#define RAPID_CUDA

#include <iostream>
#include <rapid/rapid.h>

int main()
{
	auto x = rapid::ndarray::Array<float, rapid::ndarray::GPU>({10000, 10000});

	// 26.365480 ms --> 4m 23.65s
	// 1.033473 ms  --> 0m 10.33s

	START_TIMER(0, 10000);
	x.fill(3.14159);
	END_TIMER(0);
	
	std::cout << x.toString() << "\n";

	return 0;
}
