// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP
// #define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#include <iostream>
#include "include/rapid/rapid.h"

int main()
{
	auto arr = rapid::Array<double>({3, 3, 3});
	arr.fill(0);
	std::cout << arr.toString() << "\n";

	return 0;
}
