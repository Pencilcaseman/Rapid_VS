#pragma once

#include "../array.h"

namespace rapid
{
	namespace network
	{
		template<typename t, ndarray::ArrayLocation loc>
		ndarray::Array<t> relu(const ndarray::Array<t, loc> &arr)
		{
			return ndarray::maximum(arr, 0);
		}

		template<typename t, ndarray::ArrayLocation loc>
		ndarray::Array<t> tanh(const ndarray::Array<t, loc> &arr)
		{
			return ndarray::tanh(arr, 0);
		}

		template<typename t, ndarray::ArrayLocation loc>
		ndarray::Array<t> sigmoid(const ndarray::Array<t, loc> &arr)
		{
			return 1. / (1. + ndarray::exp(-arr));
		}

		template<typename t, ndarray::ArrayLocation loc>
		struct Cache
		{
			ndarray::Array<t, loc> x;
			ndarray::Array<t, loc> w;
			ndarray::Array<t, loc> b;
		};

		template<typename t, ndarray::ArrayLocation loc>
		struct AffineOutput
		{
			ndarray::Array<t, loc> out;
			Cache<t, loc> cache;
		};

		template<typename t, ndarray::ArrayLocation loc>
		inline AffineOutput<t, loc> affineForward(const ndarray::Array<t, loc> &x, const ndarray::Array<t, loc> &w, const ndarray::Array<t, loc> &b)
		{
			auto z = x.reshape({x.shape[0], AUTO});

			std::cout << "info: " << z << "\n\n";

			auto out = z.dot(w) + b;
			return {out, {x, w, b}};
		}
	}
}
