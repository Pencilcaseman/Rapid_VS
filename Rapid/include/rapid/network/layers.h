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

		/// <summary>
		/// Contains all of the required information for an
		/// affine forward pass
		/// </summary>
		/// <typeparam name="t"></typeparam>
		template<typename t, ndarray::ArrayLocation loc>
		struct AffineOutput
		{
			ndarray::Array<t, loc> out;
			Cache<t, loc> cache;
		};

		/// <summary>
		/// Contains the information required for a backward
		/// pass of an affine network layer
		/// </summary>
		/// <typeparam name="t"></typeparam>
		template<typename t, ndarray::ArrayLocation loc>
		struct AffineBackwardOutput
		{
			Cache<t, loc> delta;
		};

		/// <summary>
		/// Compute a forward pass on an affine (fully connected)
		/// layer, provided an input, weight and bias. The output
		/// contains a cache of these values, as well as the actual
		/// output from the computation
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="x"></param>
		/// <param name="w"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		template<typename t, ndarray::ArrayLocation loc>
		inline AffineOutput<t, loc> affineForward(const ndarray::Array<t, loc> &x, const ndarray::Array<t, loc> &w, const ndarray::Array<t, loc> &b)
		{
			auto z = x.reshaped({x.shape[0], AUTO});
			auto out = z.dot(w) + b;
			return {out, {x, w, b}};
		}

		template<typename t, ndarray::ArrayLocation loc>
		inline AffineBackwardOutput<t, loc> affineBackward(const ndarray::Array<t, loc> &dOut, const Cache<t, loc> &cache)
		{
			const auto &shapes = cache.x.shape;
			const auto N = shapes[0];
			auto z = cache.x.reshaped({N, AUTO});

			std::cout << "Information\n";
			std::cout << dOut << "\n\n";
			std::cout << cache.w.transposed() << "\n";

			auto dx = dOut.dot(cache.w.transposed()).reshaped(shapes);
			auto dw = (z.transposed()).dot(dOut);
			auto db = (ndarray::ones<t>({N})).dot(dOut);

			return {{dx, dw, db}};
		}
	}
}
