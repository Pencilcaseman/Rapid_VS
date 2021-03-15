#pragma once

#include "../array.h"

namespace rapid
{
	namespace network
	{
		template<typename t>
		ndarray::Array<t> relu(const ndarray::Array<t> &arr)
		{
			return ndarray::maximum(arr, 0);
		}

		template<typename t>
		ndarray::Array<t> tanh(const ndarray::Array<t> &arr)
		{
			return ndarray::tanh(arr, 0);
		}

		template<typename t>
		ndarray::Array<t> sigmoid(const ndarray::Array<t> &arr)
		{
			return 1. / (1. + ndarray::exp(-arr));
		}

		template<typename t>
		struct Cache
		{
			ndarray::Array<t> x;
			ndarray::Array<t> w;
			ndarray::Array<t> b;
		};

		template<typename t>
		struct LayerOutput
		{
			ndarray::Array<t> out;
			Cache<t> cache;
		};


	}
}
