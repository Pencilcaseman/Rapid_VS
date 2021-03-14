#pragma once

#include "../internal.h"
#include "activations.h"
#include "optimizers.h"

namespace rapid
{
	namespace network
	{
		template<typename t>
		std::vector<ndarray::Array<t>> toCategorical(const std::vector<size_t> &data, size_t numClasses)
		{
			ndarray::Array<t> res({data.size(), numClasses, 1});
			uint64_t index = 0;
			for (const auto &val : data)
			{
				rapidAssert(val < numClasses, "Class exceeds number of classes specified");

				auto tmp = ndarray::Array<t>({numClasses, 1});
				tmp.fill(0);
				tmp.setVal({val, 0}, 1);

				res[index++] = tmp;
			}

			return res;
		}
	}
}
