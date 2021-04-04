#pragma once

#include "../array.h"

namespace rapid
{
	namespace network
	{
		namespace defaults
		{
			template<typename _Ty>
			struct Scalar
			{
				int initialized = 0;
				_Ty defaultValue = 0;
				_Ty value = 0;

				inline void setValue(_Ty val)
				{
					initialized = 1;
					value = val;
				}

				inline _Ty getValue() const
				{
					return initialized ? value : defaultValue;
				}
			};

			template<typename _Ty, ndarray::ArrayLocation loc = ndarray::CPU>
			struct NDArray
			{
				int initialized = 0;
				ndarray::Array<_Ty, loc> defaultValue;
				ndarray::Array<_Ty, loc> value;

				inline void setValue(const ndarray::Array<_Ty, loc> &val)
				{
					initialized = 1;
					value.set(val.copy());
				}

				inline ndarray::Array<_Ty, loc> getValue() const
				{
					return initialized ? value : defaultValue;
				}
			};
		}
	}
}
