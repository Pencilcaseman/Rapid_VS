#pragma once

namespace rapid
{
	namespace utils
	{
		/// <summary>
		/// Format an std::vector representing a 1D array into a string.
		/// Not for external use -- internal use only
		/// </summary>
		/// <param name="adjusted"></param>
		/// <param name="stripMiddle"></param>
		/// <returns></returns>
		std::string toString1D(const std::vector<std::string> &adjusted, bool stripMiddle)
		{
			std::string res = "[";

			for (size_t i = 0; i < adjusted.size(); i++)
			{
				if (stripMiddle && adjusted.size() > 6 && i == 3)
				{
					i = adjusted.size() - 3;
					res += "... ";
				}

				res += adjusted[i];
			}

			res[res.length() - 1] = ']';
			return res;
		}

		/// <summary>
		/// Recursive function to convert a vector into a single
		/// string, based on a given shape, starting depth and an optional parameter
		/// to remove excess values from the result.
		/// For internal use only.
		/// </summary>
		/// <param name="adjusted"></param>
		/// <param name="shape"></param>
		/// <param name="depth"></param>
		/// <param name="stripMiddle"></param>
		/// <returns></returns>
		std::string toString(const std::vector<std::string> &adjusted, const std::vector<size_t> &shape, size_t depth, bool stripMiddle)
		{
			if (shape.size() == 1)
				return toString1D(adjusted, stripMiddle);
			else if (shape.size() == 2)
			{
				std::string res = "[";

				size_t count = 0;
				for (size_t i = 0; i < adjusted.size(); i += shape[1])
				{
					if (stripMiddle && shape[0] > 6 && i == shape[1] * 3)
					{
						i = adjusted.size() - shape[1] * 3;
						res += std::string(depth, ' ') + "...\n";
						count = shape[0] - 3;
					}

					if (i != 0)
						res += std::string(depth, ' ');

					auto begin = adjusted.begin() + i;
					auto end = adjusted.begin() + i + shape[1];
					std::vector<std::string> substr(begin, end);
					res += toString1D(substr, stripMiddle);

					if (count + 1 != shape[0])
						res += "\n";

					count++;
				}

				return res + "]";
			}
			else
			{
				std::string res = "[";
				size_t count = 0;
				size_t inc = prod(shape) / shape[0];

				for (size_t i = 0; i < adjusted.size(); i += inc)
				{
					if (stripMiddle && shape[0] > 6 && i == inc * 3)
					{
						i = adjusted.size() - inc * 3;
						res += std::string(depth, ' ') + "...\n\n";
						count = shape[0] - 3;
					}

					if (i != 0)
						res += std::string(depth, ' ');

					auto adjustedStart = adjusted.begin() + i;
					auto adjustedEnd = adjusted.begin() + i + inc;
					auto shapeStart = shape.begin() + 1;
					auto shapeEnd = shape.end();

					auto subAdjusted = std::vector<std::string>(adjustedStart, adjustedEnd);
					auto subShape = std::vector<size_t>(shapeStart, shapeEnd);

					res += toString(subAdjusted, subShape, depth + 1, stripMiddle);

					if (count + 1 != shape[0])
						res += "\n\n";

					count++;
				}

				return res + "]";
			}
		}

		inline bool incArr(std::vector<uint64_t> &arr, const std::vector<uint64_t> &m)
		{
			arr[arr.size() - 1]++;
			uint64_t lastChanged = 0;

			for (uint64_t i = 0; i < arr.size(); i++)
			{
				if (arr[arr.size() - i - 1] >= m[m.size() - i - 1])
				{
					if (arr.size() - i == 1)
						return false;

					arr[arr.size() - i - 2]++;
					arr[arr.size() - i - 1] = 0;
					lastChanged = i;
				}
			}

			return true;
		}
	}

	/// <summary>
	/// Get a string representation of an array
	/// </summary>
	/// <typeparam name="t"></typeparam>
	/// <returns></returns>
// 	template<typename t>
// 	std::string Array<t>::toString() const
// 	{
// 		if (isZeroDim)
// 			return std::to_string(dataStart[0]);
// 
// 		size_t allocate = 1;
// 
// 		std::vector<utils::strContainer> formatted(prod(shape));
// 		size_t longestIntegral = 0;
// 		size_t longestDecimal = 0;
// 
// 		{
// 			auto timer = rapid::RapidTimer();
// 
// 			for (size_t i = 0; i < prod(shape); i++)
// 			{
// 				formatted[i] = utils::formatNumerical(dataStart[i]);
// 
// 				if (formatted[i].decimalPoint > longestIntegral)
// 					longestIntegral = formatted[i].decimalPoint;
// 
// 				if (formatted[i].str.length() >= formatted[i].decimalPoint && formatted[i].str.length() - formatted[i].decimalPoint > longestDecimal)
// 					longestDecimal = formatted[i].str.length() - formatted[i].decimalPoint;
// 			}
// 		}
// 
// 		std::vector<std::string> adjusted(formatted.size());
// 
// 		for (size_t i = 0; i < formatted.size(); i++)
// 		{
// 			const auto &term = formatted[i];
// 			auto decimal = term.str.length() - term.decimalPoint - 1;
// 
// 			size_t bufferLeft = 0;
// 
// 			auto tmp = std::string(longestIntegral - term.decimalPoint, ' ') + term.str + std::string(longestDecimal - decimal, ' ');
// 			adjusted[i] = tmp;
// 		}
// 
// 		// General checks
// 		bool stripMiddle = false;
// 		if (prod(shape) > 1000)
// 			stripMiddle = true;
// 
// 		// Edge case
// 		if (shape.size() == 2 && shape[1] == 1)
// 			stripMiddle = false;
// 
// 		auto res = utils::toString(adjusted, shape, 1, stripMiddle);
// 
// 		return res;
// 	}


	template<typename t>
	std::string Array<t>::toString() const
	{
		if (isZeroDim)
			return std::to_string(dataStart[0]);

		size_t allocate = 1;

		std::vector<utils::strContainer> formatted(prod(shape), {"", 0});
		size_t longestIntegral = 0;
		size_t longestDecimal = 0;

		// General checks
		bool stripMiddle = false;
		if (prod(shape) > 1000)
			stripMiddle = true;

		// Edge case
		if (shape.size() == 2 && shape[1] == 1)
			stripMiddle = false;

		std::vector<uint64_t> currentIndex(shape.size(), 0);
		currentIndex[currentIndex.size() - 1] = -1;

		while (utils::incArr(currentIndex, shape))
		{
			uint64_t index = utils::ndToScalar(currentIndex, shape);
			bool skip = false;
			for (int i = 0; i < currentIndex.size(); i++)
			{
				if (stripMiddle && (currentIndex[i] > 3 && currentIndex[i] < shape[i] - 3))
				{
					skip = true;
					break;
				}
			}

			if (!skip)
			{
				formatted[index] = utils::formatNumerical(dataStart[index]);

				if (formatted[index].decimalPoint > longestIntegral)
					longestIntegral = formatted[index].decimalPoint;

				if (formatted[index].str.length() >= formatted[index].decimalPoint && formatted[index].str.length() - formatted[index].decimalPoint > longestDecimal)
					longestDecimal = formatted[index].str.length() - formatted[index].decimalPoint;
			}
		}

		std::vector<std::string> adjusted(formatted.size(), "");

		for (size_t i = 0; i < formatted.size(); i++)
		{
			if (formatted[i].str.empty())
				continue;

			const auto &term = formatted[i];
			auto decimal = term.str.length() - term.decimalPoint - 1;

			size_t bufferLeft = 0;

			auto tmp = std::string(longestIntegral - term.decimalPoint, ' ') + term.str + std::string(longestDecimal - decimal, ' ');
			adjusted[i] = tmp;
		}

		auto res = utils::toString(adjusted, shape, 1, stripMiddle);

		return res;
	}

}
