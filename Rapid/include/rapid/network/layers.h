#pragma once

#include "../array.h"
#include "defaultedTypes.h"

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


		template<typename t, ndarray::ArrayLocation loc>
		struct ReluOutput
		{
			ndarray::Array<t, loc> out;
			Cache<t, loc> cache;
		};

		template<typename t, ndarray::ArrayLocation loc>
		struct ReluBackwardOutput
		{
			ndarray::Array<t, loc> dx;
		};

		template<typename t, ndarray::ArrayLocation loc>
		struct BatchnormParam
		{
			std::string mode;
			defaults::Scalar<t> eps{0, 1e-15, 0};
			defaults::Scalar<t> momentum{0, 0.9, 0};
			defaults::NDArray<t, loc> runningMean;
			defaults::NDArray<t, loc> runningVariance;
		};

		template<typename t, ndarray::ArrayLocation loc>
		struct BatchnormCache
		{
			ndarray::Array<t, loc> xNorm;
			t gamma;
			t beta;
			ndarray::Array<t, loc> sample_mean;
			ndarray::Array<t, loc> sample_variance;
			ndarray::Array<t, loc> x;
			t eps;
		};

		template<typename t, ndarray::ArrayLocation loc>
		struct BatchnormOutput
		{
			ndarray::Array<t, loc> out;
			BatchnormCache<t, loc> cache;
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

			auto dx = dOut.dot(cache.w.transposed()).reshaped(shapes);
			auto dw = (z.transposed()).dot(dOut);
			auto db = (ndarray::ones<t>({N})).dot(dOut);

			return {{dx, dw, db}};
		}

		template<typename t, ndarray::ArrayLocation loc>
		inline ReluOutput<t, loc> reluForward(const ndarray::Array<t, loc> &x)
		{
			auto out = relu(x);
			return {out, x};
		}

		template<typename t, ndarray::ArrayLocation loc>
		inline ReluBackwardOutput<t, loc> reluBackward(const ndarray::Array<t, loc> &dOut, const ndarray::Array<t, loc> &cache)
		{
			auto dx = ndarray::greater(cache, 0) * dOut;
			return {dx};
		}

		/*
		Forward pass for batch normalization.

		During training the sample mean and (uncorrected) sample variance are
		computed from minibatch statistics and used to normalize the incoming data.
		During training we also keep an exponentially decaying running mean of the mean
		and variance of each feature, and these averages are used to normalize data
		at test-time.

		At each timestep we update the running averages for mean and variance using
		an exponential decay based on the momentum parameter:

		running_mean = momentum * running_mean + (1 - momentum) * sample_mean
		running_var = momentum * running_var + (1 - momentum) * sample_var

		Note that the batch normalization paper suggests a different test-time
		behavior: they compute sample mean and variance for each feature using a
		large number of training images rather than using a running average. For
		this implementation we have chosen to use running averages instead since
		they do not require an additional estimation step; the torch7 implementation
		of batch normalization also uses running averages.

		Input:
		- x: Data of shape (N, D)
		- gamma: Scale parameter of shape (D,)
		- beta: Shift parameter of shape (D,)
		- bn_param: Dictionary with the following keys:
		  - mode: 'train' or 'test'; required
		  - eps: Constant for numeric stability
		  - momentum: Constant for running mean / variance.
		  - running_mean: Array of shape (D,) giving running mean of features
		  - running_var Array of shape (D,) giving running variance of features

		Returns a tuple of:
		- out: of shape (N, D)
		- cache: A tuple of values needed in the backward pass
		*/
		template<typename t, ndarray::ArrayLocation loc>
		inline BatchnormOutput<t, loc> batchnormForward(const ndarray::Array<t, loc> &x, const ndarray::Array<t, loc> &gamma,
														const ndarray::Array<t, loc> &beta, const BatchnormParam<t, loc> &batchnormParam)
		{
			auto &mode = batchnormParam.mode;
			auto eps = batchnormParam.eps.getValue();
			auto momentum = batchnormParam.momentum.getValue();

			auto D = x.shape[1];

			ndarray::Array<t, loc> runningMean;
			if (batchnormParam.runningMean.initialized)
				runningMean = batchnormParam.runningMean.getValue();
			else
				runningMean = ndarray::zeros<t, loc>({D});

			ndarray::Array<t, loc> runningVariance;
			if (batchnormParam.runningVariance.initialized)
				runningVariance = batchnormParam.runningVariance.getValue();
			else
				runningVariance = ndarray::zeros<t, loc>({D});

			// if (mode == "train")
			// {
			// 	auto sampleMean = ndarray::mean(x, 0);
			// 	auto sample_var = ndarray::var(x, 0);
			// 	auto x_norm = (x - sampleMean) / ndarray::sqrt(sample_var + eps);
			// 	auto out = gamma * x_norm + beta;
			// 	auto cache = (x_norm, gamma, beta, sample_mean, sample_var, x, eps);
			// 	auto running_mean = momentum * running_mean + (1 - momentum) * sample_mean;
			// 	auto running_var = momentum * running_var + (1 - momentum) * sample_var;
			// }
		}
	}
}
