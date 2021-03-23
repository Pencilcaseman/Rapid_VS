#pragma once

#include "../internal.h"
#include "../math.h"
#include "../array.h"

namespace rapid
{
	namespace network
	{
		namespace optim
		{
			template<typename _Ty>
			struct LearningRate
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

			template<typename _Ty, ndarray::ArrayLocation loc = ndarray::CPU>
			struct Config
			{
				int initialized = 0;               // Is the context initialized

				LearningRate<_Ty> learningRate;    // Scalar learning rate
				Scalar<_Ty> momentum;              // Scalar between 0 and 1 representing momentum value
				NDArray<_Ty, loc> velocity;        // ndarray::Array of same shape as w storing a moving average of gradients
				Scalar<_Ty> decayRate;             // Scalar between 0 and 1 storing the decay rate for the squared gradient cache
				Scalar<_Ty> epsilon;               // Small scalar used for smoothing to avoid division by zero
				NDArray<_Ty, loc> cache;           // Moving average of second moments of gradients
				Scalar<_Ty> beta1;                 // Decay rate for moving average of first moment gradient
				Scalar<_Ty> beta2;                 // Decay rate for moving average of second moment gradient
				NDArray<_Ty, loc> m;               // Moving average of gradient
				NDArray<_Ty, loc> v;               // Moving average of squared gradient
				uint64_t t = 0;                    // Iteration number
			};

			template<typename _Ty, ndarray::ArrayLocation loc = ndarray::CPU>
			struct OptimOutput
			{
				ndarray::Array<_Ty, loc> weight;
				Config<_Ty, loc> config;
			};

			template<typename _Ty, ndarray::ArrayLocation loc = ndarray::CPU>
			class Optimizer
			{
			public:
				virtual inline OptimOutput<_Ty, loc> apply(ndarray::Array<_Ty, loc> &w,
														   const ndarray::Array<_Ty, loc> &dw,
														   Config<_Ty, loc> &config) = 0;

				virtual inline OptimOutput<_Ty, loc> apply(ndarray::Array<_Ty, loc> &w,
														   const ndarray::Array<_Ty, loc> &dw) = 0;
			};

			/// <summary>
			/// Vanilla stochastic gradient descent
			/// </summary>
			/// <typeparam name="_Ty"></typeparam>
			template<typename _Ty, ndarray::ArrayLocation loc = ndarray::CPU>
			class sgd : public Optimizer<_Ty, loc>
			{
			public:
				sgd() = default;

				inline OptimOutput<_Ty, loc> apply(ndarray::Array<_Ty, loc> &w,
												   const ndarray::Array<_Ty, loc> &dw,
												   Config<_Ty, loc> &config) override
				{
					if (config.initialized == 0)
					{
						config.initialized = 1;
						config.learningRate.initialized = 0;
						config.learningRate.defaultValue = 1e-2;
					}

					w -= config.learningRate.getValue() * dw;
					return {w, config};
				}

				inline OptimOutput<_Ty, loc> apply(ndarray::Array<_Ty, loc> &w,
												   const ndarray::Array<_Ty, loc> &dw) override
				{
					Config<_Ty, loc> config;
					config.initialized = 1;
					config.learningRate.defaultValue = 1e-2;

					return apply(w, dw, config);
				}
			};

			/// <summary>
			/// Stochastic gradient descent with momentum
			/// </summary>
			/// <typeparam name="_Ty"></typeparam>
			template<typename _Ty, ndarray::ArrayLocation loc = ndarray::CPU>
			class sgdMomentum : public Optimizer<_Ty, loc>
			{
			public:
				sgdMomentum() = default;

				inline OptimOutput<_Ty, loc> apply(ndarray::Array<_Ty, loc> &w,
												   const ndarray::Array<_Ty, loc> &dw,
												   Config<_Ty, loc> &config) override
				{
					if (config.initialized == 0)
					{
						config.initialized = 1;
						config.learningRate.defaultValue = 1e-2;
						config.momentum.defaultValue = 0.9;
						config.velocity.defaultValue.set(zerosLike(w));
					}

					// Momentum update formula -- also update velocity
					auto v = config.velocity.getValue();
					v.set(config.momentum.getValue() * v - config.learningRate.getValue() * dw);
					auto nextW = w + v;
					config.velocity.setValue(v);

					return {nextW, config};
				}

				inline OptimOutput<_Ty, loc> apply(ndarray::Array<_Ty, loc> &w,
												   const ndarray::Array<_Ty, loc> &dw) override
				{
					Config<_Ty, loc> config;

					config.initialized = 1;
					config.learningRate.defaultValue = 1e-2;
					config.momentum.defaultValue = 0.9;
					config.velocity.defaultValue.set(zerosLike(w));

					return apply(w, dw, config);
				}
			};

			/// <summary>
			/// Use the RMSProp update rule -- A moving average of squared gradients
			/// to set adaptive per-parameter learning rates
			/// </summary>
			/// <typeparam name="_Ty"></typeparam>
			template<typename _Ty, ndarray::ArrayLocation loc = ndarray::CPU>
			class rmsprop : public Optimizer<_Ty, loc>
			{
			public:
				rmsprop() = default;

				inline OptimOutput<_Ty, loc> apply(ndarray::Array<_Ty, loc> &x,
												   const ndarray::Array<_Ty, loc> &dx,
												   Config<_Ty, loc> &config) override
				{
					if (config.initialized == 0)
					{
						config.initialized = 1;
						config.learningRate.defaultValue = 1e-2;
						config.decayRate.defaultValue = 0.99;
						config.epsilon.defaultValue = 1e-8;
						config.cache.defaultValue.set(ndarray::zerosLike(x));
					}

					auto cache = config.cache.getValue();
					cache.set(config.decayRate.getValue() * cache + (1 - config.decayRate.getValue()) * (dx * dx));
					auto nextX = x - config.learningRate.getValue() * dx / (sqrt(cache) + config.epsilon.getValue());
					config.cache.setValue(cache);

					return {nextX, config};
				}

				inline OptimOutput<_Ty, loc> apply(ndarray::Array<_Ty, loc> &x,
												   const ndarray::Array<_Ty, loc> &dx) override
				{
					Config<_Ty, loc> config;

					config.initialized = 1;
					config.learningRate.defaultValue = 1e-2;
					config.decayRate.defaultValue = 0.99;
					config.epsilon.defaultValue = 1e-8;
					config.cache.defaultValue.set(zerosLike(x));

					return apply(x, dx, config);
				}
			};

			/// <summary>
			/// Uses the Adam update rule, which incorporates moving averages of
			/// both the gradient and its square, and a bias correction term
			/// </summary>
			/// <typeparam name="_Ty"></typeparam>
			template<typename _Ty, ndarray::ArrayLocation loc = ndarray::CPU>
			class adam : public Optimizer<_Ty, loc>
			{
			public:
				adam() = default;

				inline OptimOutput<_Ty, loc> apply(ndarray::Array<_Ty, loc> &x,
												   const ndarray::Array<_Ty, loc> &dx,
												   Config<_Ty, loc> &config) override
				{
					if (config.initialized == 0)
					{
						config.initialized = 1;
						config.learningRate.defaultValue = 1e-3;
						config.beta1.defaultValue = 0.9;
						config.beta2.defaultValue = 0.999;
						config.epsilon.defaultValue = 1e-8;
						config.m.defaultValue.set(ndarray::zerosLike(x));
						config.v.defaultValue.set(ndarray::zerosLike(x));
						config.t = 0;
					}
					 
					auto m = config.m.getValue();
					auto v = config.v.getValue();
					auto t = config.t;

					t++;
					m.set(config.beta1.getValue() * m + (1 - config.beta1.getValue()) * dx);
					auto mCorr = m / (1. - std::pow(config.beta1.getValue(), (_Ty) t));
					v.set(config.beta2.getValue() * v + (1 - config.beta2.getValue()) * (dx * dx));
					auto vCorr = v / (1 - std::pow(config.beta2.getValue(), (_Ty) t));
					auto nextX = x - config.learningRate.getValue() * mCorr / (ndarray::sqrt(vCorr) + config.epsilon.getValue());
					config.m.setValue(m);
					config.v.setValue(v);
					config.t = t;

					return {nextX, config};
				}

				inline OptimOutput<_Ty, loc> apply(ndarray::Array<_Ty, loc> &x,
												   const ndarray::Array<_Ty, loc> &dx) override
				{
					Config<_Ty, loc> config;

					config.initialized = 1;
					config.learningRate.defaultValue = 1e-3;
					config.beta1.defaultValue = 0.9;
					config.beta2.defaultValue = 0.999;
					config.epsilon.defaultValue = 1e-8;
					config.m.defaultValue.set(ndarray::zerosLike(x));
					config.v.defaultValue.set(ndarray::zerosLike(x));
					config.t = 0;

					return apply(x, dx, config);
				}
			};
		}
	}
}
