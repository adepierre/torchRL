#pragma once
#include <vector>
#include <memory>

#include "torchrl/envs/VectorizedEnv.hpp"
#include "torchrl/envs/RunningMeanStd.hpp"

/// @brief A vectorized env that can normalize obs/rewards
/// with a moving average/std
class NormalizedVectorizedEnv : public VectorizedEnv
{
public:
	NormalizedVectorizedEnv(
		const bool training_,
		const bool norm_obs_ = true, const bool norm_reward_ = true,
		const float max_obs_ = 10.0f, const float max_reward_ = 10.0f,
		const float discount_factor_ = 0.99f, const float epsilon_ = 1e-8f
	);
	virtual ~NormalizedVectorizedEnv();

	void SetTraining(const bool b);

	/// @brief Reset all envs
	/// @return a {N, GetObservationSize()} tensor with all envs obs
	virtual torch::Tensor Reset() override;

	/// @brief Perform one step for each env
	/// @param action a {N, GetActionSize()} tensor with actions for each env
	/// @return VectorizedStepResult object with normalized results for each env
	virtual VectorizedStepResult Step(const torch::Tensor& action) override;

	/// @brief Get the current observation of all the envs
	/// @return a tensor of size {N, GetObservationSize()}
	virtual torch::Tensor GetObs() const override;

protected:
	virtual void PostCreateEnvs(const int N) override;

private:
	torch::Tensor NormalizeObs(const torch::Tensor& obs) const;
	torch::Tensor NormalizeReward(const torch::Tensor& reward) const;
	void UpdateObs(const torch::Tensor& obs);
	void UpdateReward(const torch::Tensor& reward);

private:
	bool training;
	bool norm_obs;
	bool norm_reward;
	float max_obs;
	float max_reward;
	float discount_factor;
	float epsilon;

	RunningMeanStd obs_rms;
	RunningMeanStd ret_rms;
	/// @brief reward normalization is done with the discounted reward
	torch::Tensor returns;
};
