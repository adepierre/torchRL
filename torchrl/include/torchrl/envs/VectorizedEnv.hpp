#pragma once
#include <vector>
#include <memory>

#include "torchrl/envs/AbstractEnv.hpp"
#include "torchrl/envs/RunningMeanStd.hpp"

struct VectorizedStepResult
{
	/// @brief new observations for each env, shape {N, o}
	torch::Tensor obs;
	/// @brief reward obtained after action for each env, shape {N}
	torch::Tensor rewards;
	/// @brief termination state of the episode for each env
	std::vector<TerminalState> terminal_states;
	/// @brief if terminal_state != NotTerminal, new obs after a reset operation, else empty (for each env)
	std::vector<torch::Tensor> new_episode_obs;
	/// @brief if terminal_state != NotTerminal, total reward got during the episode, else 0 (for each env)
	std::vector<float> episodes_tot_reward;
	/// @brief if terminal_state != NotTerminal, total number of steps during the episode, else 0 (for each env)
	std::vector<uint64_t> episodes_tot_length;
};

/// @brief A vectorized env runs multiple envs, allowing batch
/// inference in PPO aglorithm for better efficiency
class VectorizedEnv
{
public:
	VectorizedEnv(
		const bool norm_obs_ = true, const bool norm_reward_ = true,
		const float max_obs_ = 10.0f, const float max_reward_ = 10.0f,
		const float discount_factor_ = 0.99f, const float epsilon_ = 1e-8f
	);
	~VectorizedEnv();

	void SetTraining(const bool b);

	int64_t GetNumEnvs() const;
	int64_t GetObservationSize() const;
	int64_t GetActionSize() const;

	/// @brief Reset all envs
	/// @return a {N, GetObservationSize()} tensor with all envs obs
	torch::Tensor Reset();

	/// @brief Perform one step for each env
	/// @param action a {N, GetActionSize()} tensor with actions for each env
	/// @return VectorizedStepResult object with results for each env
	VectorizedStepResult Step(const torch::Tensor& action);

	/// @brief Render each envs
	/// @param wait_ms time to wait in ms after the render is complete
	void Render(const uint64_t wait_ms = 0);

	/// @brief Get the current observation of all the envs
	/// @return a tensor of size {N, GetObservationSize()}
	torch::Tensor GetObs() const;

	/// @brief Save this env parameters (normalizers) to a specific path
	/// @param path Directory in which data should be saved
	void Save(const std::string& path) const;

	/// @brief Load parameters (normalizers) from a specific path
	/// @param path Directory from which data should be loaded
	void Load(const std::string& path);

	/// @brief Populate the vectorized env with N env of type Env
	/// @tparam Env An Env deriving from AbstractEnv
	/// @param N Number of environments
	/// @param seed Base random seed. Will be incremented for each env. If 0 a random one is chosen.
	template<class Env>
	void CreateEnvs(const int N, unsigned int seed = 0)
	{
		if (N == 0)
		{
			throw std::runtime_error("Cannot create 0 env in VectorizedEnv");
		}

		if (seed == 0)
		{
			seed = std::random_device()();
		}

		envs.clear();
		envs.reserve(N);
		for (int i = 0; i < N; ++i)
		{
			envs.push_back(std::make_unique<Env>(seed + i));
		}

		num_envs = envs.size();
		obs_size = envs[0]->GetObservationSize();
		act_size = envs[0]->GetActionSize();

		if (norm_obs)
		{
			obs_rms = RunningMeanStd({ obs_size });
		}
		if (norm_reward)
		{
			returns = torch::zeros({ N }).set_requires_grad(false);
			ret_rms = RunningMeanStd({  });
		}
	}

private:
	torch::Tensor NormalizeObs(const torch::Tensor& obs) const;
	torch::Tensor NormalizeReward(const torch::Tensor& reward) const;
	void UpdateObs(const torch::Tensor& obs);
	void UpdateReward(const torch::Tensor& reward);

protected:
	std::vector<std::unique_ptr<AbstractEnv>> envs;

	int64_t num_envs;
	int64_t obs_size;
	int64_t act_size;

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
