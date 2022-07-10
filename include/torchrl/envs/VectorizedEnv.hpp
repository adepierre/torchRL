#pragma once
#include <vector>
#include <memory>

#include "torchrl/envs/AbstractEnv.hpp"

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
	VectorizedEnv();
	virtual ~VectorizedEnv();

	size_t GetNumEnvs() const;

	int64_t GetObservationSize() const;
	int64_t GetActionSize() const;

	/// @brief Reset all envs
	/// @return a {N, GetObservationSize()} tensor with all envs obs
	virtual torch::Tensor Reset();

	/// @brief Perform one step for each env
	/// @param action a {N, GetActionSize()} tensor with actions for each env
	/// @return VectorizedStepResult object with results for each env
	virtual VectorizedStepResult Step(const torch::Tensor& action);

	/// @brief Render each envs
	/// @param wait_ms time to wait in ms after the render is complete
	void Render(const size_t wait_ms = 0);

	/// @brief Get the current observation of all the envs
	/// @return a tensor of size {N, GetObservationSize()}
	virtual torch::Tensor GetObs() const;

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
		for (size_t i = 0; i < N; ++i)
		{
			envs.push_back(std::make_unique<Env>(seed + i));
		}


		obs_size = envs[0]->GetObservationSize();
		act_size = envs[0]->GetActionSize();

		PostCreateEnvs(N);
	}

protected:
	/// @brief Called after env are created
	/// @param N Number of created envs
	virtual void PostCreateEnvs(const int N);

protected:
	std::vector<std::unique_ptr<AbstractEnv>> envs;

	int64_t obs_size;
	int64_t act_size;
};
