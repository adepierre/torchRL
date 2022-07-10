#pragma once

#include "torchrl/rl/Policy.hpp"

struct PPOArgs;
class VectorizedEnv;
class RolloutBuffer;

class PPO
{
public:
    PPO(VectorizedEnv& env_, const PPOArgs& args);
    ~PPO();

    void Learn(const uint64_t total_timesteps);

private:
    /// @brief Use the policy to play in the env and store the data in buffer
    /// @param buffer The rollout buffer to store data in
    /// @return A tuple <reward at the end of episodes, number of steps at the end of episodes, number of end of episodes>
    std::tuple<float, uint64_t, uint64_t> CollectRollouts(RolloutBuffer& buffer);

private:
    VectorizedEnv& env;
    const PPOArgs& args;

    Policy policy{ nullptr };
};
