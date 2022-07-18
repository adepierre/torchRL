#pragma once

#include <memory>
#include <vector>

#include "torchrl/rl/Policy.hpp"

struct PPOArgs;
class VectorizedEnv;
class RolloutBuffer;

class PPO
{
public:
    PPO(VectorizedEnv& env_, const PPOArgs& args);
    ~PPO();

    /// @brief Start a PPO training
    /// @param total_timesteps Number of timesteps to play
    /// @param log_console If true will log training data to console
    /// @param draw_curves If true, will draw training curves (assuming WITH_IMPLOT, otherwise does nothing)
    /// @return The total training time (in s)
    float Learn(const uint64_t total_timesteps, const bool log_console = true, const bool draw_curves = true);

    /// @brief Play for num_episode and render the env
    /// @param num_episode The number of episode to play, if 0 will ask user to continue
    /// @param render If true, render the env between each decision and print the episode results to console
    /// @return A vector of num_episode pairs <episode length, episode reward>
    std::vector<std::pair<uint64_t, float> > Play(const uint64_t num_episode = 0, const bool render = true);

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
