#pragma once

#include "torch/torch.h"
#include <random>

enum class TerminalState : char
{
    NotTerminal,
    Terminal,
    Timeout
};

/// @brief Output result of a Step action
struct StepResult
{
    /// @brief new observation of the env after action is taken
    torch::Tensor obs;
    /// @brief reward obtained after action
    float reward;
    /// @brief termination state of the episode
    TerminalState terminal_state;
    /// @brief if terminal_state != NotTerminal, new obs after a reset operation, else empty
    torch::Tensor new_episode_obs;
    /// @brief if terminal_state != NotTerminal, total reward got during the episode, else 0
    float tot_reward = 0.0f;
    /// @brief if terminal_state != NotTerminal, total number of steps during the episode, else 0
    uint64_t tot_steps = 0;

};

/// @brief Abstract env base class
class AbstractEnv
{
public:
    /// @brief Base abstract constructor for all envs 
    /// @param seed Random seed, if 0, will be randomly generated
    AbstractEnv(const unsigned int seed = 0);
	virtual ~AbstractEnv();

    /// @brief Observation dim getter
    /// @return the flatten obs dimension
    virtual int64_t GetObservationSize() const = 0;
    /// @brief Action space dim getter
    /// @return the flatten action dimension
    virtual int64_t GetActionSize() const = 0;

    /// @brief Reset the environment in a new state, totally independant of previous one
    /// @return the observation resulting from the new state
    torch::Tensor Reset();

    /// @brief Perform action on the env, reset if ends up in a terminal state
    /// @param action the action to perform
    /// @return A step result object
    StepResult Step(const torch::Tensor& action);
    
    /// @brief Render the env
    /// @param wait_ms time to wait in ms after the render is complete
    void Render(const size_t wait_ms = 0);

    /// @brief Get the current observation of the env
    /// @return a tensor of size {1, GetObservationSize()}
    virtual torch::Tensor GetObs() const = 0;

protected:
    virtual void ResetImpl() = 0;
    virtual void RenderImpl() = 0;
    virtual StepResult StepImpl(const torch::Tensor& action) = 0;

protected:
    std::mt19937 random_engine;

    size_t current_episode_length;
    float current_episode_reward;
};
