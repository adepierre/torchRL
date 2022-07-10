#include "torchrl/envs/AbstractEnv.hpp"

AbstractEnv::AbstractEnv(const unsigned int seed)
{
    if (seed == 0)
    {
        std::random_device rd;
        random_engine = std::mt19937(rd());
    }
    else
    {
        random_engine = std::mt19937(seed);
    }
    current_episode_length = 0;
    current_episode_reward = 0.0f;
}

AbstractEnv::~AbstractEnv()
{

}

torch::Tensor AbstractEnv::Reset()
{
    current_episode_length = 0;
    current_episode_reward = 0.0f;
    ResetImpl();

    return GetObs();
}

StepResult AbstractEnv::Step(const torch::Tensor& action)
{
    current_episode_length += 1;
    StepResult result = StepImpl(action);
    current_episode_reward += result.reward;

    if (result.terminal_state != TerminalState::NotTerminal)
    {
        result.tot_reward = current_episode_reward;
        result.tot_steps = current_episode_length;
        result.new_episode_obs = Reset();
    }
    return result;
}

void AbstractEnv::Render(const size_t wait_ms)
{
    RenderImpl();
    if (wait_ms > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    }
}
