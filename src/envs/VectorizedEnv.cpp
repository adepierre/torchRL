#include "torchrl/envs/VectorizedEnv.hpp"

#include <stdexcept>

VectorizedEnv::VectorizedEnv()
{
    obs_size = 0;
    act_size = 0;
}

VectorizedEnv::~VectorizedEnv()
{

}

size_t VectorizedEnv::GetNumEnvs() const
{
    return envs.size();
}

int64_t VectorizedEnv::GetObservationSize() const
{
    return obs_size;
}

int64_t VectorizedEnv::GetActionSize() const
{
    return act_size;
}

torch::Tensor VectorizedEnv::Reset()
{
    torch::Tensor obs = torch::zeros({ static_cast<int64_t>(envs.size()), obs_size });
    for (size_t i = 0; i < envs.size(); ++i)
    {
        obs[i] = envs[i]->Reset();
    }
    return obs;
}

VectorizedStepResult VectorizedEnv::Step(const torch::Tensor& action)
{
    torch::Tensor obs = torch::zeros({ static_cast<int64_t>(envs.size()), obs_size });
    torch::Tensor rewards = torch::zeros({ static_cast<int64_t>(envs.size()) });
    std::vector<TerminalState> terminal_states(envs.size());
    std::vector<torch::Tensor> new_episode_obs(envs.size());
    std::vector<float> tot_reward(envs.size());
    std::vector<uint64_t> tot_steps(envs.size());

    for (size_t i = 0; i < envs.size(); ++i)
    {
        StepResult res = envs[i]->Step(action[i]);
        obs[i] = res.obs;
        rewards[i] = res.reward;
        terminal_states[i] = res.terminal_state;
        new_episode_obs[i] = res.new_episode_obs;
        tot_reward[i] = res.tot_reward;
        tot_steps[i] = res.tot_steps;
    }

    return VectorizedStepResult{ obs, rewards, terminal_states, new_episode_obs, tot_reward, tot_steps };
}

void VectorizedEnv::Render(const size_t wait_ms)
{
    for (size_t i = 0; i < envs.size(); ++i)
    {
        envs[i]->Render();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
}

torch::Tensor VectorizedEnv::GetObs() const
{
    torch::Tensor obs = torch::zeros({ static_cast<int64_t>(envs.size()), obs_size });
    for (size_t i = 0; i < envs.size(); ++i)
    {
        obs[i] = envs[i]->GetObs();
    }
    return obs;
}

void VectorizedEnv::PostCreateEnvs(const int N)
{

}

