#include "torchrl/envs/VectorizedEnv.hpp"

#include <filesystem>

VectorizedEnv::VectorizedEnv(
    const bool norm_obs_, const bool norm_reward_,
    const float max_obs_, const float max_reward_,
    const float discount_factor_, const float epsilon_
)
{
    num_envs = 0;
    obs_size = 0;
    act_size = 0;

    training = true;
    norm_obs = norm_obs_;
    norm_reward = norm_reward_;
    max_obs = max_obs_;
    max_reward = max_reward_;
    discount_factor = discount_factor_;
    epsilon = epsilon_;
}

VectorizedEnv::~VectorizedEnv()
{

}

void VectorizedEnv::SetTraining(const bool b)
{
    training = b;
}

int64_t VectorizedEnv::GetNumEnvs() const
{
    return num_envs;
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
    torch::Tensor obs = torch::zeros({ num_envs, obs_size });
    for (int i = 0; i < num_envs; ++i)
    {
        obs[i] = envs[i]->Reset();
    }

    UpdateObs(obs);

    if (norm_reward)
    {
        returns = torch::zeros({ num_envs }).set_requires_grad(false);
    }

    return NormalizeObs(obs);
}

VectorizedStepResult VectorizedEnv::Step(const torch::Tensor& action)
{
    torch::Tensor obs = torch::zeros({ num_envs, obs_size });
    torch::Tensor normalizer_obs = torch::zeros({ num_envs, obs_size });
    torch::Tensor rewards = torch::zeros({ num_envs });
    std::vector<TerminalState> terminal_states(num_envs);
    std::vector<torch::Tensor> new_episode_obs(num_envs);
    std::vector<float> tot_reward(num_envs);
    std::vector<uint64_t> tot_steps(num_envs);

    for (int i = 0; i < num_envs; ++i)
    {
        StepResult res = envs[i]->Step(action[i]);
        obs[i] = res.obs;
        rewards[i] = res.reward;
        terminal_states[i] = res.terminal_state;
        new_episode_obs[i] = res.new_episode_obs;
        tot_reward[i] = res.tot_reward;
        tot_steps[i] = res.tot_steps;

        if (norm_obs)
        {
            if (res.terminal_state == TerminalState::NotTerminal)
            {
                normalizer_obs[i] = res.obs;
            }
            else
            {
                normalizer_obs[i] = res.new_episode_obs;
            }
        }
    }

    UpdateObs(normalizer_obs);
    UpdateReward(rewards);

    for (int i = 0; i < num_envs; ++i)
    {
        if (terminal_states[i] != TerminalState::NotTerminal)
        {
            if (norm_obs)
            {
                new_episode_obs[i] = NormalizeObs(new_episode_obs[i].unsqueeze(0)).squeeze(0);
            }
            if (norm_reward)
            {
                returns[i] = 0.0f;
            }
        }
    }

    return VectorizedStepResult{ NormalizeObs(obs), NormalizeReward(rewards), terminal_states, new_episode_obs, tot_reward, tot_steps };
}

void VectorizedEnv::Render(const uint64_t wait_ms)
{
    for (int i = 0; i < num_envs; ++i)
    {
        envs[i]->Render();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
}

torch::Tensor VectorizedEnv::GetObs() const
{
    torch::Tensor obs = torch::zeros({ num_envs, obs_size });
    for (int i = 0; i < num_envs; ++i)
    {
        obs[i] = envs[i]->GetObs();
    }
    return NormalizeObs(obs);
}

void VectorizedEnv::Save(const std::string& path) const
{
    if (norm_obs)
    {
        obs_rms.Save((std::filesystem::path(path) / "env_obs_rms.pt").string());
    }
    if (norm_reward)
    {
        ret_rms.Save((std::filesystem::path(path) / "env_ret_rms.pt").string());
    }
}

void VectorizedEnv::Load(const std::string& path)
{
    if (norm_obs)
    {
        obs_rms.Load((std::filesystem::path(path) / "env_obs_rms.pt").string());
    }
    if (norm_reward)
    {
        ret_rms.Load((std::filesystem::path(path) / "env_ret_rms.pt").string());
    }
}

torch::Tensor VectorizedEnv::NormalizeObs(const torch::Tensor& obs) const
{
    if (norm_obs)
    {
        return torch::clip((obs - obs_rms.GetMean()) / torch::sqrt(obs_rms.GetVar() + epsilon), -max_obs, max_obs);
    }
    else
    {
        return obs;
    }
}

torch::Tensor VectorizedEnv::NormalizeReward(const torch::Tensor& reward) const
{
    if (norm_reward)
    {
        return torch::clip(reward / torch::sqrt(ret_rms.GetVar() + epsilon), -max_reward, max_reward);
    }
    else
    {
        return reward;
    }
}

void VectorizedEnv::UpdateObs(const torch::Tensor& obs)
{
    if (training && norm_obs)
    {
        obs_rms.Update(obs);
    }
}

void VectorizedEnv::UpdateReward(const torch::Tensor& reward)
{
    if (training && norm_reward)
    {
        returns = returns + discount_factor * reward;
        ret_rms.Update(returns);
    }
}
