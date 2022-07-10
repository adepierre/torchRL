#include "torchrl/envs/NormalizedVectorizedEnv.hpp"

#include <stdexcept>

NormalizedVectorizedEnv::NormalizedVectorizedEnv(
    const bool training_,
    const bool norm_obs_, const bool norm_reward_, 
    const float max_obs_, const float max_reward_, 
    const float discount_factor_, const float epsilon_
)
{
    training = training_;
    norm_obs = norm_obs_;
    norm_reward = norm_reward_;
    max_obs = max_obs_;
    max_reward = max_reward_;
    discount_factor = discount_factor_;
    epsilon = epsilon_;
}

NormalizedVectorizedEnv::~NormalizedVectorizedEnv()
{

}

void NormalizedVectorizedEnv::SetTraining(const bool t)
{
    training = t;
}

torch::Tensor NormalizedVectorizedEnv::Reset()
{
    torch::Tensor obs = VectorizedEnv::Reset();
    UpdateObs(obs);

    returns = torch::zeros({ static_cast<int64_t>(envs.size()) });

    return NormalizeObs(obs);
}

VectorizedStepResult NormalizedVectorizedEnv::Step(const torch::Tensor& action)
{
    VectorizedStepResult out = VectorizedEnv::Step(action);

    torch::Tensor normalizer_obs = torch::zeros_like(out.obs);

    for (size_t i = 0; i < envs.size(); ++i)
    {
        if (out.terminal_states[i] == TerminalState::NotTerminal)
        {
            normalizer_obs[i] = out.obs[i];
        }
        else
        {
            normalizer_obs[i] = out.new_episode_obs[i];
        }
    }

    UpdateObs(normalizer_obs);
    UpdateReward(out.rewards);

    out.obs = NormalizeObs(out.obs);
    out.rewards = NormalizeReward(out.rewards);
    for (size_t i = 0; i < envs.size(); ++i)
    {
        if (out.new_episode_obs[i].numel())
        {
            out.new_episode_obs[i] = NormalizeObs(out.new_episode_obs[i].unsqueeze(0)).squeeze(0);
        }
    }

    return out;
}

torch::Tensor NormalizedVectorizedEnv::GetObs() const
{
    torch::Tensor obs = VectorizedEnv::GetObs();
    return NormalizeObs(obs);
}

void NormalizedVectorizedEnv::PostCreateEnvs(const int N)
{
    returns = torch::zeros({ N });

    obs_rms = RunningMeanStd({ obs_size });
    ret_rms = RunningMeanStd({  });
}

torch::Tensor NormalizedVectorizedEnv::NormalizeObs(const torch::Tensor& obs) const
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

torch::Tensor NormalizedVectorizedEnv::NormalizeReward(const torch::Tensor& reward) const
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

void NormalizedVectorizedEnv::UpdateObs(const torch::Tensor& obs)
{
    if (training && norm_obs)
    {
        obs_rms.Update(obs);
    }
}

void NormalizedVectorizedEnv::UpdateReward(const torch::Tensor& reward)
{
    if (training && norm_reward)
    {
        returns = returns + discount_factor * reward;
        ret_rms.Update(returns);
    }
}
