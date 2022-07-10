#include "torchrl/rl/RolloutBuffer.hpp"

RolloutBuffer::RolloutBuffer(const uint64_t num_envs, const uint64_t reserve)

{
    data = std::vector<std::vector<RolloutSample>>(num_envs);
    rewards = std::vector<std::vector<float>>(num_envs);
    episode_ends = std::vector<std::vector<bool>>(num_envs);
    if (reserve != 0)
    {
        for (int i = 0; i < num_envs; ++i)
        {
            data[i].reserve(reserve);
            rewards[i].reserve(reserve);
            episode_ends[i].reserve(reserve);
        }
    }
}

void RolloutBuffer::Add(const torch::Tensor& obs, const torch::Tensor& action,
    const torch::Tensor& value, const torch::Tensor& log_prob,
    const torch::Tensor& reward, const std::vector<TerminalState>& episode_end)
{
    for (int i = 0; i < data.size(); ++i)
    {
        data[i].push_back(RolloutSample(obs[i], action[i], value[i], log_prob[i]));
        rewards[i].push_back(reward[i].item<float>());
        episode_ends[i].push_back(episode_end[i] != TerminalState::NotTerminal);
    }
}

void RolloutBuffer::Reset()
{
    for (int i = 0; i < data.size(); ++i)
    {
        data[i].clear();
        rewards[i].clear();
        episode_ends[i].clear();
    }
}

RolloutSample RolloutBuffer::get(uint64_t index)
{
    uint64_t i = 0;
    while (index >= data[i].size())
    {
        index -= data[i].size();
        i += 1;
    }
    return data[i][index];
}

torch::optional<uint64_t> RolloutBuffer::size() const
{
    uint64_t size = 0;
    for (int i = 0; i < data.size(); ++i)
    {
        size += data[i].size();
    }
    return size;
}

void RolloutBuffer::ComputeReturnsAndAdvantage(const torch::Tensor& value,
    const float gamma, const float lambda_gae)
{
    torch::NoGradGuard no_grad;

    for (int env = 0; env < data.size(); ++env)
    {
        for (int i = data[env].size() - 1; i > -1; --i)
        {
            torch::Tensor next_value;
            torch::Tensor next_gae_lambda;
            if (i == data[env].size() - 1)
            {
                next_value = value[env];
                next_gae_lambda = torch::zeros_like(value[env]);
            }
            else
            {
                next_value = data[env][i + 1].value;
                next_gae_lambda = data[env][i + 1].advantage;
            }
            float next_step_same_episode = episode_ends[env][i] ? 0.0f : 1.0f;
            torch::Tensor delta = rewards[env][i] + gamma * next_value * next_step_same_episode - data[env][i].value;
            data[env][i].advantage = delta + gamma * lambda_gae * next_step_same_episode * next_gae_lambda;
            data[env][i].returns = data[env][i].advantage + data[env][i].value;
        }
    }
}
