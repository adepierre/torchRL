#pragma once

#include "torch/torch.h"

#include "torchrl/envs/VectorizedEnv.hpp"
#include "torchrl/rl/Policy.hpp"

struct RolloutSample
{
    RolloutSample()
    {

    }

    RolloutSample(
        const torch::Tensor& observation_,
        const torch::Tensor& action_,
        const torch::Tensor& value_,
        const torch::Tensor& log_prob_,
        const torch::Tensor& advantage_,
        const torch::Tensor& returns_
    ) :
        observation(observation_),
        action(action_),
        value(value_),
        log_prob(log_prob_),
        advantage(advantage_),
        returns(returns_)
    {

    }

    RolloutSample(
        const torch::Tensor& observation_,
        const torch::Tensor& action_,
        const torch::Tensor& value_,
        const torch::Tensor& log_prob_
    ) :
        observation(observation_),
        action(action_),
        value(value_),
        log_prob(log_prob_)
    {

    }

    torch::Tensor observation;
    torch::Tensor action;
    torch::Tensor value;
    torch::Tensor log_prob;
    torch::Tensor advantage;
    torch::Tensor returns;
};

struct RolloutSampleBatchTransform : public torch::data::transforms::BatchTransform<std::vector<RolloutSample>, RolloutSample> 
{
    RolloutSample apply_batch(std::vector<RolloutSample> batch) override 
    {
        std::vector<torch::Tensor> observation, action, value, log_prob, advantage, returns;
        observation.reserve(batch.size());
        action.reserve(batch.size());
        value.reserve(batch.size());
        log_prob.reserve(batch.size());
        advantage.reserve(batch.size());
        returns.reserve(batch.size());

        for (auto& d : batch)
        {
            observation.push_back(d.observation);
            action.push_back(d.action);
            value.push_back(d.value);
            log_prob.push_back(d.log_prob);
            advantage.push_back(d.advantage);
            returns.push_back(d.returns);
        }

        return RolloutSample(
            torch::stack(observation),
            torch::stack(action),
            torch::stack(value),
            torch::stack(log_prob),
            torch::stack(advantage),
            torch::stack(returns)
        );
    }
};

class RolloutBuffer : public torch::data::Dataset<RolloutBuffer, RolloutSample>
{
public:
    RolloutBuffer(const size_t num_envs, const size_t reserve = 0);

    void Add(const torch::Tensor& obs, const torch::Tensor& action,
        const torch::Tensor& value, const torch::Tensor& log_prob,
        const torch::Tensor& reward, const std::vector<TerminalState>& episode_end);

    void Reset();

    RolloutSample get(size_t index) override;

    torch::optional<size_t> size() const override;

    void ComputeReturnsAndAdvantage(const torch::Tensor& value, 
        const float gamma, const float lambda_gae);

private:
    std::vector<std::vector<RolloutSample> > data;
    std::vector<std::vector<float> > rewards;
    std::vector<std::vector<bool> > episode_ends;
};
