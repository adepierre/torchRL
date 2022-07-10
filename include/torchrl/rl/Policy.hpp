#pragma once

#include "torch/torch.h"
#include "torchrl/rl/MLP.hpp"

class PolicyImpl : public torch::nn::Module
{
public:
    PolicyImpl(const int64_t obs_dim, const int64_t action_dim, const bool ortho_init = true, const float init_log_std = 0.0f);
    ~PolicyImpl();

    /// @brief Forward pass in all the networks (actor and critic)
    /// @param observations Observations
    /// @param deterministic Whether to sample or use deterministic actions
    /// @return A tuple <actions, values, log probabilities of the actions>
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const torch::Tensor& observations, const bool deterministic = false);

    /// @brief Evaluate actions according to the current policy, given the observations.
    /// @param observations Observations
    /// @param actions Actions
    /// @return A tuple <value, log likelihood of the actions, entropy of the action dist>
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> EvaluateActions(const torch::Tensor& observations, const torch::Tensor& actions);

    /// @brief Get the estimated values according to the current policy given the observations.
    /// @param observations Observations
    /// @return The estimated values
    torch::Tensor PredictValues(const torch::Tensor& observations);

private:
    MLP pi_net{ nullptr };
    MLP v_net{ nullptr };

    torch::Tensor log_std;
};
TORCH_MODULE(Policy);
