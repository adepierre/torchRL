#include "torchrl/rl/Policy.hpp"
#include "torchrl/rl/NormalDistribution.hpp"

PolicyImpl::PolicyImpl(const int64_t obs_dim, const int64_t action_dim, const bool ortho_init, const float init_log_std)
{
    pi_net = register_module("pi_net", MLP(obs_dim, 64, action_dim));
    v_net = register_module("v_net", MLP(obs_dim, 64, 1));

    log_std = register_parameter("log_std", torch::ones({ action_dim }) * init_log_std);

    if (ortho_init)
    {
        pi_net->InitOrtho(std::sqrtf(2.0f), 0.01f);
        v_net->InitOrtho(std::sqrtf(2.0f), 1.0f);
    }
}

PolicyImpl::~PolicyImpl()
{

}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PolicyImpl::forward(const torch::Tensor& observations, const bool deterministic)
{
    torch::Tensor action_means = pi_net(observations);
    torch::Tensor values = v_net(observations);

    NormalDistribution dist(action_means, log_std.exp());
    torch::Tensor actions = deterministic ? action_means : dist.Sample(observations.size(0));

    return { actions, values, dist.LogProb(actions) };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PolicyImpl::EvaluateActions(const torch::Tensor& observations, const torch::Tensor& actions)
{
    torch::Tensor action_means = pi_net(observations);
    torch::Tensor values = v_net(observations);

    NormalDistribution dist(action_means, log_std.exp());

    return { values, dist.LogProb(actions), dist.Entropy() };
}

torch::Tensor PolicyImpl::PredictValues(const torch::Tensor& observations)
{
    return v_net(observations);
}
