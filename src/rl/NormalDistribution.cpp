#include "torchrl/rl/NormalDistribution.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

const static float half_log_2_pi = 0.5f * std::log(2.0f * M_PI);

NormalDistribution::NormalDistribution(const torch::Tensor& mean_, const torch::Tensor& std_)
{
    event_dim = mean_.size(-1);
    mean = mean_;
    std = std_;
}

NormalDistribution::~NormalDistribution()
{

}

torch::Tensor NormalDistribution::Sample(const int64_t N)
{
    torch::Tensor eps = torch::normal(0.0, 1.0, { N, event_dim });
    return mean + eps * std;
}

torch::Tensor NormalDistribution::LogProb(const torch::Tensor& samples)
{
    return (
        - std.log()
        - half_log_2_pi
        - torch::pow(samples - mean, 2) / (2.0f * torch::pow(std, 2))
        ).sum(1, true);
}

torch::Tensor NormalDistribution::Entropy()
{
    return (0.5f + half_log_2_pi + torch::log(std)).sum(-1, true);
}
