#pragma once

#include <torch/torch.h>


class NormalDistribution
{
public:
    NormalDistribution(const torch::Tensor& mean_, const torch::Tensor& std_);
    ~NormalDistribution();

    torch::Tensor Sample(const int64_t N);
    torch::Tensor LogProb(const torch::Tensor& samples);
    torch::Tensor Entropy();

private:
    int64_t event_dim;
    torch::Tensor mean;
    torch::Tensor std;
};
