#include "torchrl/envs/RunningMeanStd.hpp"

RunningMeanStd::RunningMeanStd(const c10::IntArrayRef shape, const float epsilon)
{
    mean = torch::zeros(shape).set_requires_grad(false);
    var = torch::ones(shape).set_requires_grad(false);
    count = epsilon;
}

RunningMeanStd::~RunningMeanStd()
{

}

const torch::Tensor& RunningMeanStd::GetMean() const
{
    return mean;
}

const torch::Tensor& RunningMeanStd::GetVar() const
{
    return var;
}

void RunningMeanStd::Update(const torch::Tensor& batch)
{
    torch::NoGradGuard no_grad;

    torch::Tensor delta = batch.mean(0) - mean;
    const int64_t N = batch.size(0);
    const float tot_count = count + N;

    mean = mean + delta * N / tot_count;
    var = (var * count + batch.var(0, false) * N + delta * delta * count * N / tot_count) / tot_count;
    count = tot_count;
}

void RunningMeanStd::Save(const std::string& path) const
{
    torch::save({ mean, var, torch::zeros({}) + count }, path);
}

void RunningMeanStd::Load(const std::string& path)
{
    std::vector<torch::Tensor> data;
    torch::load(data, path);
    mean = data[0];
    var = data[1];
    count = data[2].item<float>();
}

