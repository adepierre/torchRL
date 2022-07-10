#include "torchrl/rl/MLP.hpp"

MLPImpl::MLPImpl(const int64_t num_in, const int64_t num_hidden, const int64_t out_num)
{
    l1 = register_module("l1", torch::nn::Linear(num_in, num_hidden));
    l2 = register_module("l2", torch::nn::Linear(num_hidden, num_hidden));
    
    out_layer = register_module("out_layer", torch::nn::Linear(num_hidden, out_num));
}

MLPImpl::~MLPImpl()
{

}

torch::Tensor MLPImpl::forward(const torch::Tensor& in)
{
    torch::Tensor out = torch::tanh(l1(in));
    out = torch::tanh(l2(out));

    return out_layer(out);
}

void MLPImpl::InitOrtho(const float gain_backbone, const float gain_out)
{
    torch::NoGradGuard no_grad;
    torch::nn::init::orthogonal_(l1->weight, gain_backbone);
    torch::nn::init::constant_(l1->bias, 0.0);
    torch::nn::init::orthogonal_(l2->weight, gain_backbone);
    torch::nn::init::constant_(l2->bias, 0.0);

    torch::nn::init::orthogonal_(out_layer->weight, gain_out);
    torch::nn::init::constant_(out_layer->bias, 0.0);
}
