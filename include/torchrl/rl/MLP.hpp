#pragma once

#include "torch/torch.h"

class MLPImpl : public torch::nn::Module
{
public:
	MLPImpl(const int64_t num_in, const int64_t num_hidden, const int64_t out_num);
	~MLPImpl();

	torch::Tensor forward(const torch::Tensor& in);

	void InitOrtho(const float gain_backbone, const float gain_out);

private:
	torch::nn::Linear l1{ nullptr };
	torch::nn::Linear l2{ nullptr };

	torch::nn::Linear out_layer{ nullptr };
};
TORCH_MODULE(MLP);
