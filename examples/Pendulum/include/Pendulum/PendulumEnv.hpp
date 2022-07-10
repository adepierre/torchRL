#pragma once

#include "torchrl/envs/AbstractEnv.hpp"

class PendulumEnv : public AbstractEnv
{
public:
	PendulumEnv(const unsigned int seed = 0);
	virtual ~PendulumEnv();

    virtual int64_t GetObservationSize() const override;
    virtual int64_t GetActionSize() const override;

    virtual void ResetImpl() override;
    virtual void RenderImpl() override;
    virtual StepResult StepImpl(const torch::Tensor& action) override;
    virtual torch::Tensor GetObs() const override;

private:
    float theta;
    float thetadot;

    float last_action;
};
