#pragma once

#include "torchrl/envs/AbstractEnv.hpp"

class MountainCarContinuousEnv : public AbstractEnv
{
public:
    MountainCarContinuousEnv(const unsigned int seed = 0);
	virtual ~MountainCarContinuousEnv();

    virtual int64_t GetObservationSize() const override;
    virtual int64_t GetActionSize() const override;

    virtual void ResetImpl() override;
    virtual void RenderImpl() override;
    virtual StepResult StepImpl(const torch::Tensor& action) override;
    virtual torch::Tensor GetObs() const override;

private:
    float position;
    float velocity;

    float last_action;
};
