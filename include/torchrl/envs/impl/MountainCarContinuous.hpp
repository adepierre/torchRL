#pragma once

#include "torchrl/envs/AbstractEnv.hpp"

class MountainCarContinuous : public AbstractEnv
{
public:
    MountainCarContinuous(const unsigned int seed = 0);
	virtual ~MountainCarContinuous();

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
