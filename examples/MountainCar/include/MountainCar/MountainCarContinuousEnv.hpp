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
    static constexpr float min_action = -1.0f;
    static constexpr float max_action = 1.0f;
    static constexpr float min_position = -1.2f;
    static constexpr float max_position = 0.6f;
    static constexpr float max_speed = 0.07f;
    static constexpr float goal_position = 0.45f;
    static constexpr float power = 0.0015f;

    float position;
    float velocity;

    float last_action;
};
