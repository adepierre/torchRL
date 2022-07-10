#include "Pendulum/PendulumEnv.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

PendulumEnv::PendulumEnv(const unsigned int seed) : AbstractEnv(seed)
{
    theta = M_PI_2;
    thetadot = 0.0f;
    last_action = 0.0f;
}

PendulumEnv::~PendulumEnv()
{

}

int64_t PendulumEnv::GetObservationSize() const
{
    return 3;
}

int64_t PendulumEnv::GetActionSize() const
{
    return 1;
}

void PendulumEnv::ResetImpl()
{
    theta = std::uniform_real_distribution<float>(-M_PI, M_PI)(random_engine);
    thetadot = std::uniform_real_distribution<float>(-1.0f, 1.0f)(random_engine);
    last_action = 0.0f;
}

void PendulumEnv::RenderImpl()
{
    std::cout 
        << std::setw(8) << std::fixed << std::setprecision(4) << remainderf(theta, 2 * M_PI) 
        << std::setw(8) << std::fixed << std::setprecision(4) << thetadot 
        << std::setw(8) << std::fixed << std::setprecision(4) << last_action
        << std::endl;
}

StepResult PendulumEnv::StepImpl(const torch::Tensor& action)
{
    const float a = std::min(2.0f, std::max(-2.0f, action.item<float>()));
    const float normalized_theta = remainderf(theta, 2 * M_PI); // normalized between -M_PI and M_PI
    const float pos_reward = normalized_theta * normalized_theta + 0.1f * thetadot * thetadot + 0.001f * a * a;

    thetadot = thetadot + (3.0f * 10.0f / 2.0f * std::sin(theta) + 3.0f * a) * 0.05f;
    thetadot = std::min(8.0f, std::max(-8.0f, thetadot));
    theta = theta + thetadot * 0.05f;
    last_action = a;

    torch::Tensor obs = GetObs();
    const TerminalState is_final = current_episode_length == 200 ? TerminalState::Timeout : TerminalState::NotTerminal;

    return StepResult{obs, -pos_reward, is_final};
}

torch::Tensor PendulumEnv::GetObs() const
{
    torch::Tensor output = torch::zeros({ 3 });
    float* data = output.data<float>();

    data[0] = std::cos(theta);
    data[1] = std::sin(theta);
    data[2] = thetadot;

    return output;
}
