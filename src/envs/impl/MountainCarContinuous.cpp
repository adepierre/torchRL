#include "torchrl/envs/impl/MountainCarContinuous.hpp"

MountainCarContinuous::MountainCarContinuous(const unsigned int seed) : AbstractEnv(seed)
{
    position = 0.0f;
    velocity = 0.0f;
    last_action = 0.0f;
}

MountainCarContinuous::~MountainCarContinuous()
{

}

int64_t MountainCarContinuous::GetObservationSize() const
{
    return 2;
}

int64_t MountainCarContinuous::GetActionSize() const
{
    return 1;
}

void MountainCarContinuous::ResetImpl()
{
    position = std::uniform_real_distribution<float>(-0.6f, 0.4f)(random_engine);
    velocity = 0.0f;
    last_action = 0.0f;
}

void MountainCarContinuous::RenderImpl()
{
    std::cout << "Position: " << std::setprecision(4) << position << " | Velocity: " << velocity << " | a: " << last_action << std::endl;
}

StepResult MountainCarContinuous::StepImpl(const torch::Tensor& action)
{
    const float raw_a = action.item<float>();
    const float a = std::min(1.0f, std::max(-1.0f, raw_a));

    velocity += a * 0.0015 - 0.0025f * std::cos(3.0f * position);
    velocity = std::min(0.07f, std::max(-0.07f, velocity));

    position += velocity;
    position = std::min(0.6f, std::max(-1.2f, position));

    if (position == -1.2f && velocity < 0)
    {
        velocity = 0.0f;
    }

    const TerminalState is_final = current_episode_length == 999 ? TerminalState::Timeout : (position >= 0.45f ? TerminalState::Terminal : TerminalState::NotTerminal );

    float reward = -raw_a * raw_a * 0.1f + (is_final == TerminalState::Terminal ? 100.0f : 0.0f);

    torch::Tensor obs = GetObs();

    return StepResult{ obs, reward, is_final };
}

torch::Tensor MountainCarContinuous::GetObs() const
{
    torch::Tensor output = torch::zeros({ 1, 2 });
    float* data = output.data<float>();

    data[0] = position;
    data[1] = velocity;

    return output;
}

