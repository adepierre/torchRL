#include "MountainCar/MountainCarContinuousEnv.hpp"

MountainCarContinuousEnv::MountainCarContinuousEnv(const unsigned int seed) : AbstractEnv(seed)
{
    position = 0.0f;
    velocity = 0.0f;
    last_action = 0.0f;
}

MountainCarContinuousEnv::~MountainCarContinuousEnv()
{

}

int64_t MountainCarContinuousEnv::GetObservationSize() const
{
    return 2;
}

int64_t MountainCarContinuousEnv::GetActionSize() const
{
    return 1;
}

void MountainCarContinuousEnv::ResetImpl()
{
    position = std::uniform_real_distribution<float>(-0.6f, 0.4f)(random_engine);
    velocity = 0.0f;
    last_action = 0.0f;
}

void MountainCarContinuousEnv::RenderImpl()
{
    const int width = 62;
    const int height = 27;
    const float max_y = 1.25f;

    const float pixel_width = (max_position - min_position) / (width - 2);
    const float pixel_height = 2.5f / (height - 2);

    const int goal_col = static_cast<int>(std::round((goal_position - min_position) / pixel_width)) + 1;
    const int goal_row = static_cast<int>(std::round((max_y - std::sin(3.0f * goal_position)) / pixel_height)) - 1;

    const int car_col = static_cast<int>(std::round((position - min_position) / pixel_width)) + 1;
    const int car_row = static_cast<int>(std::round((max_y - std::sin(3.0f * position)) / pixel_height)) - 1;

    std::vector<int> sin_first_row(width - 2);
    for (int i = 0; i < width - 1; ++i)
    {
        const float val = std::sin(3.0f * (min_position + i * pixel_width));
        sin_first_row[i] = static_cast<int>(std::round((max_y - val) / pixel_height));
    }

    std::stringstream output;
    output << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";
    // Top line
    for (int i = 0; i < width; ++i)
    {
        output << static_cast<char>(219);
    }
    output << '\n';

    // Main sin thing
    for (int y = 1; y < height - 4; ++y)
    {
        output << static_cast<char>(219);
        for (int x = 1; x < width - 1; ++x)
        {
            if ((y == goal_row || y == goal_row - 1) && x == goal_col)
            {
                output << 'x';
            }
            else if (y == car_row && x == car_col)
            {
                output << static_cast<char>(184);
            }
            else if (y >= sin_first_row[x - 1])
            {
                output << '#';
            }
            else
            {
                output << ' ';
            }
        }
        output << static_cast<char>(219) << "\n";
    }

    // Acceleration line
    const int m = static_cast<int>(width / 2.0f - std::min(last_action, 0.0f) / min_action * (width - 2) / 2.0f);
    const int M = static_cast<int>(width / 2.0f + std::max(last_action, 0.0f) / max_action * (width - 2) / 2.0f);
    output << static_cast<char>(219);
    for (int i = 1; i < width - 1; ++i)
    {
        output << ((i >= m && i <= M) ? (last_action > 0 ? static_cast<char>(175) : static_cast<char>(174)) : ' ');
    }
    output << static_cast<char>(219) << "\n";

    // Closing line
    for (int i = 0; i < width; ++i)
    {
        output << static_cast<char>(219);
    }
    std::cout << output.str() << std::endl;
}

StepResult MountainCarContinuousEnv::StepImpl(const torch::Tensor& action)
{
    const float raw_a = action.item<float>();
    const float a = std::min(max_action, std::max(min_action, raw_a));

    velocity += a * power - 0.0025f * std::cos(3.0f * position);
    velocity = std::min(max_speed, std::max(-max_speed, velocity));

    position += velocity;
    position = std::min(max_position, std::max(min_position, position));
    if (position == min_position && velocity < 0)
    {
        velocity = 0.0f;
    }
    last_action = a;

    const TerminalState is_final = current_episode_length == 999 ? TerminalState::Timeout : (position >= goal_position ? TerminalState::Terminal : TerminalState::NotTerminal );
    float reward = -raw_a * raw_a * 0.1f + (is_final == TerminalState::Terminal ? 100.0f : 0.0f);

    torch::Tensor obs = GetObs();

    return StepResult{ obs, reward, is_final };
}

torch::Tensor MountainCarContinuousEnv::GetObs() const
{
    torch::Tensor output = torch::zeros({ 2 });
    float* data = output.data<float>();

    data[0] = position;
    data[1] = velocity;

    return output;
}

