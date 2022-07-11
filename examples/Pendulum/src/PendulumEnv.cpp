#include "Pendulum/PendulumEnv.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

/// @brief Check if two lines given by points intersect (see https://stackoverflow.com/questions/4977491/determining-if-two-line-segments-intersect/4977569#4977569)
/// @param p0x Line 1 point 1 x
/// @param p0y Line 1 point 1 y
/// @param p1x Line 1 point 2 x
/// @param p1y Line 1 point 2 y
/// @param p2x Line 2 point 1 x
/// @param p2y Line 2 point 1 y
/// @param p3x Line 2 point 2 x
/// @param p3y Line 2 point 2 y
/// @return True if the two lines intersect, false otherwise
bool LinesCross(const float p0x, const float p0y,
    const float p1x, const float p1y,
    const float p2x, const float p2y,
    const float p3x, const float p3y)
{
    const float s1x = p1x - p0x;
    const float s1y = p1y - p0y;
    const float s2x = p3x - p2x;
    const float s2y = p3y - p2y;

    const float det = (-s2x * s1y + s1x * s2y);
    const float s = (-s1y * (p0x - p2x) + s1x * (p0y - p2y)) / det;
    const float t = (s2x * (p0y - p2y) - s2y * (p0x - p2x)) / det;

    if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
    {
        return true;
    }

    return 0;
}

/// @brief Check if a line defined by two points cross a pixel
/// @param x1 line point 1 x
/// @param y1 line point 1 y
/// @param x2 line point 2 x
/// @param y2 line point 2 y
/// @param px pixel x coordinate
/// @param py pixel y coordinate
/// @return true if the line crosses one of the 4 pixel sides, false otherwise
bool LinesCrossPixel(const float x1, const float y1,
    const float x2, const float y2,
    const int px, const int py)
{
    return LinesCross(x1, y1, x2, y2, px - 0.5f, py - 0.5f, px - 0.5f, py + 0.5f) ||
           LinesCross(x1, y1, x2, y2, px + 0.5f, py - 0.5f, px + 0.5f, py + 0.5f) ||
           LinesCross(x1, y1, x2, y2, px - 0.5f, py - 0.5f, px + 0.5f, py - 0.5f) ||
           LinesCross(x1, y1, x2, y2, px - 0.5f, py + 0.5f, px + 0.5f, py + 0.5f);           
}

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
    const int width = 41;
    const int height = 21;
    const float length = std::min((width - 2) / 3.0f, (height - 2) / 3.0f);

    const float center_x = width / 2.0f - 0.5f;
    const float center_y = height / 2.0f - 0.5f;
    const float end_x = center_x - length * std::sin(theta);
    const float end_y = center_y - length * std::cos(theta);

    std::stringstream output;
    output << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";
    for (int i = 0; i < width; ++i)
    {
        output << '#';
    }
    output << '\n';
    for (int y = 1; y < height - 3; ++y)
    {
        output << '#';
        for (int x = 1; x < width - 1; ++x)
        {
            output << (LinesCrossPixel(center_x, center_y, end_x, end_y, x, y) ? '#' : ' ');
        }
        output << "#\n";
    }
    const int M = static_cast<int>(width / 2.0f - std::min(last_action, 0.0f) / 2.0f * (width - 2) / 2.0f);
    const int m = static_cast<int>(width / 2.0f - std::max(last_action, 0.0f) / 2.0f * (width - 2) / 2.0f);
    output << "#";
    for (int i = 1; i < width - 1; ++i)
    {
        output << ((i >= m && i <= M) ? "#" : " ");
    }
    output << "#\n";
    output << "#";
    for (int i = 1; i < width - 1; ++i)
    {
        output << " ";
    }
    output << "#\n";
    for (int i = 0; i < width; ++i)
    {
        output << '#';
    }
    std::cout << output.str() << std::endl;
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
