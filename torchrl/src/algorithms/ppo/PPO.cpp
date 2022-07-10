#include <filesystem>

#include "torchrl/algorithms/ppo/PPO.hpp"
#include "torchrl/algorithms/ppo/PPOArgs.hpp"
#include "torchrl/rl/RolloutBuffer.hpp"
#include "torchrl/envs/VectorizedEnv.hpp"

PPO::PPO(VectorizedEnv& env_, const PPOArgs& args_) : env(env_), args(args_)
{
    policy = Policy(env.GetObservationSize(), env.GetActionSize(), args.ortho_init, args.init_sampling_log_std);
}

PPO::~PPO()
{

}

void PPO::Learn(const uint64_t total_timesteps)
{
    torch::optim::Adam optimizer(policy->parameters(), torch::optim::AdamOptions(args.lr));
    RolloutBuffer rollout_buffer(env.GetNumEnvs(), args.n_steps);

    env.SetTraining(true);
    env.Reset();

    uint64_t timestep = 0;
    uint64_t iteration = 0;

    float average_reward_per_episode = 0.0f;
    float average_reward_per_step = 0.0f;
    float average_episode_length = 0.0f;
    bool has_average = false;

    while (timestep < total_timesteps)
    {
        auto [total_reward, total_steps, total_episodes] = CollectRollouts(rollout_buffer);

        if (total_episodes > 0)
        {
            has_average = true;
            average_reward_per_episode = total_reward / total_episodes;
            average_reward_per_step = total_reward / total_steps;
            average_episode_length = static_cast<float>(total_steps) / total_episodes;
        }

        auto dataset = rollout_buffer.map(RolloutSampleBatchTransform());
        timestep += dataset.size().value();

        auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), torch::data::DataLoaderOptions().batch_size(args.batch_size));

        float policy_loss_val = 0.0f;
        float value_loss_val = 0.0f;
        float entropy_loss_val = 0.0f;
        int num_batches = 0;

        policy->train(true);
        for (uint64_t i = 0; i < args.n_epochs; ++i)
        {
            for (auto& rollout_data : *dataloader)
            {
                auto [values, log_probs, entropy] = policy->EvaluateActions(rollout_data.observation, rollout_data.action);

                // Normalize advantages
                torch::Tensor advantages = rollout_data.advantage;
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);

                // Compute pi ratio (should be == 1 for the first iteration)
                torch::Tensor ratio = torch::exp(log_probs - rollout_data.log_prob);

                // Clipped surrogate loss
                torch::Tensor surrogate_loss_1 = advantages * ratio;
                torch::Tensor surrogate_loss_2 = advantages * torch::clamp(ratio, 1.0f - args.clip_value, 1.0f + args.clip_value);
                torch::Tensor policy_loss = -torch::min(surrogate_loss_1, surrogate_loss_2).mean();
                policy_loss_val += policy_loss.item<float>();

                // Value loss with TD(lambda)
                torch::Tensor value_loss = torch::mse_loss(rollout_data.returns, values);
                value_loss_val += value_loss.item<float>();

                // Entropy loss
                torch::Tensor entropy_loss = -torch::mean(entropy);
                entropy_loss_val += entropy_loss.item<float>();

                torch::Tensor loss = policy_loss + args.entropy_loss_weight * entropy_loss + args.val_loss_weight * value_loss;

                optimizer.zero_grad();
                loss.backward();
                if (args.max_grad_norm > 0.0f)
                {
                    torch::nn::utils::clip_grad_norm_(policy->parameters(), args.max_grad_norm);
                }
                optimizer.step();
                num_batches += 1;
            }
        }

        std::cout 
            << "Timestep: " << timestep << "\n";
        if (has_average)
        {
            std::cout
                << "Average reward per episode: " << average_reward_per_episode << "\n"
                << "Average reward per step: " << average_reward_per_step << "\n"
                << "Average episode length: " << average_episode_length << "\n";
        }
        std::cout
            << "Policy Loss: " << policy_loss_val / num_batches << "\n"
            << "Value Loss: " << value_loss_val / num_batches << "\n"
            << "Entropy Loss: " << entropy_loss_val / num_batches << "\n"
            << std::endl;
    }

    std::filesystem::path exp_path = args.exp_path;
    if (!std::filesystem::exists(args.exp_path))
    {
        std::filesystem::create_directories(args.exp_path);
    }
    torch::save(policy, (exp_path / "policy.pt").string());
    env.Save(exp_path.string());
}

void PPO::Play(const uint64_t total_timesteps)
{
    torch::NoGradGuard no_grad;

    std::filesystem::path exp_path = args.exp_path;
    if (!std::filesystem::exists(exp_path))
    {
        std::cerr << "Error, can't find trained files in " << exp_path << std::endl;
        return;
    }

    torch::load(policy, (exp_path / "policy.pt").string());
    env.Load(exp_path.string());

    policy->train(false);
    env.SetTraining(false);
    env.Reset();

    uint64_t t = 0;
    VectorizedStepResult step_result;
    torch::Tensor obs = env.GetObs();

    while (t < total_timesteps)
    {
        env.Render();
        // Use policy to deterministically predict an action
        auto [action, value, log_prob] = policy(obs, true);

        // Perform action in the env
        step_result = env.Step(action);

        // Set the observation for the next step
        obs = step_result.obs;
        for (uint64_t i = 0; i < step_result.terminal_states.size(); ++i)
        {
            if (step_result.terminal_states[i] != TerminalState::NotTerminal)
            {
                obs[i] = step_result.new_episode_obs[i];
                std::cout 
                    << std::setw(13) << std::fixed << "Episode done!"
                    << std::setw(10) << std::fixed << "Reward"
                    << std::setw(10) << std::fixed << "Length"
                    << "\n"
                    << std::setw(13) << std::fixed << ""
                    << std::setw(10) << std::fixed << std::setprecision(2) << step_result.episodes_tot_reward[i]
                    << std::setw(10) << std::fixed << std::setprecision(4) << step_result.episodes_tot_length[i]
                    << std::endl;
            }
        }
        t += 1;
    }
}

std::tuple<float, uint64_t, uint64_t> PPO::CollectRollouts(RolloutBuffer& buffer)
{
    policy->train(false);
    buffer.Reset();

    uint64_t t = 0;
    float total_reward = 0.0f;
    uint64_t total_steps = 0;
    uint64_t total_end_episodes = 0;
    VectorizedStepResult step_result;

    // Get current observation
    torch::Tensor obs = env.GetObs();

    while (t < args.n_steps)
    {
        // Use policy to predict an action
        torch::Tensor action, value, log_prob;
        {
            torch::NoGradGuard no_grad;
            std::tie(action, value, log_prob) = policy(obs);
        }

        // Perform action in the env
        step_result = env.Step(action);

        bool has_env_timeout = false;
        for (uint64_t i = 0; i < step_result.terminal_states.size(); ++i)
        {
            has_env_timeout = has_env_timeout || step_result.terminal_states[i] == TerminalState::Timeout;
        }
        // In case at least one of the envs timeout, approx potential future reward
        // using policy estimation and add it to the reward for these envs
        if (has_env_timeout)
        {
            torch::NoGradGuard no_grad;
            torch::Tensor terminal_value = policy->PredictValues(step_result.obs);
            for (uint64_t i = 0; i < step_result.terminal_states.size(); ++i)
            {
                if (step_result.terminal_states[i] == TerminalState::Timeout)
                {
                    step_result.rewards[i] += args.gamma * terminal_value[i].item<float>();
                }
            }
        }

        buffer.Add(obs, action, value, log_prob, step_result.rewards, step_result.terminal_states);

        // Set the observation for the next step
        obs = step_result.obs;
        for (uint64_t i = 0; i < step_result.terminal_states.size(); ++i)
        {
            if (step_result.terminal_states[i] != TerminalState::NotTerminal)
            {
                obs[i] = step_result.new_episode_obs[i];
                total_reward += step_result.episodes_tot_reward[i];
                total_steps += step_result.episodes_tot_length[i];
                total_end_episodes += 1;
            }
        }
        t += 1;
    }

    // As the last episode might not be complete,
    // approx potential future rewards using
    // policy estimation
    torch::Tensor future_value;
    {
        torch::NoGradGuard no_grad;
        future_value = policy->PredictValues(obs);
        for (uint64_t i = 0; i < future_value.size(0); ++i)
        {
            if (step_result.terminal_states[i] != TerminalState::NotTerminal)
            {
                future_value[i] = 0.0f;
            }
        }
    }

    buffer.ComputeReturnsAndAdvantage(future_value, args.gamma, args.lambda_gae);
    return { total_reward, total_steps, total_end_episodes };
}
