#include "torchrl/algorithms/ppo/PPO.hpp"
#include "torchrl/algorithms/ppo/PPOArgs.hpp"
#include "torchrl/envs/VectorizedEnv.hpp"
#include "MountainCar/MountainCarContinuousEnv.hpp"

int main(char argc, char* argv[])
{
    try
    {
        PPOArgs args;

        // Manually set args for this example
        args.seed = 12345;
        args.n_envs = 1;

        args.batch_size = 256;
        args.n_steps = 8;
        args.gamma = 0.9999f;
        args.lr = 1e-4f;
        args.entropy_loss_weight = 5e-3f;
        args.clip_value = 0.1f;
        args.n_epochs = 10;
        args.lambda_gae = 0.9f;
        args.max_grad_norm = 5.0f;
        args.val_loss_weight = 0.2f;
        args.init_sampling_log_std = -3.0f;
        args.ortho_init = false;

        // Parse user specified ones
        args.ParseArgs(argc, argv);

        torch::manual_seed(args.seed);

        /* ------------- TRAIN ------------- */
        VectorizedEnv env(args.normalize_env_obs, args.normalize_env_reward);
        env.CreateEnvs<MountainCarContinuousEnv>(args.n_envs, args.seed);

        PPO ppo(env, args);

        auto start = std::chrono::steady_clock::now();
        ppo.Learn(50000);
        auto end = std::chrono::steady_clock::now();
        std::cout << "Training done in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0 << "s" << std::endl;
    

        /* ------------- PLAY ------------- */
        // We recreate everything so we're sure data will be loaded from the files
        VectorizedEnv env_play(args.normalize_env_obs, args.normalize_env_reward);
        env_play.CreateEnvs<MountainCarContinuousEnv>(1, args.seed + 42);
        PPO ppo_play(env_play, args);

        ppo_play.Play();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
