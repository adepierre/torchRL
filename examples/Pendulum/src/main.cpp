#include "torchrl/algorithms/ppo/PPO.hpp"
#include "torchrl/algorithms/ppo/PPOArgs.hpp"
#include "torchrl/envs/VectorizedEnv.hpp"

#include "Pendulum/PendulumEnv.hpp"

int main(char argc, char* argv[])
{
    try
    {
        PPOArgs args;

        // Manually set args for this example
        args.seed = 12345;

        args.normalize_env_obs = false;
        args.normalize_env_reward = false;

        // Parse user specified ones
        args.ParseArgs(argc, argv);

        torch::manual_seed(args.seed);

        /* ------------- TRAIN ------------- */
        VectorizedEnv env(args.normalize_env_obs, args.normalize_env_reward);
        env.CreateEnvs<PendulumEnv>(args.n_envs, args.seed);

        PPO ppo(env, args);

        auto start = std::chrono::steady_clock::now();
        ppo.Learn(100000);
        auto end = std::chrono::steady_clock::now();
        std::cout << "Training done in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0 << "s" << std::endl;


        /* ------------- PLAY ------------- */
        // We recreate everything so we're sure data will be loaded from the files
        VectorizedEnv env_play(args.normalize_env_obs, args.normalize_env_reward);
        env_play.CreateEnvs<PendulumEnv>(1, args.seed + 1);
        PPO ppo_play(env_play, args);

        ppo_play.Play(1000);

    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
