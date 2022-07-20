#include "torchrl/algorithms/ppo/PPO.hpp"
#include "torchrl/algorithms/ppo/PPOArgs.hpp"
#include "torchrl/envs/VectorizedEnv.hpp"

#include "Pendulum/PendulumEnv.hpp"

#if 1
int main(char argc, char* argv[])
{
    try
    {
        PPOArgs args;

        // Manually set args for this example
        args.seed = 12345;

        // Parse user specified ones
        args.ParseArgs(argc, argv);

        torch::manual_seed(args.seed);

        //########################################################
        //######################### TRAIN ########################
        //########################################################
        VectorizedEnv env(args.normalize_env_obs, args.normalize_env_reward);
        env.CreateEnvs<PendulumEnv>(args.n_envs, args.seed);

        PPO ppo(env, args);

        auto start = std::chrono::steady_clock::now();
        ppo.Learn(150000);
        auto end = std::chrono::steady_clock::now();
        std::cout << "Training done in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0 << "s" << std::endl;


        //#######################################################
        //######################### PLAY ########################
        //#######################################################
        
        // We recreate everything so we're sure data will be loaded from the files
        VectorizedEnv env_play(args.normalize_env_obs, args.normalize_env_reward);
        env_play.CreateEnvs<PendulumEnv>(1, args.seed + 42);
        PPO ppo_play(env_play, args);

        ppo_play.Play();

    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
// Code to generate data for plotting
#else
int main(char argc, char* argv[])
{
    try
    {
        PPOArgs args;

        // Manually set args for this example
        args.seed = 12345;

        // Parse user specified ones
        args.ParseArgs(argc, argv);

        const std::string base_path = args.exp_path;
        const unsigned int base_seed = args.seed;

        for (size_t i = 0; i < 10; ++i)
        {
            args.exp_path = base_path + "_" + std::to_string(i);
            args.seed = base_seed + i;
            std::cout << "Starting training " << i << " with seed " << args.seed << std::endl;


            torch::manual_seed(args.seed);

            //########################################################
            //######################### TRAIN ########################
            //########################################################
            VectorizedEnv env(args.normalize_env_obs, args.normalize_env_reward);
            env.CreateEnvs<PendulumEnv>(args.n_envs, args.seed);

            PPO ppo(env, args);

            ppo.Learn(150000, false, false);

            //#######################################################
            //######################### PLAY ########################
            //#######################################################
            std::cout << "Starting testing " << i << std::endl;

            // We recreate everything so we're sure data will be loaded from the files
            VectorizedEnv env_play(args.normalize_env_obs, args.normalize_env_reward);
            env_play.CreateEnvs<PendulumEnv>(1, args.seed + 42);
            PPO ppo_play(env_play, args);

            std::vector<std::pair<uint64_t, float> > played_episodes = ppo_play.Play(100, false);
            std::ofstream played(args.exp_path + "/played.csv", std::ios::out);
            played << "Episode length\t" << "Episode reward\t" << std::endl;
            for (const auto& p : played_episodes)
            {
                played << p.first << "\t" << p.second << "\t" << std::endl;
            }
            played.close();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
#endif
