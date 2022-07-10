#include <torch/torch.h>
#include "torchrl/algorithms/ppo/PPO.hpp"
#include "torchrl/algorithms/ppo/PPOArgs.hpp"
#include "torchrl/envs/VectorizedEnv.hpp"
#include "torchrl/envs/NormalizedVectorizedEnv.hpp"
#include "torchrl/envs/impl/PendulumEnv.hpp"
#include "torchrl/envs/impl/MountainCarContinuous.hpp"

int main(char argc, char* argv[])
{
    try
    {
        PPOArgs args;
        args.ParseArgs(argc, argv);
        args.seed = 1234;

        //Continous moutain car
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

        torch::manual_seed(args.seed);
        NormalizedVectorizedEnv env(true);
        //VectorizedEnv env;
        env.CreateEnvs<MountainCarContinuous>(args.n_envs, args.seed);

        PPO ppo(env, args);

        auto start = std::chrono::steady_clock::now();
        ppo.Learn(20000);
        auto end = std::chrono::steady_clock::now();
        std::cout << "Training done in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0 << "s" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
