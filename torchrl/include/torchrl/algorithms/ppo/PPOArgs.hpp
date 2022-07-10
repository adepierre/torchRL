#pragma once

#include "torchrl/rl/Args.hpp"

struct PPOArgs : public Args
{
    // Training parameters

    /// @brief Learning mini-batch size
    uint64_t batch_size = 64;
    /// @brief Number of steps collected for each env
    uint64_t n_steps = 1024;
    /// @brief Number of times each collected sample is used for training
    uint64_t n_epochs = 5;

    /// @brief Value loss weight
    float val_loss_weight = 0.5f;
    /// @brief Entropy loss weight
    float entropy_loss_weight = 0.0f;
    /// @brief Max norm of the grad (disabled if 0)
    float max_grad_norm = 0.5f;
    /// @brief Initial log value for the gaussian distribution std
    float init_sampling_log_std = 0.0f;
    /// @brief Whether to use or not orthogonal initialization
    bool ortho_init = true;
    /// @brief Gamma value
    float gamma = 0.9f;
    /// @brief Lambda value
    float lambda_gae = 0.95f;

    /// @brief PPO Clip value
    float clip_value = 0.2f;
    /// @brief Learning rate
    float lr = 0.001f;

    std::string GenerateHelp(const char* argv0, const bool include_parent_help = true)
    {
        std::stringstream s;
        if (include_parent_help)
        {
            s << Args::GenerateHelp(argv0);
        }
        s
            << "\t--batch_size\tSize of a minibatch, default: 64\n"
            << "\t--n_steps\tNumber of steps collected by each env during one rollout, default: 1024\n"
            << "\t--n_epochs\tNumber of times each collected sample is used for training, default: 5\n"
            << "\t--val_loss_weight\tValue loss weight, default: 0.5\n"
            << "\t--entropy_loss_weight\tEntropy loss weight, default: 0.0\n"
            << "\t--max_grad_norm\tMax norm of the grad (disabled if 0), default: 0.5\n"
            << "\t--init_sampling_log_std\tInitial log value for the gaussian distribution std, default: 0.0\n"
            << "\t--ortho_init\tWhether to use or not orthogonal initialization, default: true\n"
            << "\t--gamma\tGamma value, default: 0.9\n"
            << "\t--lambda_gae\tLambda value for GAE, default: 0.95\n"
            << "\t--clip_value\tPPO Clip value, default: 0.2\n"
            << "\t--lr\tLearning rate, default: 0.001\n";

        return s.str();
    }

    void ParseArgs(char argc, char* argv[])
    {
        // First, parse parents args
        Args::ParseArgs(argc, argv);

        // Then parse self args
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "-h" || arg == "--help")
            {
                std::cout << GenerateHelp(argv[0], false) << std::endl;
            }
            else if (arg == "--batch_size")
            {
                if (i + 1 < argc)
                {
                    batch_size = std::stoull(argv[++i]);
                }
                else
                {
                    std::cerr << "--batch_size requires an argument" << std::endl;
                    return;
                }
            }
            else if (arg == "--n_steps")
            {
                if (i + 1 < argc)
                {
                    n_steps = std::stoull(argv[++i]);
                }
                else
                {
                    std::cerr << "--n_steps requires an argument" << std::endl;
                    return;
                }
            }
            else if (arg == "--n_epochs")
            {
                if (i + 1 < argc)
                {
                    n_epochs = std::stoull(argv[++i]);
                }
                else
                {
                    std::cerr << "--n_epochs requires an argument" << std::endl;
                    return;
                }
            }
            else if (arg == "--val_loss_weight")
            {
                if (i + 1 < argc)
                {
                    val_loss_weight = std::stof(argv[++i]);
                }
                else
                {
                    std::cerr << "--val_loss_weight requires an argument" << std::endl;
                    return;
                }
            }
            else if (arg == "--entropy_loss_weight")
            {
                if (i + 1 < argc)
                {
                    entropy_loss_weight = std::stof(argv[++i]);
                }
                else
                {
                    std::cerr << "--entropy_loss_weight requires an argument" << std::endl;
                    return;
                }
            }
            else if (arg == "--max_grad_norm")
            {
                if (i + 1 < argc)
                {
                    max_grad_norm = std::stof(argv[++i]);
                }
                else
                {
                    std::cerr << "--max_grad_norm requires an argument" << std::endl;
                    return;
                }
            }
            else if (arg == "--init_sampling_log_std")
            {
                if (i + 1 < argc)
                {
                    init_sampling_log_std = std::stof(argv[++i]);
                }
                else
                {
                    std::cerr << "--init_sampling_log_std requires an argument" << std::endl;
                    return;
                }
            }
            else if (arg == "--ortho_init")
            {
            if (i + 1 < argc)
            {
                ortho_init = std::stoi(argv[++i]) != 0;
            }
            else
            {
                std::cerr << "--ortho_init requires an argument" << std::endl;
                return;
            }
            }
            else if (arg == "--gamma")
            {
                if (i + 1 < argc)
                {
                    gamma = std::stof(argv[++i]);
                }
                else
                {
                    std::cerr << "--gamma requires an argument" << std::endl;
                    return;
                }
            }
            else if (arg == "--lambda_gae")
            {
                if (i + 1 < argc)
                {
                    lambda_gae = std::stof(argv[++i]);
                }
                else
                {
                    std::cerr << "--lambda_gae requires an argument" << std::endl;
                    return;
                }
            }
            else if (arg == "--clip_value")
            {
                if (i + 1 < argc)
                {
                    clip_value = std::stof(argv[++i]);
                }
                else
                {
                    std::cerr << "--clip_value requires an argument" << std::endl;
                    return;
                }
            }
            else if (arg == "--lr")
            {
                if (i + 1 < argc)
                {
                    lr = std::stof(argv[++i]);
                }
                else
                {
                    std::cerr << "--lr requires an argument" << std::endl;
                    return;
                }
            }
        }
    }
};
