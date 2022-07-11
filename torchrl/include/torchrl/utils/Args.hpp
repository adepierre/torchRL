#pragma once

#include <iostream>
#include <random>
#include <string>
#include <sstream>

struct Args
{
    /// @brief random seed for repeatability, random if not set
    unsigned int seed = std::random_device()();
    /// @brief number of parallel environment collecting data
    uint64_t n_envs = 4;
    /// @brief path to save (resp. load) model weights after (resp. before) training (resp. inference) 
    std::string exp_path = "exp";
    /// @brief whether or not the env observations should be normalized
    bool normalize_env_obs = true;
    /// @brief whether or not the env rewards should be normalized
    bool normalize_env_reward = true;

    std::string GenerateHelp(const char* argv0, const bool include_parent_help = true)
    {
        std::stringstream s;
        s << "Usage: " << argv0 << " <options>\n"
            << "Options:\n"
            << "\t-h, --help\tShow this help message\n"
            << "\t--seed\tRandom seed for envs and libtorch, default: random\n"
            << "\t--n_envs\tNumber of identical environments in the vectorized env, default: 4\n"
            << "\t--exp_path\tPath to save (resp. load) model weights after (resp. before) training (resp. inference), default: \"exp\"\n"
            << "\t--normalize_env_obs\tWhether or not the env observations should be normalized default: 1\n"
            << "\t--normalize_env_reward\tWhether or not the env rewards should be normalized default: 1\n";

        return s.str();
    }

    void ParseArgs(char argc, char* argv[])
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "-h" || arg == "--help")
            {
                std::cout << GenerateHelp(argv[0], false) << std::endl;
            }
            else if (arg == "--seed")
            {
                if (i + 1 < argc)
                {
                    seed = std::stoul(argv[++i]);
                }
                else
                {
                    std::cerr << "--seed requires an argument" << std::endl;
                    return;
                }
            }
            else if (arg == "--n_envs")
            {
                if (i + 1 < argc)
                {
                    n_envs = std::stoull(argv[++i]);
                }
                else
                {
                    std::cerr << "--n_envs requires an argument" << std::endl;
                    return;
                }
            }
            else if (arg == "--exp_path")
            {
                if (i + 1 < argc)
                {
                    exp_path = argv[++i];
                }
                else
                {
                    std::cerr << "--exp_path requires an argument" << std::endl;
                    return;
                }
            }
            else if (arg == "--normalize_env_obs")
            {
                if (i + 1 < argc)
                {
                    normalize_env_obs = std::stoi(argv[++i]) != 0;
                }
                else
                {
                    std::cerr << "--normalize_env_obs requires an argument" << std::endl;
                    return;
                }
            }
            else if (arg == "--normalize_env_reward")
            {
                if (i + 1 < argc)
                {
                    normalize_env_reward = std::stoi(argv[++i]) != 0;
                }
                else
                {
                    std::cerr << "--normalize_env_reward requires an argument" << std::endl;
                    return;
                }
            }
        }
    }
};
