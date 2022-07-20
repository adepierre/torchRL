import gym
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

if __name__ == '__main__':
    envs = ['MountainCarContinuous-v0', 'Pendulum-v1']
    for env_id in envs:
        base_seed = 12345
        base_path = os.path.join(env_id, 'exp_')

        for i in range(10):
            path = base_path + str(i)
            seed = base_seed + i
            os.makedirs(path, exist_ok=True)
            set_random_seed(seed)
            if env_id == 'MountainCarContinuous-v0':
                env = make_vec_env(env_id, n_envs=1, seed=seed)
                env = VecNormalize(env)
                env.training = True
            else:
                env = make_vec_env(env_id, n_envs=4, seed=seed)
            
            my_logger = configure(path, ['csv'])

            if env_id == 'MountainCarContinuous-v0':
                model = PPO('MlpPolicy', env,
                    1e-4, #lr
                    8, #n_steps
                    8, # batch_size
                    10, # n_epochs
                    0.9999, # gamma
                    0.9, #gae_lambda
                    0.1, # clip_range
                    None, # clip_range_vf
                    True, # normalize_advantage
                    5e-3, #ent_coef
                    0.2, # vf_coef
                    5.0, #max_grad_norm
                    False, #use_sde
                    -1, #sde_sample_freq
                    None, #target_kl
                    device='cpu',
                    policy_kwargs={'log_std_init': -3.0, 'ortho_init': False }
                )
            else:
                model = PPO('MlpPolicy', env,
                    1e-3, #lr
                    1024, #n_steps
                    64, # batch_size
                    10, # n_epochs
                    0.9, # gamma
                    0.95, #gae_lambda
                    0.2, # clip_range
                    None, # clip_range_vf
                    True, # normalize_advantage
                    0.0, #ent_coef
                    0.5, # vf_coef
                    0.5, #max_grad_norm
                    False, #use_sde
                    -1, #sde_sample_freq
                    None, #target_kl
                    device='cpu'
                )
            model.set_logger(my_logger)
            if env_id == 'MountainCarContinuous-v0':
                model.learn(total_timesteps=100000)
            else:
                model.learn(total_timesteps=150000)


            model.save(os.path.join(path, 'policy.pt'))
            
            if env_id == 'MountainCarContinuous-v0':
                env.save(os.path.join(path, 'env.pt'))

            del model
            del env

            model = PPO.load(os.path.join(path, 'policy.pt'))
            if env_id == 'MountainCarContinuous-v0':
                env = make_vec_env(env_id, 1, seed +42)
                env = VecNormalize.load(os.path.join(path, 'env.pt'), env)
                env.training = False
                env.norm_reward = False
            else:
                env = gym.make(env_id)
                env.seed(seed + 42)
                env = Monitor(env)

            rewards, lengths = evaluate_policy(model, env, 100, True, False, return_episode_rewards=True)
            with open(os.path.join(path, 'played.csv'), 'w') as f:
                f.write('Episode length\tEpisode reward\t\n')
                for l, r in zip(lengths, rewards):
                    f.write(f'{l}\t{r}\t\n')
