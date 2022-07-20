import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def get_data_train(env):

    li = []
    for lib in os.listdir(os.path.join('data', env)):
        for f in os.listdir(os.path.join('data', env, lib)):
            if os.path.exists(os.path.join('data', env, lib, f, 'training_logs.csv')):
                df = pd.read_csv(os.path.join('data', env, lib, f, 'training_logs.csv'), sep='\t', index_col=None, header=0)
            else:
                df = pd.read_csv(os.path.join('data', env, lib, f, 'progress.csv'), sep=',', index_col=None, header=0)
                df['Play steps'] = df['time/total_timesteps']
                df['Reward episode'] = df['rollout/ep_rew_mean']
                df['Train time'] = df['time/time_elapsed']
            df['lib'] = lib
            li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)

    return frame

def get_data_test(env, offset = 0.0):
    # offset is just to slightly separate multiple libs in the scatter plot
    li = []
    libs = os.listdir(os.path.join('data', env))
    for i, lib in enumerate(libs):
        for f in os.listdir(os.path.join('data', env, lib)):
            df = pd.read_csv(os.path.join('data', env, lib, f, 'played.csv'), sep='\t', index_col=None, header=0)
            df['exp'] = float(f.split('_')[-1]) + (-offset / 2.0 + i * offset / (len(libs) - 1))
            df['lib'] = lib
            li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)

    return frame

if __name__ == '__main__':
    # Pendulum train data
    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(6)
    fig.set_figwidth(16)

    df_pendulum = get_data_train('Pendulum')
    sns.lineplot(x='Play steps', y='Reward episode', hue='lib', data=df_pendulum, ax=axs[0])
    axs[0].legend(loc='lower right', title='')
    axs[0].set_ylabel('Episode reward')
    sns.lineplot(x='Train time', y='Reward episode', hue='lib', data=df_pendulum.groupby(['Play steps', 'lib']).mean(), ax=axs[1])
    axs[1].legend(loc='lower right', title='')
    axs[1].set_ylabel('Episode reward')

    plt.savefig('train_pendulum.png', bbox_inches='tight')

    # Test data
    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(6)
    fig.set_figwidth(16)

    df_pendulum_test = get_data_test('Pendulum', 0.1)
    df_mountaincar_test = get_data_test('MountainCar', 0.1)
    
    sns.scatterplot(x='exp', y='Episode reward', hue='lib', data=df_pendulum_test, ax=axs[0])
    axs[0].set_title('Pendulum')
    axs[0].set_xlabel('Seed')
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[0].legend(loc='center right', title='')
    
    sns.scatterplot(x='exp', y='Episode reward', hue='lib', data=df_mountaincar_test, ax=axs[1])
    axs[1].set_title('Mountain Car')
    axs[1].set_xlabel('Seed')
    axs[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[1].legend(title='')

    plt.savefig('test.png', bbox_inches='tight')


