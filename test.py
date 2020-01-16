from model import ActorCritic
import torch
import gym
from PIL import Image

def test(n_episodes=5, name='LunarLander_TWO.pth'):
    env = gym.make('LunarLander-v2')
    policy = ActorCritic()
    
    policy.load_state_dict(torch.load('./model/{}'.format(name)))
    
    render = True
    save_gif = False

    for i_episode in range(1, n_episodes+1):
        # reset the environment
        state = env.reset()
        running_reward = 0
        # for each of the steps in the episode
        for t in range(10000):
            # predict the action based on the state
            action = policy(state)
            # perform the action
            state, reward, done, _ = env.step(action)
            # accumulate the rewards in the episode
            running_reward += reward
            if render:
                 env.render()
                 if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
            
if __name__ == '__main__':
    test()
