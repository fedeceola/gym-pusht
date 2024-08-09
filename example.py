import gymnasium as gym
import gym_pusht

env = gym.make("gym_pusht/PushT-v0", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step([(observation[0] + action[0]) * 512,
                                                                 (observation[1] + action[1]) * 512])
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()
