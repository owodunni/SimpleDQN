import gym

env = gym.make('Pendulum-v0')
env.reset()
is_done = False
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # take a random action
