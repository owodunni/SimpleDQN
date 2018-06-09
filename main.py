import gym
from DQNetwork import RandomAgent

env = gym.make('Pendulum-v0')
env.reset()
is_done = False

myAgent = RandomAgent(env.action_space)

episode_count = 1000
reward = 0
done = False

for i in range(episode_count):
    env.seed(55)
    ob = env.reset()
    while True:
        env.render()
        action = myAgent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        if done:
            break
        # Note there's no env.render() here. But the environment still can open window and
        # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
        # Video is not recorded every episode, see capped_cubic_video_schedule for details.

# Close the env and write monitor result info to disk
env.close()
