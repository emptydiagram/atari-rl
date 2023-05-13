import gymnasium as gym

RAND_SEED = 278933

def run_breakout():
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env.action_space.seed(RAND_SEED)

    obs, info = env.reset(seed=RAND_SEED)

    for _ in range(1000):
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())

        if terminated or truncated:
            print("Resetting!")
            obs, info = env.reset()

    env.close()

if __name__ == '__main__':
    # TODO kick it off
    print("hello, breakout")
    run_breakout()
