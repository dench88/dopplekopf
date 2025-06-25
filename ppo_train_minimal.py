# ppo_train_minimal.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ai import DoppelkopfEnv
from agents import ExpectiMaxAgent

def make_env():
    env = DoppelkopfEnv("ALICE", expectimax_prob=0.9)
    # Attach a rule‐based agent so obs/_compute_reward see correct team flags
    env.agent = ExpectiMaxAgent("ALICE")
    return env

def main():
    # Wrap in a DummyVecEnv so PPO can step it
    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        gamma=1.0,          # no discounting across the 12‐trick episode
        verbose=1,          # prints training progress
        tensorboard_log="./ppo_tensorboard/",  # optional
    )

    model.learn(total_timesteps=50_000)
    model.save("doppelkopf_ppo_minimal")
    env.close()

if __name__ == "__main__":
    main()
