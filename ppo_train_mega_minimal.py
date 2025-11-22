from pathlib import Path
from stable_baselines3 import PPO
from ai import DoppelkopfEnv
from agents import ExpectiMaxAgent

TOTAL_TIMESTEPS = 10_000
SAVE_PATH = Path("models") / "dk_ppo_mm_single.zip"

def make_env():
    env = DoppelkopfEnv("ALICE", expectimax_prob=0.7)
    env.agent = ExpectiMaxAgent("ALICE")
    return env

def main():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    env = make_env()
    model = PPO(
        "MlpPolicy",
        env,
        gamma=1.0,          # full episode, no discount
        n_steps=128,        # rollout length
        batch_size=256,     # minibatch size
        learning_rate=3e-4,
        ent_coef=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(SAVE_PATH)
    env.close()


if __name__ == "__main__":
    main()
