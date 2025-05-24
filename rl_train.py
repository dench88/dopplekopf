from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from ai import DoppelkopfEnv    # ensure ai.py is in same folder or on PYTHONPATH

def main():
    env = Monitor(DoppelkopfEnv("ALICE"))  # Monitor for logging
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50_000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
    )
    model.learn(total_timesteps=200_000)
    model.save("doppelkopf_dqn")
    print("Training complete, model saved as doppelkopf_dqn.zip")

if __name__ == "__main__":
    main()
