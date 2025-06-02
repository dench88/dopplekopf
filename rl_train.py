from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from ai import DoppelkopfEnv

def main():
    # 1) Load Phase-3
    model = DQN.load("doppelkopf_dqn_phase3")

    # 2) Create an env that’s 80% EM, 20% Random
    env4 = Monitor(DoppelkopfEnv("ALICE", expectimax_prob=0.8), "logs/phase4/")

    # 3) Reset exploration so we still sample a bit:
    model.exploration_fraction = 0.3   # now it decays over 60 k steps, not 20 k
    model.exploration_initial_eps = 1.0
    model.exploration_final_eps = 0.02

    # 4) Optionally clear or enlarge the replay buffer
    model.replay_buffer.reset()
    # or if you prefer: 
    # model.replay_buffer = ReplayBuffer(buffer_size=200_000, ...)  

    # 5) Attach the new env and learn 50k more
    model.set_env(env4)
    model.learn(total_timesteps=50_000)

    # 6) Save as a new checkpoint
    model.save("doppelkopf_dqn_phase4_mixed")

    env4.close()
    print("Phase 4 (mixed) complete; saved as phase4_mixed.")
    
if __name__ == "__main__":
    main()
