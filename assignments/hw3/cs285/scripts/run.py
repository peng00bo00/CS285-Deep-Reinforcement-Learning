import os

# os.system("python run_hw3_dqn.py --env_name PongNoFrameskip-v4 --exp_name test_pong")

# ## Q1
# os.system("python run_hw3_dqn.py --env_name PongNoFrameskip-v4 -gpu --exp_name q1")

# ## Q2
# for i in range(1, 4):
#     os.system(f"python run_hw3_dqn.py --env_name LunarLander-v2 -gpu --exp_name q2_dqn_{i} --seed {i}")

# for i in range(1, 4):
#     os.system(f"python run_hw3_dqn.py --env_name LunarLander-v2 -gpu --exp_name q2_doubledqn_{i} --double_q --seed {i}")

# ## Q3
# os.system("python run_hw3_dqn.py --env_name PongNoFrameskip-v4 -gpu --double_q --exp_name q3_hparam1")
# os.system("python run_hw3_dqn.py --env_name PongNoFrameskip-v4 -gpu --double_q --batch_size 64 --exp_name q3_hparam2")
# os.system("python run_hw3_dqn.py --env_name PongNoFrameskip-v4 -gpu --double_q --batch_size 128 --exp_name q3_hparam3")

# ## Q4
# os.system("python run_hw3_actor_critic.py --env_name CartPole-v0 -gpu -n 100 -b 1000 --exp_name 1_1 -ntu 1 -ngsptu 1")

# os.system("python run_hw3_actor_critic.py --env_name CartPole-v0 -gpu -n 100 -b 1000 --exp_name 100_1 -ntu 100 -ngsptu 1")
# os.system("python run_hw3_actor_critic.py --env_name CartPole-v0 -gpu -n 100 -b 1000 --exp_name 10_10 -ntu 10 -ngsptu 10")
# os.system("python run_hw3_actor_critic.py --env_name CartPole-v0 -gpu -n 100 -b 1000 --exp_name 1_100 -ntu 1 -ngsptu 100")

## Q5
os.system("python run_hw3_actor_critic.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name 10_10 -ntu 10 -ngsptu 10")
os.system("python run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name 10_10 -ntu 10 -ngsptu 10")