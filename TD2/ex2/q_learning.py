import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """

    td_target = r + gamma * np.max(Q[sprime])
    td_error = td_target - Q[s, a]
    Q[s, a] += alpha * td_error

    return Q

def epsilon_greedy(Q, s, epsilone, action_space):
    """
    This function implements the epsilon greedy algorithm.
    Takes as unput the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
  
    if np.random.rand() < epsilone:
        return action_space.sample()
    
    
    q_row = Q[s]
    max_val = np.max(q_row)
    best_actions = np.flatnonzero(q_row == max_val)

    return np.random.choice(best_actions)

if __name__ == "__main__":

    env = gym.make("Taxi-v3", render_mode="human")  # "human" pour l'animation et None pur pas d'animation 

    # seed pour la reproductibilité
    seed = 42
    np.random.seed(seed)
    env.reset(seed=seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    
    alpha = 0.1      # choose your own
    gamma = 0.99     # choose your own
    epsilon = 0.2    # choose your own
    n_epochs = 500   # choose your own
    max_itr_per_epoch = 200  # choose your own

    rewards = []

    for e in range(n_epochs):
        S, _ = env.reset()
        r = 0

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon, action_space=env.action_space)

            Sprime, R, terminated, truncated, info = env.step(A)
            done = terminated or truncated

            r += R

            Q = update_q_table(Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma)

               # Update state and put a stopping criteria

            S = Sprime
            if done:
                break

        rewards.append(r)
        if (e + 1) % 50 == 0:
            print(f"Episode {e+1}/{n_epochs} | reward: {r:.1f} | avg last 50: {np.mean(rewards[-50:]):.2f}")


    print("episode #", e, " : r = ", r)
    
    # plot the rewards in function of epochs

    print("Training finished.")
    print("Average reward over all episodes:", np.mean(rewards))

   
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-learning on Taxi-v3: Training Rewards")
    plt.grid(True)
    plt.show()

    """
    
    Evaluate the q-learning algorihtm
    
    """   

    n_eval_episodes = 20
    eval_rewards = []
    for _ in range(n_eval_episodes):
        s, _ = env.reset()
        ep_r = 0
        for _ in range(max_itr_per_epoch):
            # action gloutonne
            q_row = Q[s]
            max_val = np.max(q_row)
            best_actions = np.flatnonzero(q_row == max_val)
            a = np.random.choice(best_actions)

            s, r, terminated, truncated, _ = env.step(a)
            ep_r += r
            if terminated or truncated:
                break
        eval_rewards.append(ep_r)

    print(f"Greedy policy average reward over {n_eval_episodes} eval episodes: {np.mean(eval_rewards):.2f}")

    env.close()
