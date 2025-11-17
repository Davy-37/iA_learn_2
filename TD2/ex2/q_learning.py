import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """
    Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[sprime, :]) - Q[s, a])
    return Q    


def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as unput the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    Q_ = Q[s, :]
    if np.random.rand() < epsilone:
        return np.random.randint(len(Q_))
    else:
        return np.argmax(Q_)    


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.01 # choose your own

    gamma = 0.8 # choose your own

    epsilon = 0.2 # choose your own

    n_epochs = 20 # choose your own
    max_itr_per_epoch = 100 # choose your own
    rewards = []

    for e in range(n_epochs):
        r = 0

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            # Update state and put a stoping criteria

        print("episode #", e, " : r = ", r)

        rewards.append(r)

    print("Average reward = ", np.mean(rewards))

    # plot the rewards in function of epochs

    print("Training finished.\n")

    
    """
    
    Evaluate the q-learning algorihtm (ici on ne fais plus la mise à jour de la Q-table mais on utilise la Q-table apprise pour prendre les actions et compteter le nombre ditérations pour terminer l'épisode)
    
    """
    n_test_epochs = 10
    for e in range(n_test_epochs):
        S, _ = env.reset()
        done = False
        itr = 0
        while not done:
            A = np.argmax(Q[S, :])
            S, R, done, _, info = env.step(A)
            env.render()
            itr += 1
        print("Test episode #", e, " : number of iterations = ", itr)
        plt.plot(rewards)
    plt.xlabel("Epochs")
    plt.ylabel("Rewards")
    plt.title("Rewards over Epochs")
    plt.show()
    

    env.close()
