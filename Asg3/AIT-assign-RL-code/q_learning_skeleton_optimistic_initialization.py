from os import stat
import numpy as np
import random

from simple_grid import BROKEN_LEG_PENALTY

NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500


DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1

printing = False

class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE, epsilon=EPSILON):
        # Initialize Q-table, discount value, learning-rate, number of possible actions and states and epsilon
        self.name = "agent1"
        self.q_table = 15*np.ones((num_states, num_actions)) # Optimal Initialization
        self.discount_factor = discount
        self.learning_rate = learning_rate
        self.possible_actions = num_actions
        self.possible_states = num_states
        self.epsilon = epsilon


    def process_experience(self, state, action, next_state, reward, done):
        """
        Update the Q-value based on the state, action, next state and reward.
        """
        old = self.q_table[state, action]
        if not done:
            # Q^new(state, action) = (1 - learning_rate) * Q^old(state, action) +
            #                                                   learning_rate (reward + discount_factor * value of next_best_action)
            max_action_next_state = np.max(self.q_table[next_state,:]) # Calculate value next best action = max_{next_action in Pos_actions}(Q^old(next_state, next_action)
            self.q_table[state, action] = (1-self.learning_rate)*old + self.learning_rate*(reward + self.discount_factor * max_action_next_state)
        else:
            # Q^new(state, action) = (1 - learning_rate) * Q^old(state, action) + learning+rate*reward
            # Since the episode terminates in the next state
            self.q_table[state, action] = (1-self.learning_rate)*old + self.learning_rate*reward



    def select_action(self, state):
        """
        Returns an action, selected based on the current state
        """
        # Epsilon chance of choosing a random action
        if random.uniform(0,1)<=self.epsilon:
            return random.randint(0, self.possible_actions-1)
        # Otherwise take the greedy action
        else:
            # Random is used to choose between two actions with the same q-value
            return random.choice(np.argwhere(self.q_table[state, :] == np.max(self.q_table[state,:])).flatten())




    def report(self):
        """
        Function to print useful information, printed during the main loop
        """
        policy = []
        for s in range(self.possible_states):
            policy.append(np.argmax(self.q_table[s, :]))
        # The printed policy are the actions the agent would take if it would only choose the greedy actions
        print("Policy Learned", policy)
    








        

