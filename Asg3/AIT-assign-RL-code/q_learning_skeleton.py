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


def initialize_transition_table():

    # P(state, action, next_state)
    P = np.zeros([13, 4, 13])
    # Normal pavement
    for i in [1, 2, 3, 5, 6, 7, 9, 10, 11]:
        # Left
        P[i, 0, i - 1] = 0.8
        P[i, 0, i] = 0.2
        # Down
        P[i, 1, i - 1] = 0.1
        P[i, 1, i] = 0.8
        P[i, 1, i + 1] = 0.1
        # Right
        P[i, 2, i] = 0.2
        P[i, 2, i + 1] = 0.8
        # Up
        P[i, 3, i - 1] = 0.1
        P[i, 3, i] = 0.8
        P[i, 3, i + 1] = 0.1

    # Start state
    P[0, 0, 0] = 1
    P[0, 1, 0] = 0.9
    P[0, 1, 1] = 0.1
    P[0, 2, 0] = 0.2
    P[0, 2, 1] = 0.8
    P[0, 3, 0] = 0.9
    P[0, 3, 1] = 0.1

    # End state
    P[12, 0, 12] = 1
    P[12, 1, 12] = 1
    P[12, 2, 12] = 1
    P[12, 3, 12] = 1

    # Pothole state (4, 8)
    for i in [4, 8]:
        # Left
        P[i, 0, i - 1] = 0.8
        P[i, 0, 0] = 0.2
        # Down
        P[i, 1, i] = 0.8
        P[i, 1, 0] = 0.2
        # Right
        P[i, 2, i + 1] = 0.8
        P[i, 2, 0] = 0.2
        # Up
        P[i, 3, i] = 0.8
        P[i, 3, 0] = 0.2
    return P

class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE, epsilon=EPSILON):
        # Initialize Q-table, discount value, learning-rate, number of possible actions and states and epsilon
        self.name = "agent1"
        self.q_table = np.zeros((num_states, num_actions))
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

        ## FINDING OF Q-star, (independent of exploration strategy)

        # # First we initialize all the transition probabilities
        P = initialize_transition_table()

        # Initialize the reward function
        reward = np.zeros((13, 4, 13))
        reward[11, 1, 12] = 10
        reward[11, 2, 12] = 10
        reward[11, 3, 12] = 10

        for i in [4, 8]:
            for j in range(4):
                reward[i, j, 0] = BROKEN_LEG_PENALTY

        ## Value iteration: Q*_n+1(s, a) : = sum_{s'\in S}p(next state|s, a) * (R(state, action, next_state)  + gamma * max(Q*_n(next_state, next action)))

        # Initialize optimal q-table and intermediate q-table
        optimal_q = np.zeros([13, 4])
        int_q_table = {}
        for state in range(13):
            int_q_table[state] = {}
            for action in range(4):
                int_q_table[state][action] = 1000
        # Initialize counter that keeps track of the amount of q-table entries that have not changed
        identical_value_counter = 0
        while identical_value_counter != 13*4:
            # New loop so we reset the value to zero
            identical_value_counter = 0
            for state in range(13):
                for action in range(4):
                    intermediate = 0
                    for next_state in range(13):
                        intermediate += P[state, action, next_state] * (
                                reward[state, action, next_state] + self.discount_factor * np.max(optimal_q[next_state, :]))
                    if intermediate == int_q_table[state][action]:
                        # increment counter whenever value has not changed last
                        identical_value_counter += 1

                    # Set q-table entries
                    int_q_table[state][action] = intermediate
                    optimal_q[state, action] = intermediate
        print("Optimal Q*-table", optimal_q)

    








        

