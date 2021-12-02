from os import stat
import numpy as np
import random

from simple_grid import BROKEN_LEG_PENALTY

NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500


DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1
COUNTER = 1

printing = True

class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE): 
        self.name = "agent1"
        self.q_table = np.zeros((num_states, num_actions)) + 10000
        print(self.q_table)
        self.discount = discount
        self.learning_rate = learning_rate
        self.possible_actions = num_actions
        self.possible_states = num_states
        self.explored_states = np.zeros((num_states, num_actions))


    def reset_episode(self):
        """
        Here you can update some of the statistics that could be helpful to maintain
        """
        pass


    def process_experience(self, state, action, next_state, reward, done): 
        """
        Update the Q-value based on the state, action, next state and reward.
        """
        self.explored_states[state, action] += 1
        old = self.q_table[state, action]
        if (self.explored_states[state, action] >= COUNTER):
            old = 0

        if not done:
            max_action_next_state = np.max(self.q_table[next_state,:])
            if (max_action_next_state > 1000) :
                max_action_next_state = 1
            self.q_table[state, action] = (1-self.learning_rate)*old + self.learning_rate*(reward+self.discount*max_action_next_state)
        else:
            self.q_table[state, action] = (1-self.learning_rate)*old + self.learning_rate*reward

    def select_action(self, state): 
        """
        Returns an action, selected based on the current state
        """
        #print("Index of greedy action is ", np.argmax(self.q_table[state,:]))
        return random.choice(np.argwhere(self.q_table[state, :] == np.max(self.q_table[state,:])).flatten())




    def report(self):
        """
        Function to print useful information, printed during the main loop
        """
        # Value iteration implementation
        
        # P(state, action, next_state)
        P = np.zeros([13, 4, 13])

        #Normal pavement
        for i in [1,2,3,5,6,7,9,10,11]:
            # Left
            P[i, 0, i-1] = 0.8
            P[i, 0, i] = 0.2
            # Down
            P[i, 1, i-1] = 0.1
            P[i, 1, i] = 0.8
            P[i, 1, i+1] = 0.1
            # Right
            P[i, 2, i] = 0.2
            P[i, 2, i+1] = 0.8
            # Up 
            P[i, 3, i-1] = 0.1
            P[i, 3, i] = 0.8
            P[i, 3, i+1] = 0.1
        
        #Start state
        P[0, 0, 0] = 1
        P[0, 1, 0] = 0.9
        P[0, 1, 1] = 0.1
        P[0, 2, 0] = 0.2
        P[0, 2, 1] = 0.8
        P[0, 3, 0] = 0.9
        P[0, 3, 1] = 0.1

        #End state
        P[12, 0, 12] = 1
        P[12, 1, 12] = 1
        P[12, 2, 12] = 1
        P[12, 3, 12] = 1

        #Pothole state (4, 8)
        for i in [4, 8]:
            # Left
            P[i, 0, i-1] = 0.8
            P[i, 0, 0] = 0.2
            # Down
            P[i, 1, i] = 0.8
            P[i, 1, 0] = 0.2
            # Right
            P[i, 2, i+1] = 0.8
            P[i, 2, 0] = 0.2
            # Up
            P[i, 3, i] = 0.8
            P[i, 3, 0] = 0.2

        
        reward = np.zeros((13, 4, 13))
        reward[11, 1, 12] = 10
        reward[11, 2, 12] = 10
        reward[11, 3, 12] = 10

        for i in [4, 8]:
            for j in range(4):
                reward[i, j, 0] = BROKEN_LEG_PENALTY
        
        ## Value iteration: Q*_n+1(s, a) : = sum_{s'\in S}p(next state|s, a) * (R(state, action, next_state)  + gamma * max(Q*_n(next_state, next action)))
        optimal_q = np.zeros([13, 4]) 
        for counter in range(1000):
            for state in range(13):
                for action in range(4):
                    intermediate = 0
                    for next_state in range(13):
                        intermediate += P[state, action, next_state] * (reward[state, action, next_state] + self.discount * np.max(optimal_q[next_state, :]))
                    optimal_q[state, action] = intermediate
        print("Optimal" ,optimal_q)
                
             

        if printing:
            ncols = 13
            nrows = 1

            print("---")
            print("Greedy policy goes through these states:")
            state = 0
            counter = 0
            print("state is 0")
            while state != self.possible_states-1 and counter < 20:
                action = np.argmax(self.q_table[state,:])
                if action == 0:
                    print("GO LEFT")
                    if state % ncols != 0:
                        state = state - 1
                elif action == 1:
                    print("GO DOWN")
                    if state <= ncols*(nrows-1):
                        state = state + ncols
                elif action == 2:
                    print("GO RIGHT")
                    if state % ncols != (ncols-1):
                        state = state + 1
                elif action == 3:
                    print("GO UP")
                    if state > ncols:
                        state = state - ncols
                
                print("state is ", state) 
                counter += 1
            print("---")

            print(self.q_table)
    








        
