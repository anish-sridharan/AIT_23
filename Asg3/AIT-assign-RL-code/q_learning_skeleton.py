import numpy as np
import random

NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 5000


DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1

class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE): 
        self.name = "agent1"
        self.map_name = 'walkInThePark' if num_states == 48 else 'theAlley' #Quick and dirty hack to toggle bw the two envs
        #self.q_table = np.zeros((num_states, num_actions))
        self.q_table = 1000*np.ones((num_states, num_actions)) # Optimistic initialisation to encourage exploration
        self.discount = discount
        self.learning_rate = learning_rate
        self.possible_actions = num_actions
        self.possible_states = num_states
        self.explored_states = np.zeros(num_states)
        self.goal_counter = 0


    def reset_episode(self):
        """
        Here you can update some of the statistics that could be helpful to maintain
        """
        pass


    def process_experience(self, state, action, next_state, reward, done): 
        """
        Update the Q-value based on the state, action, next state and reward.
        """
        self.explored_states[state] += 1
        old = self.q_table[state, action]
        if not done:
            max_action_next_state = np.max(self.q_table[next_state,:])
            self.q_table[state, action] = (1-self.learning_rate)*old + self.learning_rate*(reward+self.discount*max_action_next_state)
        else:
            self.q_table[state, action] = (1-self.learning_rate)*old + self.learning_rate*reward
        if(next_state == self.possible_states-1):
            self.goal_counter = self.goal_counter + 1 


    def select_action(self, state): 
        """
        Returns an action, selected based on the current state
        """
        if random.uniform(0,1)<=EPSILON or np.all((self.q_table[state, :] == 0)):
            return random.randint(0, self.possible_actions-1)
        else:
            #print("Index of greedy action is ", np.argmax(self.q_table[state,:]))
            return np.argmax(self.q_table[state,:])



    def report(self):
        """
        Function to print useful information, printed during the main loop
        """
        # Warning: you have to manually set the rows and columns to the correct number here
        # currently not working for theAlley
        
        if(self.map_name == 'walkInThePark'):
            ncols = 8
            nrows = 6
            print("---")
            print("Greedy policy goes through these states:")
            state = 0
            print("state is 0")
            while state != self.possible_states-1:
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
            print("---")


        if(self.map_name == 'theAlley'):
            ncols = 13
            nrows = 1
            print("---")
            print("Greedy policy goes through these states:")
            state = 0
            print("state is 0")
            while state != self.possible_states-1:
                action = np.argmax(self.q_table[state,:])
                if action == 0:
                    print("GO LEFT")
                    if state % (ncols-1) != 0:
                        state = state - 1
                elif action == 2: 
                    print("GO RIGHT")
                    if state % ncols != (ncols-1):
                        state = state + 1
                print("state is ", state)
            print("---")









        
