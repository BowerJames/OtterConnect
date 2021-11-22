from .Agent import Agent
import numpy as np

class RandomAgent(Agent):

    def choose_action(self, state):
        possible_actions = state.possible_actions()
        return np.random.choice(possible_actions)

class EPSGreedyAgent(Agent):
    def __init__(self, num_actions, eps=0.1, discount_factor = 0.99, learning_rate=0.1):
        self.q_table = {}
        self.num_actions = num_actions
        self.eps=eps
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

    def choose_top_action(self,state):
        try:
            state_values = self.q_table[state]
        except:
            self.q_table[state] = np.zeros((self.num_actions,))
            state_values = self.q_table[state]
        finally:
            top_actions = np.reshape(np.argwhere(state_values == np.max(state_values)),(-1,))
            return np.random.choice(top_actions)

    def choose_action(self, state):

        if np.random.random() < self.eps:
            return self.choose_random_action(state)
        else:
            return self.choose_top_action(state)

    def choose_random_action(self, state):
        possible_actions = state.possible_actions()
        return np.random.choice(possible_actions)

    def update_table_with_game(self, states, actions, rewards):

        next_state = None

        for i in range(len(states)):
            state = states[-1*(i+1)]
            action = actions[-1*(i+1)]
            reward = rewards[-1*(i+1)]
            self.update_table(state, action, reward, next_state)
            next_state=state

    def update_table(self, state, action, reward, next_state):
        try:
            action_values = self.q_table[state]
        except:
            self.q_table[state] = np.zeros((self.num_actions,))
            action_values = self.q_table[state]
        finally:
            if next_state is None:
                action_values[action] = action_values[action] + self.learning_rate*(reward - action_values[action])
            else:
                action_values[action] = action_values[action] + self.learning_rate*(reward - self.discount_factor*np.max(self.q_table[next_state]) - action_values[action])

class EPSAverageAgent(Agent):
    
    def __init__(self, num_actions, eps=0.1, discount_factor = 0.99, learning_rate=0.1):
        self.q_table = {}
        self.num_actions = num_actions
        self.eps=eps
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

    def choose_top_action(self,state):
        try:
            state_values = self.q_table[state]
        except:
            self.q_table[state] = np.zeros((self.num_actions,))
            state_values = self.q_table[state]
        finally:
            top_actions = np.reshape(np.argwhere(state_values == np.max(state_values)),(-1,))
            return np.random.choice(top_actions)

    def choose_action(self, state):

        if np.random.random() < self.eps:
            return self.choose_random_action(state)
        else:
            return self.choose_top_action(state)

    def choose_random_action(self, state):
        possible_actions = state.possible_actions()
        return np.random.choice(possible_actions)

    def update_table_with_game(self, states, actions, rewards):

        next_state = None

        for i in range(len(states)):
            state = states[-1*(i+1)]
            action = actions[-1*(i+1)]
            reward = rewards[-1*(i+1)]
            self.update_table(state, action, reward, next_state)
            next_state=state

    def update_table(self, state, action, reward, next_state):
        try:
            action_values = self.q_table[state]
        except:
            self.q_table[state] = np.zeros((self.num_actions,))
            action_values = self.q_table[state]
        finally:
            if next_state is None:
                action_values[action] = action_values[action] + self.learning_rate*(reward - action_values[action])
            else:
                action_values[action] = action_values[action] + self.learning_rate*(reward - self.discount_factor*np.mean(self.q_table[next_state]) - action_values[action])
