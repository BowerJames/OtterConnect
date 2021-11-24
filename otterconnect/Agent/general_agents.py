from .Agent import Agent
import numpy as np

class RandomAgent(Agent):
    '''
    Agent that implements a random policy.

    Methods
    -------
    choose_test_action(state):
        Chooses random possible action.

    choose_training_action(state):
        Chooses random possible action.
    '''


    def choose_train_action(self, state):
        possible_actions = state.possible_actions()
        return np.random.choice(possible_actions)

    def choose_test_action(self, state):
        possible_actions = state.possible_actions()
        return np.random.choice(possible_actions)

class EPSGreedyQTable(Agent):
    '''
    Epsilon greedy Q table agent.

    Properties
    ----------
    q_table : dict
        Dictionary that acts as the q table object.
    num_actions : int
        number of possible actions to pick from.
    eps : float
        Epsilon factor dtermining proportion of random moves.
    discount_factor : float
        Discount factor applied to the returned rewards.
    learning_rate : float
        Learning rate for the Q table updates.

    Methods
    -------
    __init__(num_actions, eps,discount_factor, learning_rate):
        Initialie agent setting num_actions, eps,discount_factor and learning_rate.

    choose_test_action(state):
        Chooses the top action according to the values stored in the Q Table.

    choose_train_action(state):
        Chooses a random action eps proportion of the time els choose the top action by the Q table.

    choose_random_action(state):
        Chooses random possible action.
    
    update_table_with_game(states, action, rewards):
        Updates the Q table based on the state actions and rewards of a single game.
    
    update_table(state,action,reward,next_state):
        Updates the Q table based on a specific state, action, reward and next state combination.
    '''
    def __init__(self, num_actions, eps=0.1, discount_factor = 0.99, learning_rate=0.1):
        self.q_table = {}
        self.num_actions = num_actions
        self.eps=eps
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

    def choose_test_action(self,state):
        try:
            state_values = self.q_table[state]
        except:
            self.q_table[state] = np.zeros((self.num_actions,))
            state_values = self.q_table[state]
        finally:
            top_actions = np.reshape(np.argwhere(state_values == np.max(state_values)),(-1,))
            return np.random.choice(top_actions)

    def choose_train_action(self, state):

        if np.random.random() < self.eps:
            return self.choose_random_action(state)
        else:
            return self.choose_test_action(state)

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

class EPSExpectedQTable(Agent):
    '''
    Epsilon greedy Q table agent.

    Properties
    ----------
    q_table : dict
        Dictionary that acts as the q table object.
    num_actions : int
        number of possible actions to pick from.
    eps : float
        Epsilon factor dtermining proportion of random moves.
    discount_factor : float
        Discount factor applied to the returned rewards.
    learning_rate : float
        Learning rate for the Q table updates.

    Methods
    -------
    __init__(num_actions, eps,discount_factor, learning_rate):
        Initialie agent setting num_actions, eps,discount_factor and learning_rate.

    choose_test_action(state):
        Chooses the top action according to the values stored in the Q Table.

    choose_train_action(state):
        Chooses a random action eps proportion of the time els choose the top action by the Q table.

    choose_random_action(state):
        Chooses random possible action.
    
    update_table_with_game(states, action, rewards):
        Updates the Q table based on the state actions and rewards of a single game.
    
    update_table(state,action,reward,next_state):
        Updates the Q table based on a specific state, action, reward and next state combination.
    '''
    
    def __init__(self, num_actions, eps=0.1, discount_factor = 0.99, learning_rate=0.1):
        self.q_table = {}
        self.num_actions = num_actions
        self.eps=eps
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

    def choose_test_action(self,state):
        try:
            state_values = self.q_table[state]['score']
        except:
            self.q_table[state] = {}
            self.q_table[state]['score']= np.zeros((self.num_actions,))
            self.q_table[state]['visits'] = np.zeros((self.num_actions,))
            state_values = self.q_table[state]['score']
            return self.choose_random_action(state)
        finally:
            top_actions = np.reshape(np.argwhere(state_values == np.max(state_values)),(-1,))
            return np.random.choice(top_actions)

    def choose_train_action(self, state):

        if np.random.random() < self.eps:
            return self.choose_random_action(state)
        else:
            return self.choose_test_action(state)

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
            action_values = self.q_table[state]['score']
            action_visits = self.q_table[state]['visits']
        except:
            self.q_table[state] = {}
            self.q_table[state]['score']= np.zeros((self.num_actions,))
            self.q_table[state]['visits'] = np.zeros((self.num_actions,))
            action_values = self.q_table[state]['score']
            action_visits = self.q_table[state]['visits']
        finally:
            if next_state is None:
                action_values[action] = action_values[action] + self.learning_rate*(reward - action_values[action])
            else:
                action_values[action] = action_values[action] + self.learning_rate*(reward - self.discount_factor*(np.sum(self.q_table[next_state]['score']*self.q_table[next_state]['visits']) / np.sum(self.q_table[next_state]['visits'])) - action_values[action])
            action_visits[action]+=1
