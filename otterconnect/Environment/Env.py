from abc import ABC, abstractmethod

class Env(ABC):
    '''
    Class to represent a generic environment.

    Properties
    ----------
    state : list<State>
        List of past states.
    rewards : list<float>
        List of past rewards.
    actions : list<int>
        List of past actions.
    T : Bool
        Whether the environment has terminated.
    current_state_ : State
        The current state of the environment.

    Methods
    -------
    __init__():
        Initialise the environment object setting the state, action, rewards records to blank lists and the termination (T) to False.

    step(action):
        A method to apply an action to the environment.

    reset():
        Reset the environment object setting the state, action, rewards records to blank lists and the termination (T) to False.
    '''

    def __init__(self):
        self.reset()

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        self.states = []
        self.rewards = []
        self.actions=[]
        self.T=False

class State(ABC):
    '''
    Class to represent the state object in an environnment.

    Methods
    -------
    possible_actions():
        Function that can return a list of possible actions for the state.
    
    __hash__():
        A hash function so that the state can be stored in a dictionary.

    __eq__():
        Equality method so that states can be compared.
    '''

    @abstractmethod
    def possible_actions(self):
        pass
    
    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass
