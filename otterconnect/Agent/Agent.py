from abc import ABC, abstractmethod

class Agent(ABC):
    '''
    Agent metaclass.

    Methods
    -------
    choose_test_action(state):
        Chooses an action based on provided state when in testing runs.

    choose_training_action(state):
        Chooses an action based on provided state when in training runs.
    '''

    @abstractmethod
    def choose_test_action(self, state):
        pass

    @abstractmethod
    def choose_train_action(self, state):
        pass