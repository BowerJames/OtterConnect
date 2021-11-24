import numpy as np


from .Env import Env, State
class Connect4(Env):
    '''
    Class to represent a connect 4 environment

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
        A method that plays a piece in column 'action' recording the states actions and rewards.

    reset():
        Reset the environment object setting the state, action, rewards records to blank lists, the termination (T) to False and current_state_ to an empty board.

    take_action(action):
        Plays piece in column 'action' returnin gthe old and new states.

    evaluate(state):
        Evauates a state returning the reward of the previous action and whether the environment is terminated.

    win_from(frame, column, row):
        Returns whether there is a win from a specific square for a specific frame.

    from_state(state):
        Creates a Connect4 environment object with no history but the current_state_ set to a copy of the input state.
    '''

    def __init__(self, state=None):
        super().__init__()

    def reset(self):
        state_setup = np.zeros((7,6,3))
        state_setup[:,:,0] = 1
        self.current_state_ = Connect4State(state_setup, 1)
        super().reset()


    def step(self, action):
        self.actions.append(action)
        old_state, new_state = self.take_action(action)
        self.states.append(old_state)
        self.current_state_ = new_state
        reward , termination = Connect4.evaluate(new_state)
        self.rewards.append(reward)
        self.T=termination
        return self

    def take_action(self,action):
        new_state = self.current_state_.copy()
        new_state_rep = new_state.board
        column = new_state_rep[action,:,:]
        spots_filled = int(6 - column[:,0].sum())
        if spots_filled==6:
            new_state.board = None
            return self.current_state_ , new_state
        column[spots_filled,0] = 0
        column[spots_filled,new_state.players_turn] = 1
        new_state.players_turn = 3 - new_state.players_turn
        return self.current_state_ , new_state

    @classmethod
    def evaluate(cls, state):
        np_state = state.board
        if np_state is None:
            return -1,True
        empty_frame = np_state[:, :, 0]
        player_1_frame = np_state[:, :, 1]
        player_2_frame = np_state[:, :, 2]
        for row in range(6):
            for column in range(7):
                if Connect4.win_from(player_1_frame, column, row):
                    return 1, True
                if Connect4.win_from(player_2_frame, column, row):
                    return 1, True
        if np.sum(empty_frame)==0:
            return 0 , True
        return 0 , False

    @classmethod
    def win_from(cls,frame, column, row):
        if row < 3:
            if np.sum(frame[column, row:row+4]) == 4:
                return True
        if column < 4:
            if np.sum(frame[column:column+4, row]) == 4:
                return True
        if row < 3 and column < 4:
            if sum([frame[column, row], frame[column+1, row+1], frame[column+2, row+2], frame[column+3, row+3]]) == 4:
                return True
        if column > 2 and row < 3:
            if sum([frame[column, row], frame[column-1, row+1], frame[column-2, row + 2], frame[column-3, row+3]]) == 4:
                return True
        return False

    @classmethod
    def from_state(cls, state):
        env = cls()
        env.current_state_ = state.copy()

        






class Connect4State(State):
    '''
    Class object for a connect 4 state.

    Properties
    ----------
    board : ndarray
        Numpy array representing the location on the counters.
    players_turn : int
        Determines who is to play next.

    Methods
    -------
    __init__(state_rep, players_turn):
        Initialises a state object with given board and players_turn.

    possible_actions():
        Function that can return a list of possible actions for the state.
    
    __hash__():
        A hash function so that the state can be stored in a dictionary.

    __eq__():
        Equality method so that states can be compared.

    as_numpy():
        TBD
    
    copy():
        Creates a new instance that is a copy of this state.

    print_state():
        Prints a visual of the state.

    to_env()
        Creates a Connect 4 environment with current_state_ set to a copy of this State instance.
    '''

    def __init__(self, board, players_turn):
        self.board = board
        self.players_turn = players_turn

    def as_numpy(self):
        pass

    def copy(self):
        return Connect4State(self.board.copy(),self.players_turn)

    def print_state(self):
        if self.board is None:
            print("Invalid state position was found.")
        else:
            spot_mapping = {0:' ',1:'o',2:'x'}
            to_print = ''
            for row in range(6):
                row_string = '|'
                np_row = self.board[:,row,:]
                for column in range(7):
                    row_string = row_string + f'{spot_mapping[np.argmax(np_row[column,:])]}|'
                row_string = row_string + '\n'
                to_print = row_string + to_print
            print(to_print)

    def to_env(self):
        return Connect4.from_state(state)

    def possible_actions(self):
        board = self.board
        empty_frame = board[:,:,0]
        column_counts = np.sum(empty_frame, axis=1)
        return np.squeeze(np.argwhere(column_counts > 0))

    def __hash__(self):
        return hash(str(self.board)  + str(self.players_turn))

    def __eq__(self, other):
        return (self.players_turn==other.players_turn) and np.array_equal(self.board, other.board)

