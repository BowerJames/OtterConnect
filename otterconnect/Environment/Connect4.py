import numpy as np


from .Env import Env
class Connect4(Env):

    def __init__(self, state=None):
        self.current_state_ = state
        self.T=False
        self.states = []
        self.actions = []
        self.rewards = []

    def reset(self):
        state_setup = np.zeros((7,6,3))
        state_setup[:,:,0] = 1
        self.current_state_ = Connect4State(state_setup, 1)
        self.T=False
        self.states = []
        self.actions = []
        self.rewards = []


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

        






class Connect4State():

    def __init__(self, state_rep, players_turn):
        self.board = state_rep
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
        return Connect4Environment(state)

    def possible_actions(self):
        board = self.board
        empty_frame = board[:,:,0]
        column_counts = np.sum(empty_frame, axis=1)
        return np.squeeze(np.argwhere(column_counts > 0))

    def __hash__(self):
        return hash(str(self.board)  + str(self.players_turn))

