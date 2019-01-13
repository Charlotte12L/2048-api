import numpy as np
import json

class Agent:
    '''Agent Base.'''
    #state_all=[[]]
    def __init__(self, game, display=None):
        self.game = game
        self.display = display
        #self.state_all=[[]]

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        state_all00=[]
        direction_all00=[]
        state_all01=[]
        direction_all01=[]                
        state_all0=[]
        direction_all0=[]
        state_all1=[]
        direction_all1=[]
        state_all2=[]
        direction_all2=[]
        state_all3=[]
        direction_all3=[]
        state_all4=[]
        direction_all4=[]
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            if self.game.score <32:
                direction_all00=np.append(direction_all00,direction)
                state_all00 = np.append(state_all00, self.game.board)
            elif 32<=self.game.score<64:
                direction_all01=np.append(direction_all01,direction)
                state_all01 = np.append(state_all01, self.game.board)                            
            elif 64<=self.game.score <128:
                direction_all0=np.append(direction_all0,direction)
                state_all0 = np.append(state_all0, self.game.board)
            elif 128<=self.game.score <256:
                direction_all1=np.append(direction_all1,direction)
                state_all1 = np.append(state_all1, self.game.board)
            elif 256 <=self.game.score <512:
                direction_all2=np.append(direction_all2,direction)
                state_all2 = np.append(state_all2, self.game.board)
            elif 512<=self.game.score<1024:
                direction_all3=np.append(direction_all3,direction)
                state_all3 = np.append(state_all3, self.game.board)
            elif 1024<=self.game.score:
                direction_all4=np.append(direction_all4,direction)
                state_all4 = np.append(state_all4, self.game.board)
            else:
                self.game.end=True
            #print('now:')
           # print(direction_all)
            self.game.move(direction)
            n_iter += 1
            #state_all=np.append(state_all,self.game.board)
            #state_all=state_all.reshape(n_iter,4,4)
            if verbose:
                # print("Iter: {}".format(n_iter))
                # print("======Direction: {}======".format(
                    # ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)
                   # print(state_all)       
        return state_all00,direction_all00,state_all01,direction_all01,state_all0,direction_all0,state_all1,direction_all1,state_all2,direction_all2,state_all3,direction_all3,state_all4,direction_all4

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):
   # state_all=[[]]
    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
       # from model import DQN_MLP, DDQN_MLP
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction
