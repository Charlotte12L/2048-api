import numpy as np
import torch
import time
class Agent:
    '''Agent Base.'''

    def __init__(self, model,game, display=None):
        self.game = game
        self.display = display
        self.model = model

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
           # print(type(direction),direction)
            self.game.move(direction)
            #print(time.time())
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        print(type(self.game.board),self.game.board)
        return direction

class LXYAgent(Agent):

    def __init__(self,model, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(model, game, display)
        '''  
        from .Model import Net
        model = Net()
        model.load_state_dict(torch.load("./game2048/epoch_24.pkl", map_location='cpu'))
        model.eval()'''
        self.model =model
        self.search_func = model


    def step(self):
        new_board = self.game.board
        new_board[new_board== 0] = 1
        new_board=np.log2(new_board)
        new_board = np.reshape(new_board, (-1, 4, 4))
        new_board = torch.FloatTensor(new_board)
        new_board = new_board.unsqueeze(dim=1)
        output = self.search_func(new_board)
        direction = output.data.max(1, keepdim=True)[1]
        #direction = direction.Variable(direction)
       # print(type(self.game.board),self.game.board)
        return direction
