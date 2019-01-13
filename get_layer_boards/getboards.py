from game2048.game import Game
from game2048.displays import Display
import numpy as np
import json

def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    oneboard00,onedirection00,oneboard01,onedirection01,oneboard0,onedirection0,oneboard1,onedirection1,oneboard2,onedirection2,oneboard3,onedirection3,oneboard4,onedirection4 = agent.play(verbose=True)
    #agent.play(verbose=True)
    return game.score,oneboard00,onedirection00,oneboard01,onedirection01,oneboard0,onedirection0,oneboard1,onedirection1,oneboard2,onedirection2,oneboard3,onedirection3,oneboard4,onedirection4


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 10000

    '''====================
    Use your own agent here.'''
    from game2048.agents import ExpectiMaxAgent as TestAgent
    #from game2048.agents import RandomAgent as TestAgent
    '''===================='''

    scores = []
    all_board00=[]
    all_direction00=[]   
    all_board01=[]
    all_direction01=[]
    all_board0=[]
    all_direction0=[]
    all_board1=[]
    all_direction1=[]
    all_board2=[]
    all_direction2=[]
    all_board3=[]
    all_direction3=[]
    all_board4=[]
    all_direction4=[]
    b_name00='b32x4.npy'
    d_name00='d32x4.npy'
    b_name01='b64x4.npy'
    d_name01='d64x4.npy'
    b_name0='b128x4.npy'
    d_name0='d128x4.npy'
    b_name1 = 'b256x4.npy' #通过扩展名指定文件存储的数据为json格式
    d_name1 = 'd256x4.npy'
    b_name2 = 'b512x4.npy' #通过扩展名指定文件存储的数据为json格式
    d_name2 = 'd512x4.npy'
    b_name3 = 'b1024x4.npy' #通过扩展名指定文件存储的数据为json格式
    d_name3 = 'd1024x4.npy'
    b_name4 = 'b2048x4.npy'
    d_name4 = 'd2048x4.npy'
    np.save(b_name00,all_board00)
    np.save(d_name00,all_direction00)
    np.save(b_name01,all_board01)
    np.save(d_name01,all_direction01)
    np.save(b_name0,all_board0)
    np.save(d_name0,all_direction0)
    np.save(b_name1,all_board1)
    np.save(d_name1,all_direction1)
    np.save(b_name2,all_board2)
    np.save(d_name2,all_direction2)
    np.save(b_name3,all_board3)
    np.save(d_name3,all_direction3)
    np.save(b_name4,all_board4)
    np.save(d_name4,all_direction4)
    for _ in range(N_TESTS):
        score,oneboard00,onedirection00,oneboard01,onedirection01,oneboard0,onedirection0,oneboard1,onedirection1,oneboard2,onedirection2,oneboard3,onedirection3,oneboard4,onedirection4 = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent)
        scores.append(score)

        all_board00=np.load(b_name00)
        all_board00 = np.append(all_board00, oneboard00)
        np.save(b_name00, all_board00)
        all_direction00 = np.load(d_name00)
        all_direction00 = np.append(all_direction00, onedirection00)
        np.save(d_name00, all_direction00)

        all_board01=np.load(b_name01)
        all_board01 = np.append(all_board01, oneboard01)
        np.save(b_name01, all_board01)
        all_direction01 = np.load(d_name01)
        all_direction01 = np.append(all_direction01, onedirection01)
        np.save(d_name01, all_direction01)

        all_board0=np.load(b_name0)
        all_board0 = np.append(all_board0, oneboard0)
        np.save(b_name0, all_board0)
        all_direction0 = np.load(d_name0)
        all_direction0 = np.append(all_direction0, onedirection0)
        np.save(d_name0, all_direction0)

        all_board1=np.load(b_name1)
        all_board1 = np.append(all_board1, oneboard1)
        np.save(b_name1, all_board1)
        all_direction1 = np.load(d_name1)
        all_direction1 = np.append(all_direction1, onedirection1)
        np.save(d_name1, all_direction1)

        all_board2 = np.load(b_name2)
        all_board2 = np.append(all_board2, oneboard2)
        np.save(b_name2, all_board2)
        all_direction2 = np.load(d_name2)
        all_direction2 = np.append(all_direction2, onedirection2)
        np.save(d_name2, all_direction2)

        all_board3 = np.load(b_name3)
        all_board3 = np.append(all_board3, oneboard3)
        np.save(b_name3, all_board3)
        all_direction3 = np.load(d_name3)
        all_direction3 = np.append(all_direction3, onedirection3)
        np.save(d_name3, all_direction3)

        all_board4 = np.load(b_name4)
        all_board4 = np.append(all_board4, oneboard4)
        np.save(b_name4, all_board4)
        all_direction4 = np.load(d_name4)
        all_direction4 = np.append(all_direction4, onedirection4)
        np.save(d_name4, all_direction4)

    all_board00=np.load(b_name00)
    all_direction00=np.load(d_name00)
    num00=int(np.size(all_board00)/16)
    all_board00=all_board00.reshape(num00,4,4)
    all_direction00=all_direction00.reshape(num00,1)
    np.save(b_name00,all_board00)
    np.save(d_name00,all_direction00)

    all_board01=np.load(b_name01)
    all_direction01=np.load(d_name01)
    num01=int(np.size(all_board01)/16)
    all_board01=all_board01.reshape(num01,4,4)
    all_direction01=all_direction01.reshape(num01,1)
    np.save(b_name01,all_board01)
    np.save(d_name01,all_direction01)

    all_board0=np.load(b_name0)
    all_direction0=np.load(d_name0)
    num0=int(np.size(all_board0)/16)
    all_board0=all_board0.reshape(num0,4,4)
    all_direction0=all_direction0.reshape(num0,1)
    np.save(b_name0,all_board0)
    np.save(d_name0,all_direction0)

    all_board1=np.load(b_name1)
    all_direction1=np.load(d_name1)
    num1=int(np.size(all_board1)/16)
    all_board1=all_board1.reshape(num1,4,4)
    all_direction1=all_direction1.reshape(num1,1)
    np.save(b_name1,all_board1)
    np.save(d_name1,all_direction1)

    all_board2=np.load(b_name2)
    all_direction2=np.load(d_name2)
    num2=int(np.size(all_board2)/16)
    all_board2=all_board2.reshape(num2,4,4)
    all_direction2=all_direction2.reshape(num2,1)
    np.save(b_name2,all_board2)
    np.save(d_name2,all_direction2)

    all_board3=np.load(b_name3)
    all_direction3=np.load(d_name3)
    num3=int(np.size(all_board3)/16)
    all_board3=all_board3.reshape(num3,4,4)
    all_direction3=all_direction3.reshape(num3,1)
    np.save(b_name3,all_board3)
    np.save(d_name3,all_direction3)

    all_board4=np.load(b_name4)
    all_direction4=np.load(d_name4)
    num4=int(np.size(all_board4)/16)
    all_board4=all_board4.reshape(num4,4,4)
    all_direction4=all_direction4.reshape(num4,1)
    np.save(b_name4,all_board4)
    np.save(d_name4,all_direction4)


    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))

