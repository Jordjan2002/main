import copy
import gym
import chess
import gym_chess
from helpers import custom_reset, evaluate_minimax, king_distance_reward, evaluation_KBNK
from generatingEndgame import generate_endgame_FEN
from tqdm import tqdm

from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget

import random
import chess.svg
import chess.engine
engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\\20203734\\Downloads\\stockfish-11-win\\stockfish\\stockfish-windows-x86-64-avx2")

#nr_nodes = 0

nr_test = 10
stockfish = True # stockfish opponent, else random opponent
pieces = [chess.ROOK] # pieces to checkmate with
reward_func = king_distance_reward


def minimax(env, depth, turn):

    obs = env._observation()   
    if depth == 0 or obs.is_game_over():
        # print('naar dit')
        # print(env.render())
        # print(evaluate_minimax(env))
        # if evaluate_minimax_KQK(env) == 100:
        #     print(env.render())
        #     print(' ')
        return reward_func(env) # evaluate_old(env.observation)
    
    if turn: # if white to move:
        eval = float('-inf')
        for move in env.legal_moves:
            #nr_nodes += 1
            new_env = copy.deepcopy(env)
            new_env.step(move)
            eval = max(eval, minimax(new_env, depth-1, False))
        
        return eval
    
    else:
        eval = float('inf')
        for move in env.legal_moves:
            new_env = copy.deepcopy(env)
            new_env.step(move)
            eval = min(eval, minimax(new_env, depth-1, True))

        return eval


def play_minimax_move(env, depth, turn, positions):
    moves = env.legal_moves
    moves_score = []
    for move in moves:
        new_env = copy.deepcopy(env)
        new_env.step(move)
        
        if new_env._observation().is_repetition(): #positions.count(new_env._observation().board_fen()) >= 3: # i.e. avoid repeting positions and therefore three fold repetitions
            score = 0
            
        else:    
            score = minimax(new_env, depth, not turn)

        if new_env._observation().is_checkmate():
            score = 1000

        moves_score.append((move,score))

    # print(moves_score)
    # print(max(moves_score, key=lambda x: x[1]))

    best_move, eval = max(moves_score, key=lambda x: x[1])
    #print(best_move, eval)
    return best_move


env = gym.make('Chess-v0')
env.reset = custom_reset


for depth in range(1,4):
    results = []
    for i in tqdm(range(nr_test)):
        nr_moves = 0
        Fen = generate_endgame_FEN([i for i in range(64)], pieces)
        obs = env.reset(env, Fen)
        # print(env.render())
        # print(' ')
        positions = [obs.board_fen()] # used to take into account 3 fold repeitiion
        done = False

        while not done and nr_moves <= 100:
            if obs.turn: # if white to move
                best_move = play_minimax_move(copy.deepcopy(env), depth, obs.turn, positions) 
                #print(nr_nodes)
                obs, reward, done, info = env.step(best_move)
                # print(env.render())
                # print(' ')
                positions.append(obs.board_fen())
                nr_moves += 1

            else:
                moves = env.legal_moves

                if stockfish:
                    move = engine.play(obs, chess.engine.Limit(time=0.1)).move
                else:
                    move = random.choice(moves)
                obs, reward, done, info = env.step(move)
                # print(env.render())
                # print(' ')
                positions.append(obs.board_fen())
                nr_moves += 1


        if obs.is_checkmate():
            results.append(1)
        else:
            results.append(0)


    print('depth:',depth, 'win rate:', sum(results)/len(results))

env.close()