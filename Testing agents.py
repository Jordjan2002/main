import gym
import chess
import gym_chess
import random
from helpers import custom_reset, custom_step_reward, find_max_Q_move_KQK, pos_to_index_KQK, king_distance_reward, pos_to_index, find_max_Q_move
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import chess.engine

from tqdm import tqdm
from timeit import default_timer as timer
from generatingEndgame import generate_endgame_FEN
from tempfile import TemporaryFile
from numpy import genfromtxt
from itertools import compress

engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\\20203734\\Downloads\\stockfish-11-win\\stockfish\\stockfish-windows-x86-64-avx2")

with open(r"C:\Users\20203734\OneDrive - TU Eindhoven\Yearfour\BEPChess\Agents\qtable-1716482263.pickle", 'rb') as f:
    data = pickle.load(f)
Q_table = np.array(data)

print()

chess_board = [i for i in range(64)]
print([chess.square_name(square) for square in chess_board])

chess_pieces = [chess.QUEEN]

env = gym.make('Chess-v0')
env.reset = custom_reset
env.step = custom_step_reward

nr_test = 100
starting_positions = [generate_endgame_FEN(chess_board, chess_pieces) for i in range(nr_test)]
results = []
stalemates = 0
loses_piece = 0
repetitions = 0
fifty_move_rules = 0

# playing against random agent
for starting_pos in tqdm(starting_positions):
    obs = env.reset(env, starting_pos)

    done = False
    checkmate = False
    played_moves = []
    positions = []

    while not done and len(played_moves) < 100:

        if obs.turn:
            K, Q, k = pos_to_index_KQK(obs.piece_map())
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in chess_board]

            # three_folds = []
            # for move in moves:
            #     new_env = copy.deepcopy(env)
            #     new_env.step(new_env, move)
            #     if positions.count(env._observation()) >= 3:
            #         three_folds.append(False)
            #     else:
            #         three_folds.append(True)

            # moves = list(compress(moves, three_folds))


            Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, Q, k)
            obs, reward, done, info = env.step(env, best_move)
            positions.append(obs)
            played_moves.append(best_move)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                else:
                    if not obs.is_repetition(count=5):
                        stalemates += 1
                break

        else:   
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in chess_board]
            if not moves:
                if obs.is_check():
                    checkmate = True
                else:
                    stalemates += 1
                break
            
            move_to_play = random.choice(moves)
            obs, reward, done, info = env.step(env, move_to_play)
            positions.append(obs)
            played_moves.append(move_to_play)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                break

    if checkmate:
        results.append(1)
    else:
        if len(played_moves) >= 100:
            fifty_move_rules += 1
        if obs.is_insufficient_material():
            loses_piece += 1
        if obs.is_repetition(count=5):
            repetitions += 1
        results.append(0)


print(sum(results), stalemates, fifty_move_rules, loses_piece, repetitions)
print(sum(results)/len(results))
print(len(results))


results = []
# playing against trained agent

stalemates = 0
fifty_move_rules = 0
loses_piece = 0
repetitions = 0
for starting_pos in tqdm(starting_positions):
    obs = env.reset(env, starting_pos)

    done = False
    played_moves = []
    checkmate = False


    while not done and len(played_moves) < 100:

        if obs.turn:
            K, Q, k = pos_to_index_KQK(obs.piece_map())
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in chess_board]
            Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, Q, k)
            obs, reward, done, info = env.step(env, best_move)
            played_moves.append(best_move)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                else:
                    if not obs.is_repetition(count=5):
                        stalemates += 1

                break


        else:   
            K, Q, k = pos_to_index_KQK(obs.piece_map())
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in chess_board]
            if not moves:
                if obs.is_check():
                    checkmate = True
                else:
                    stalemates += 1
                break

            Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, Q, k)
            obs, reward, done, info = env.step(env, best_move)
            played_moves.append(best_move)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                break

    if checkmate:
        results.append(1)
    else:
        # print(env.render())
        # print(played_moves)
        if len(played_moves) >= 100:
            fifty_move_rules += 1
        if obs.is_insufficient_material():
            loses_piece += 1
        if obs.is_repetition(count=5):
            repetitions += 1

        results.append(0)


print(sum(results), stalemates, fifty_move_rules, loses_piece, repetitions)
print(sum(results)/len(results))
print(len(results))


results = []
stalemates = 0
loses_piece = 0
repetitions = 0
fifty_move_rules = 0
# playing against heuristic agent
for starting_pos in tqdm(starting_positions):
    obs = env.reset(env, starting_pos)

    done = False
    played_moves = []
    checkmate = False

    while not done and len(played_moves) < 100:

        if obs.turn:
            K, Q, k = pos_to_index_KQK(obs.piece_map())
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in chess_board]
            Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, Q, k)
            obs, reward, done, info = env.step(env, best_move)
            played_moves.append(best_move)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                else:
                    if not obs.is_repetition(count=5):
                        stalemates += 1
                break

        else:   
            K, Q, k = pos_to_index_KQK(obs.piece_map())
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in chess_board]

            if not moves:
                if obs.is_check():
                    checkmate = True
                else:
                    stalemates += 1
                break

            # if sum([move for move in moves if obs.is_capture(move)]) >= 1:
            #     move_to_play = random.choice([move for move in moves if obs.is_capture(move)])

            else:
                move_scores = []
                for move in moves:
                    new_env = copy.deepcopy(env)
                    new_env.step(new_env, move)
                    reward = king_distance_reward(env)
                    move_scores.append((move, reward))
                move_to_play, score = min(move_scores,  key = lambda x: x[1])

            obs, reward, done, info = env.step(env, move_to_play)
            played_moves.append(move_to_play)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                break

    if checkmate:
        results.append(1)
    else:
        if len(played_moves) >= 100:
            fifty_move_rules += 1
        if obs.is_insufficient_material():
            loses_piece += 1
        if obs.is_repetition(count=5):
            repetitions += 1
        results.append(0)


print(sum(results), stalemates, fifty_move_rules, loses_piece, repetitions)
print(sum(results)/len(results))
print(len(results))


results = []
stalemates = 0
loses_piece = 0
repetitions = 0
fifty_move_rules = 0
# playing against Stockfish
for starting_pos in tqdm(starting_positions):
    obs = env.reset(env, starting_pos)

    done = False
    played_moves = []
    checkmate = False

    while not done and len(played_moves) < 100:

        if obs.turn:
            K, Q, k = pos_to_index_KQK(obs.piece_map())
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in chess_board]
            Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, Q, k)
            obs, reward, done, info = env.step(env, best_move)
            played_moves.append(best_move)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                else:
                    if not obs.is_repetition(count=5):
                        stalemates += 1
                break

        else:   
            K, Q, k = pos_to_index_KQK(obs.piece_map())
            moves = env.legal_moves
            moves = [move for move in moves if move.to_square in chess_board]

            if not moves:
                if obs.is_check():
                    checkmate = True
                else:
                    stalemates += 1
                break

            # if sum([move for move in moves if obs.is_capture(move)]) >= 1:
            #     move_to_play = random.choice([move for move in moves if obs.is_capture(move)])

            else:

                best_move_black = engine.play(obs, chess.engine.Limit(time=0.1)).move
                
            obs, reward, done, info = env.step(env, best_move_black)
            played_moves.append(best_move_black)
            if done:
                if obs.is_checkmate():
                    checkmate = True
                break

    if checkmate:
        results.append(1)
    else:
        if len(played_moves) >= 100:
            fifty_move_rules += 1
        if obs.is_insufficient_material():
            loses_piece += 1
        if obs.is_repetition(count=5):
            repetitions += 1
        results.append(0)


print(sum(results), stalemates, fifty_move_rules, loses_piece, repetitions)
print(sum(results)/len(results))
print(len(results))

