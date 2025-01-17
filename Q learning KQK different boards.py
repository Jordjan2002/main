import gym
import chess
import gym_chess
import random
from helpers import custom_reset, custom_step, find_max_Q_move_KQK, pos_to_index_KQK
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import os
from tqdm import tqdm
from timeit import default_timer as timer
from generatingEndgame import generate_endgame_FEN

env = gym.make('Chess-v0')
env.reset = custom_reset
env.step = custom_step

nr_of_pieces_white = 2
nr_of_pieces_black = 1
squares = 64

# initiating Q-learning
learning_rate = 0.01
Show_Every = 500
discount = 0.99
episodes = 30_000
white_epsilon = 0.9 #0.5
black_epsilon = 0.9
min_epsilon = 0.05
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = (white_epsilon-min_epsilon)/(end_epsilon_decaying-start_epsilon_decaying)

chess_boards = [[32,33,34,35,40,41,42,43,48,49,50,51,56,57,58,59],[24,25,26,27,28,32,33,34,35,36,40,41,42,43,44,48,49,50,51,52,56,57,58,59,60],
                [16,17,18,19,20,21, 24,25,26,27,28,29, 32,33,34,35,36,37, 40,41,42,43,44,45, 48,49,50,51,52,53, 56,57,58,59,60,61],
                [8,9,10,11,12,13,14, 16,17,18,19,20,21,22,  24,25,26,27,28,29,30, 32,33,34,35,36,37,38, 40,41,42,43,44,45,46, 48,49,50,51,52,53,54, 56,57,58,59,60,61,62],[i for i in range(64)]] 

board_sizes = ['avg 4x4', 'avg 5x5', 'avg 6x6', 'avg 7x7', 'avg 8x8']
results = []
white_epsilon_values = []
black_epsilon_values = []
aggr_results = {'ep': [i for i in range(0, episodes+Show_Every, Show_Every)] , 'avg 4x4':[], 'avg 5x5': [], 'avg 6x6': [], 'avg 7x7':[], 'avg 8x8':[] }


board_index = 0
for chess_board in chess_boards:
    Q_table_white = np.zeros([squares, squares, squares,nr_of_pieces_white,squares])
    Q_table_black = np.ones([squares, squares, squares,nr_of_pieces_white,squares])*50
    Q_table = np.append(Q_table_white, Q_table_black, axis=3)
    board_size = board_sizes[board_index]
    for episode in tqdm(range(episodes+1)):

        played_moves = [] # keep track of moves
        #Fen = random.choice(Fens) # taking one of the FEN positions
        Fen = generate_endgame_FEN(chess_board, [chess.QUEEN])
        obs = env.reset(env, Fen)
        done = False
        nr_moves = 0

        while not done and len(played_moves) < 50: # with 50 move rule

            checkmate = False
            nr_moves +=1

            if obs.turn: # if White to move
                K, R, k = pos_to_index_KQK(obs.piece_map())
                moves = env.legal_moves
                moves = [move for move in moves if move.to_square in chess_board] # only regard moves in 4x4 board

                if np.random.random() > white_epsilon: # chance based on epsilon to look into random move
                    Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, R, k)
                else:
                    Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, R, k, random_move=True)
                obs, reward, done, info = env.step(env, best_move)

                played_moves.append(best_move)

                if done: # if the game ends new Q_value will be the reward
                    Q_table[K][R][k][a][b] = reward
                    if obs.is_checkmate():
                        checkmate = True
                    break # go to next episode


                # look one extra step ahead because of different Q updating function.
                new_env = copy.deepcopy(env)
                K_t1, R_t1, k_t1 = pos_to_index_KQK(obs.piece_map())
                moves = new_env.legal_moves
                moves = [move for move in moves if move.to_square in chess_board] # only regard moves in 4x4 board

                if not moves: # if black will have no moves in the smaller board after move of white it is either checkmate or stalemate and game is over
                    if obs.is_check():
                        checkmate = True
                        reward = 50
                        Q_table[K][R][k][a][b] = reward
                    else:
                        reward = -50
                        Q_table[K][R][k][a][b] = reward
                    break

                Q_value_best_move_t1, best_move_t1, a_t1, b_t1 = find_max_Q_move_KQK(moves, Q_table, obs, K_t1, R_t1, k_t1)
                obs_t1, reward_t1, done_t1, info_t1 = new_env.step(new_env, best_move_t1)
                

                # find max Q-value after opponent would play best move
                if done_t1: # if black can end the game
                    est_fut_value = -50
                    Q_table[K][R][k][a][b] = Q_value_best_move + learning_rate*(reward+discount*est_fut_value-Q_value_best_move)

                else:
                    K_t2, R_t2, k_t2 = pos_to_index_KQK(obs.piece_map())
                    moves = new_env.legal_moves
                    moves = [move for move in moves if move.to_square in chess_board] # only regard moves in 4x4 board

                    Q_value_best_move_t2, best_move_t2, a_t2, b_t2 = find_max_Q_move_KQK(moves, Q_table, obs, K_t2, R_t2, k_t2)

                    #update the value in the Q-table
                    Q_table[K][R][k][a][b] =  Q_value_best_move + learning_rate*(reward+discount*Q_value_best_move_t2-Q_value_best_move) 


            else: # if Black to move
                K, R, k = pos_to_index_KQK(obs.piece_map())
                moves = env.legal_moves
                moves = [move for move in moves if move.to_square in chess_board] # only regard moves in 4x4 board

                if np.random.random() > black_epsilon: # chance based on epsilon to look into random move
                    Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, R, k)
                else:
                    Q_value_best_move, best_move, a, b = find_max_Q_move_KQK(moves, Q_table, obs, K, R, k, random_move=True)            

                obs, reward, done, info = env.step(env, best_move)

                played_moves.append(best_move)

                if done: # if the game ends new Q_value will be the reward
                    Q_table[K][R][k][a][b] = reward # 1-reward, because interest of black is opposite
                    break


                # look one extra step ahead because of different Q updating function.
                new_env = copy.deepcopy(env)
                K_t1, R_t1, k_t1 = pos_to_index_KQK(obs.piece_map())
                moves = new_env.legal_moves
                moves = [move for move in moves if move.to_square in chess_board] # only regard moves in 4x4 board

                Q_value_best_move_t1, best_move_t1, a_t1, a_t2 = find_max_Q_move_KQK(moves, Q_table, obs, K_t1, R_t1, k_t1)
                obs_t1, reward_t1, done_t1, info_t1 = new_env.step(new_env, best_move_t1)

                if done_t1:
                    #Q_table[K][R][k][a][b] = -reward_t1
                    est_fut_value = 0 #reward_t1
                    Q_table[K][R][k][a][b] =  Q_value_best_move + learning_rate*(reward+discount*est_fut_value-Q_value_best_move) # ipv het gewoon (-) reward maken


                # find max Q-value after opponent would play best move
                else:
                    K_t2, R_t2, k_t2 = pos_to_index_KQK(obs.piece_map())
                    moves = new_env.legal_moves
                    moves = [move for move in moves if move.to_square in chess_board] # only regard moves in 4x4 board

                    if not moves: # if black will have no moves in the smaller board after best move of white it is either checkmate or stalemate and game is over
                        if obs_t1.is_check():
                            estimated_future_value = 0
                            Q_table[K][R][k][a][b] = Q_value_best_move + learning_rate*(reward+discount*estimated_future_value-Q_value_best_move)
                        else:
                            estimated_future_value = 0
                            Q_table[K][R][k][a][b] = Q_value_best_move + learning_rate*(reward+discount*estimated_future_value-Q_value_best_move)

                    else:
                        Q_value_best_move_t2, best_move_t2, a_t2, b_t2 = find_max_Q_move_KQK(moves, Q_table, obs, K_t2, R_t2, k_t2)
                        Q_table[K][R][k][a][b] =  Q_value_best_move + learning_rate*(reward+discount*Q_value_best_move_t2-Q_value_best_move)
            

        if end_epsilon_decaying >= episode >= start_epsilon_decaying:
            white_epsilon -= epsilon_decay_value
            black_epsilon -= epsilon_decay_value

        if env._board.is_checkmate() or reward == 50: #or reward == -1: # last two to make sure it prints it out well if checkmate with 4x4 board
                results.append(1)

        else: # not env._board.is_checkmate() or reward == 50: # or reward == -1: # last two to make sure it prints it out well if checkmate with 4x4 board
                results.append(0)

        # white_epsilon_values.append(white_epsilon)
        # black_epsilon_values.append(black_epsilon)


# code for printing out the results
        if episode % Show_Every == 0:
                print('episode:', episode)
                print(env.render())
                if env._board.is_checkmate(): # or reward == -1: # last two to make sure it prints it out well if checkmate with 4x4 board
                    result = 'checkmate'
                else:
                    result = 'draw'
                print(result, 'in', nr_moves, 'moves')  
                print(played_moves)
                print('reward:', reward)
                #print(f'Episode: {episode} avg: {average_reward} min: {min(results[-Show_Every:])} max: {max(results[-Show_Every:])}')    
                print(' ')   

        if not episode % Show_Every:
            average_reward = sum(results[-Show_Every:])/len(results[-Show_Every:])
            aggr_results[board_size].append(average_reward)

    # Navigeer naar de gewenste directory
    os.chdir(r"C:\Users\20203734\OneDrive - TU Eindhoven\Yearfour\BEPChess")

    # Opslaan van het object naar een pickle-bestand in de huidige directory

    with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(Q_table, f)

    board_index += 1


    
env.close()

plt.plot(aggr_results['ep'], aggr_results['avg 4x4'], label='avg win rate 4x4')
plt.plot(aggr_results['ep'], aggr_results['avg 5x5'], label='avg win rate 5x5')
plt.plot(aggr_results['ep'], aggr_results['avg 6x6'], label='avg win rate 6x6')
plt.plot(aggr_results['ep'], aggr_results['avg 7x7'], label='avg win rate 7x7')
plt.plot(aggr_results['ep'], aggr_results['avg 8x8'], label='avg win rate 8x8')

plt.title('Q-learning King Queen vs king on different board sizes')
plt.xlabel('episode')
plt.ylabel('value')
plt.legend()
plt.show()

plt.savefig('Q-learning simulations')
