from collections import deque
from os import path, mkdir
import threading
import time
import math
import pickle
import numpy as np
import pickle
import concurrent.futures
import random
from functools import reduce

import sys
sys.path.append('../build')
from library import MCTS, MCTS_PURE, Gamecore, predict_model, Int2StrMap, single_lib
from overall_predictor import overall_predictor
from neural_network import NeuralNetWorkWrapper

def tuple_2d_to_numpy_2d(tuple_2d):
    # help function
    # convert type
    res = [None] * len(tuple_2d)
    for i, tuple_1d in enumerate(tuple_2d):
        res[i] = list(tuple_1d)
    return np.array(res)
class random_player():
    def __init__(self, idx_job_map_c):
        self.init = True
        self.idx_job_map_c = idx_job_map_c

    def get_action_probs(self, gamecore, temp):
        legal_moves = list(gamecore.get_legal_moves(self.idx_job_map_c))
        return_probs = np.zeros([98])
        # print(legal_moves, len(legal_moves))
        while True:
            action = random.randint(0, 97)
            if legal_moves[action] == 1:
                break
        return_probs[action] = 1
        return return_probs
    
    def update_with_move(self, best_move):
        pass

class greedy_player():
    def __init__(self, idx_job_map_c, idx_job_map):
        import csv
        f = open('../hero.csv', 'rt', encoding='utf8')
        reader = csv.reader(f)
        for line in reader:
            if line[0]=='winrate':
                self.data_list = line
            if line[0]=='':
                self.order_list = line
            if line[0]=='id':
                self.id_list = line
        self.id_to_order_dict = {}
        for i in range(len(self.order_list)):
            self.id_to_order_dict[i-1] = self.order_list[i]
        self.data_list = self.data_list[1:]
        self.id_list = self.id_list[1:]
        self.data_list = list(zip(self.data_list, self.id_list))
        self.data_list.sort(reverse=True, key=lambda x:x[0])
        self.idx_job_map_c = idx_job_map_c
        self.idx_job_map = idx_job_map
        self.job_list_for_current_round = []

    def get_action_probs(self, gamecore, temp):
        legal_moves = list(gamecore.get_legal_moves(self.idx_job_map_c))
        if len(self.job_list_for_current_round) == 5:
            self.job_list_for_current_round = []
        selected_hero = -1
        for i in range(len(self.data_list)):
            ID = int(self.data_list[i][1])
            if self.idx_job_map[ID] not in self.job_list_for_current_round and legal_moves[ID] == 1:
                selected_hero = ID
                self.job_list_for_current_round.append(self.idx_job_map[ID])
                print(selected_hero, self.job_list_for_current_round)
                break
        # 3 heroes is not listed
        return_probs = np.zeros([len(self.data_list)+3])
        return_probs[selected_hero] = 1
        return return_probs
    def update_with_move(self, best_move):
        pass

class Leaner():
    def __init__(self, config):
        # see config.py
        # gamecore
        self.init_job_info(config['hero_job_file'])
        self.main_player_order = config['main_player_order']
        self.main_round_order = config['main_round_order']
        self.action_size = config['hero_numbers']
        self.consider_pick_only = config['consider_pick_only']
        self.max_round = config['game_number']
        self.step_num_of_round = config['step_num_of_round']

        # train
        self.num_iters = config['num_iters']
        self.num_eps = config['num_eps']
        self.num_train_threads = config['num_train_threads']
        self.check_freq = config['check_freq']
        self.num_contest = config['num_contest']
        self.dirichlet_alpha = config['dirichlet_alpha']
        self.temp = config['temp']
        self.update_threshold = config['update_threshold']
        self.num_explore = config['num_explore']

        self.examples_buffer = deque([], maxlen=config['examples_buffer_max_len'])

        # mcts
        self.num_mcts_sims = config['num_mcts_sims']
        self.c_puct = config['c_puct']
        self.c_virtual_loss = config['c_virtual_loss']
        self.num_mcts_threads = config['num_mcts_threads']
        self.libtorch_use_gpu = config['libtorch_use_gpu']

        # neural network
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.nnet = overall_predictor(config)

    def init_job_info(self, hero_job_file):
        configid_list = []
        hero_job_map = {}
        idx_job_map = {}
        hero_idx_map = {}
        idx_hero_map = {}
        with open(hero_job_file, "rb") as f:
            for line in f.readlines():
               sp_line = line.decode().strip().split(" ")
               configid = int(sp_line[0])
               job_name = sp_line[2]
               configid_list.append(configid)
               hero_job_map[configid] = job_name
        configid_list.sort()
        for i in range(len(configid_list)):
            hero_idx_map[configid_list[i]] = i
            idx_hero_map[i] = configid_list[i]
            idx_job_map[i] = hero_job_map[configid_list[i]]
        idx_job_map_c = Int2StrMap()
        for k,v in idx_job_map.items():
            idx_job_map_c[k] = v
        self.configid_list = configid_list
        self.hero_job_map = hero_job_map
        self.idx_job_map = idx_job_map
        self.hero_idx_map = hero_idx_map
        self.idx_hero_map = idx_hero_map
        self.idx_job_map_c = idx_job_map_c

    def learn(self):
        # train the model by self play

        if path.exists(path.join('models', 'checkpoint.example')):
            print("loading checkpoint...")
            self.nnet.load_model()
            self.nnet.parallel()
            self.load_samples()
        else:
            # save torchscript
            self.nnet.parallel()
            self.nnet.save_model()
            self.nnet.save_model('models', "best_checkpoint")

        for itr in range(1, self.num_iters + 1):

            # self play in parallel
            #libtorch = predict_model('./models/checkpoint.pt',
            #                         self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads)
            itr_examples = []
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_train_threads) as executor:
                futures = [executor.submit(self.self_play, k) for k in range(1, self.num_eps + 1)]
                for k, f in enumerate(futures):
                    examples = f.result()
                    itr_examples += examples

                    # decrease libtorch batch size
                    remain = min(len(futures) - (k + 1), self.num_train_threads)
                    # libtorch.set_batch_size(max(remain * self.num_mcts_threads, 1))
                    print("EPS: {}, EXAMPLES: {}".format(k + 1, len(examples)))

            # release gpu memory
            # del libtorch
            end_time = time.time()
            print("ITER :: {}, PROCESS_TIME:: {}".format(itr, end_time-start_time))

            # prepare train data
            self.examples_buffer.append(itr_examples)
            train_data = reduce(lambda a, b : a + b, self.examples_buffer)
            print('training start')
            random.shuffle(train_data)

            # train neural network
            epochs = self.epochs * (len(itr_examples) + self.batch_size - 1) // self.batch_size
            self.nnet.train(train_data, self.batch_size, int(epochs))
            self.nnet.save_model()
            # self.save_samples()

            # compare performance
            if False:
            # if itr % self.check_freq == 0:
                libtorch_current = predict_lib('./models/checkpoint.pt',
                                         self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2, 0)
                libtorch_best = predict_lib('./models/best_checkpoint.pt',
                                              self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2, 0)

                one_won, two_won, draws = self.contest(libtorch_current, libtorch_best, self.num_contest)
                print("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (one_won, two_won, draws))

                if one_won + two_won > 0 and float(one_won) / (one_won + two_won) > self.update_threshold:
                    print('ACCEPTING NEW MODEL')
                    self.nnet.save_model('models', "best_checkpoint")
                else:
                    print('REJECTING NEW MODEL')

                # release gpu memory
                del libtorch_current
                del libtorch_best
    
    def self_play(self, order):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.
        """
        gpu_selected = (order-1) % 8
        libtorch = predict_model('./models/checkpoint.pt', self.libtorch_use_gpu, self.num_mcts_threads, gpu_selected)
        s_lib = single_lib('./models/checkpoint_single.pt', self.num_mcts_threads, gpu_selected)
        train_examples = []

        player1 = MCTS(libtorch, self.num_mcts_threads, self.c_puct,
                    self.num_mcts_sims, self.c_virtual_loss, self.action_size,
                    self.max_round, 0, 0, self.step_num_of_round, self.idx_job_map_c)
        player2 = MCTS(libtorch, self.num_mcts_threads, self.c_puct,
                    self.num_mcts_sims, self.c_virtual_loss, self.action_size,
                    self.max_round, 0, 0, self.step_num_of_round, self.idx_job_map_c)
        players_order = [-1]
        for order in self.main_player_order:
            players_order.append(order+1)
        players = [None, player1, player2]
        player_index = 1

        gamecore = Gamecore(self.max_round, self.action_size, self.consider_pick_only, s_lib)
        

        episode_step = 0
        while True:
            episode_step += 1
            
            player = players[players_order[episode_step]]

            # get action prob
            if episode_step <= self.num_explore:
                prob = np.array(list(player.get_action_probs(gamecore, self.temp)))
            else:
                prob = np.array(list(player.get_action_probs(gamecore, 0)))
            #prob = np.array(list(player.get_action_probs(gamecore, self.temp)))

            # generate sample
            current_feature = gamecore.get_feature()
            #last_action = gamecore.get_last_move()
            current_step = gamecore.get_current_step()
            current_round = gamecore.get_current_round()
            feature = list(current_feature)
            train_examples.append([feature, prob, current_step, current_round])
            # dirichlet noise
            legal_moves = list(gamecore.get_legal_moves(self.idx_job_map_c))
            noise = 0.1 * np.random.dirichlet(self.dirichlet_alpha * np.ones(np.count_nonzero(legal_moves)))
            #if current_step == 9:
            #    print([feature, prob, current_step, current_round])
            #for p in prob:
            #    if np.isnan(p):
            #        print([feature, prob, current_step, current_round])
            #        break
            prob = 0.9 * prob
            j = 0
            for i in range(len(prob)):
                if legal_moves[i] == 1:
                    prob[i] += noise[j]
                    j += 1
            prob /= np.sum(prob)
            # execute move
            action = np.random.choice(len(prob), p=prob)

            gamecore.execute_move(action)
            player1.update_with_move(action)
            player2.update_with_move(action)

            # next player
            player_index = -player_index
             
            # is ended
            ended, winrate, is_round_ended = gamecore.get_game_status()
            if ended == 1:
                final_winrates = gamecore.get_past_winrates()
                del libtorch
                
                print("episode_step:%d "%episode_step + "".join(str(w)+" " for w in final_winrates))
                if len(final_winrates) != len(self.main_round_order):
                    print("error end info\n")
                    exit(0)
                player1_winrates = []
                for i in range(self.max_round):
                    if self.main_round_order[i] == 0:
                        player1_winrates.append(final_winrates[i])
                    else:
                        player1_winrates.append(-final_winrates[i])
                player1_winrates_sum = player1_winrates
                for i in reversed(range(self.max_round-1)):
                    player1_winrates_sum[i] += player1_winrates_sum[i+1]

                final_train_examples = []
                for example in train_examples:
                    current_step = example[2]
                    current_round = example[3]
                    current_total_step = current_step + current_round*self.step_num_of_round
                    current_main_player = self.main_player_order[current_total_step]
                    current_value = player1_winrates_sum[current_round] if current_main_player == 0 else -player1_winrates_sum[current_round]
                    final_train_examples.append([example[0],example[1],current_value])
                    now_time = time.time()
                    #cache_file_name = "cache/" + str(now_time) + ".pkl"
                #with open(cache_file_name, "wb") as f:
                #    pickle.dump(final_train_examples, f)
                return final_train_examples

    def contest(self, network1, network2, num_contest):
        """compare new and old model
           Args: player1, player2 is neural network
           Return: one_won, two_won, draws
        """
        one_won, two_won, draws = 0, 0, 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_train_threads) as executor:
            futures = [executor.submit(\
                self._contest, network1, network2, 1 if k <= num_contest // 2 else -1) for k in range(1, num_contest + 1)]
            k = 1
            for f in futures:
                winner = f.result()
                if (winner == 1 and k <= num_contest // 2) or (winner == -1 and k > num_contest // 2):
                    one_won += 1
                else:
                    two_won +=1
                k += 1

        return one_won, two_won, draws
    def contest_with_greedy(self):
        num_contest = self.num_contest
        one_winrate, two_winrate = 0.0, 0.0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_train_threads) as executor:
            futures = [executor.submit(\
                self._contest_with_greedy, 1 if k<= num_contest // 2 else -1, ((k-1)%8)) for k in range(1, num_contest + 1)]
            k = 1
            for f in futures:
                winrate = f.result()
                if k<= num_contest //2:
                    one_winrate += (winrate/2+0.5)
                    two_winrate += 1-(winrate/2+0.5)
                else:
                    one_winrate += 1-(winrate/2+0.5)
                    two_winrate += (winrate/2+0.5)
                print(k, k<=num_contest//2, winrate)
                k += 1
        print(one_winrate/num_contest, two_winrate/num_contest)

    def _contest_with_greedy(self, first_player, order):
        # create MCTS
        libtorch = predict_model('./models/checkpoint.pt', self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2, order)
        s_lib = single_lib('./models/checkpoint_single.pt', self.num_mcts_threads, order)
        player1 = MCTS(libtorch, self.num_mcts_threads, self.c_puct,
                    self.num_mcts_sims, self.c_virtual_loss, self.action_size,
                    self.max_round, 0, 0, self.step_num_of_round, self.idx_job_map_c)
        player2 = greedy_player(self.idx_job_map_c, self.idx_job_map)
        player2.idx_job_map_c = self.idx_job_map_c

        # prepare
        players_order = [-1]
        for order in self.main_player_order:
            players_order.append(order+1)
        if first_player == 1:
            players = [None, player1, player2]
        else:
            players = [None, player2, player1]
        player_index = first_player
        gamecore = Gamecore(self.max_round, self.action_size, self.consider_pick_only, s_lib)

        # play
        print("game start, first_player:", first_player)
        episode_step = 0 
        while True:
            episode_step += 1
            start_time = time.time()
            player = players[players_order[episode_step]]

            # select best move
            prob =  list(player.get_action_probs(gamecore, self.temp))
            legal_moves = list(gamecore.get_legal_moves(self.idx_job_map_c))
            for i in range(len(legal_moves)):
                if legal_moves[i] == 0:
                    prob[i] = 0.0
            if episode_step == 1:
                noise = 0.01 * np.random.dirichlet(self.dirichlet_alpha * np.ones(np.count_nonzero(legal_moves)))
                j = 0
                for i in range(len(legal_moves)):
                    if legal_moves[i] == 1:
                        prob[i] += noise[j]
                        j += 1 
            best_move = int(np.argmax(np.array(prob)))
            # execute move
            gamecore.execute_move(best_move)
            #check game status
            ended, winrate, is_round_ended = gamecore.get_game_status()
            end_time = time.time()
            print("player_index:%d"%(players_order[episode_step]) + " select_hero:%d"%(self.idx_hero_map[best_move]) + "process_time: %f" %(end_time - start_time))
            if ended == 1:
                winrates_list = gamecore.get_past_winrates()
                final_winrate = 0.0
                for i in range(self.max_round):
                    if self.main_round_order[i] == 0:
                        final_winrate += winrates_list[i]
                    else:
                        final_winrate += -winrates_list[i]
                final_winrate /= self.max_round
                del libtorch
                del s_lib
                return winrate

            # update search tree
            player1.update_with_move(best_move)
            player2.update_with_move(best_move)
            # next player
            player_index = -player_index
            
    def _contest(self, network1, network2, first_player):
        # create MCTS
        player1 = MCTS(network1, self.num_mcts_threads, self.c_puct,
                    self.num_mcts_sims, self.c_virtual_loss, self.action_size,
                    self.max_round, 0, 0, self.step_num_of_round)
        player2 = MCTS(network2, self.num_mcts_threads, self.c_puct,
                    self.num_mcts_sims, self.c_virtual_loss, self.action_size,
                    self.max_round, 0, 0, self.step_num_of_round)

        # prepare
        players_order = [-1]
        for order in self.main_player_order:
            players_order.append(order+1)
        players = [None, player1, player2]
        player_index = first_player
        gamecore = Gamecore(self.max_round, self.action_size, self.consider_pick_only)

        # play
        while True:
            player = players[players_order[episode_step]]

            # select best move
            prob = player.get_action_probs(gamecore)
            best_move = int(np.argmax(np.array(list(prob))))

            # execute move
            game.execute_move(best_move)

            # check game status
            ended, winner = game.get_game_status()
            if ended == 1:
                return winner

            # update search tree
            player1.update_with_move(best_move)
            player2.update_with_move(best_move)

            # next player
            player_index = -player_index
    
    def contest_with_pure_mcts(self):
        """compare new and old model
           Args: with network, without network
           Return: sum_winrate_for_network, sum_winrate_for_pure_mcts
        """
        num_contest = self.num_contest
        one_winrate, two_winrate = 0.0, 0.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_train_threads) as executor:
            futures = [executor.submit(\
                self._contest_with_pure_mcts, 1 if k <= num_contest // 2 else -1, ((k-1)%8)) for k in range(1, num_contest + 1)]
            k = 1
            for f in futures:
                winrate = f.result()
                if k <= num_contest // 2:
                    one_winrate += (winrate/2+0.5)
                    two_winrate += 1-(winrate/2+0.5)
                else:
                    one_winrate += 1-(winrate/2+0.5)
                    two_winrate += (winrate/2+0.5)
                print(k, k <= num_contest//2, winrate)
                k += 1
        print(one_winrate/num_contest, two_winrate/num_contest)

    def _contest_with_pure_mcts(self, first_player, order):
        # create MCTS
        libtorch = predict_model('./models/checkpoint.pt', self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2, order)
        s_lib = single_lib('./models/checkpoint_single.pt', self.num_mcts_threads, order)
        player1 = MCTS(libtorch, self.num_mcts_threads, self.c_puct,
                    self.num_mcts_sims, self.c_virtual_loss, self.action_size,
                    self.max_round, 0, 0, self.step_num_of_round, self.idx_job_map_c)
        player2 = MCTS_PURE(self.num_mcts_threads, self.c_puct,
                    self.num_mcts_sims, self.c_virtual_loss, self.action_size,
                    self.max_round, 0, 0, self.step_num_of_round, self.idx_job_map_c)

        # prepare
        players_order = [-1]
        for order in self.main_player_order:
            players_order.append(order+1)
        if first_player == 1:
            players = [None, player1, player2]
        else:
            players = [None, player2, player1]
        player_index = first_player
        gamecore = Gamecore(self.max_round, self.action_size, self.consider_pick_only, s_lib)

        # play
        print("game start, first_player:", first_player)
        episode_step = 0
        while True:
            episode_step += 1
            start_time = time.time()
            player = players[players_order[episode_step]]

            # select best move
            prob = list(player.get_action_probs(gamecore, self.temp))
            legal_moves = list(gamecore.get_legal_moves(self.idx_job_map_c))

            for i in range(len(legal_moves)):
                if legal_moves[i] == 0:
                    prob[i] = 0.0

            if episode_step == 1:
                noise = 0.01 * np.random.dirichlet(self.dirichlet_alpha * np.ones(np.count_nonzero(legal_moves)))
                j = 0
                for i in range(len(legal_moves)):
                    if legal_moves[i] == 1:
                        prob[i] += noise[j]
                        j += 1

            best_move = int(np.argmax(np.array(prob)))

            # execute move
            gamecore.execute_move(best_move)
            # check game status
            ended, winrate, is_round_ended = gamecore.get_game_status()
            end_time = time.time()
            print("player_index:%d"%(players_order[episode_step]) + " select_hero:%d"%(self.idx_hero_map[best_move]) + "process_time: %f" %(end_time - start_time))
            if ended == 1:
                winrates_list = gamecore.get_past_winrates()
                final_winrate = 0.0
                for i in range(self.max_round):
                    if self.main_round_order[i] == 0:
                        final_winrate += winrates_list[i]
                    else:
                        final_winrate += -winrates_list[i]
                final_winrate /= self.max_round
                del libtorch
                del s_lib
                return final_winrate

            # update search tree
            player1.update_with_move(best_move)
            player2.update_with_move(best_move)

            # next player
            player_index = -player_index

    def get_symmetries(self, board, pi, last_action):
        # mirror, rotational
        assert(len(pi) == self.action_size)  # 1 for pass

        pi_board = np.reshape(pi, (self.n, self.n))
        last_action_board = np.zeros((self.n, self.n))
        last_action_board[last_action // self.n][last_action % self.n] = 1
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                newAction = np.rot90(last_action_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                    newAction = np.fliplr(last_action_board)
                l += [(newB, newPi.ravel(), np.argmax(newAction) if last_action != -1 else -1)]
        return l

    # this function is forbidden now
    def play_with_human(self, human_first=True, checkpoint_name="best_checkpoint"):
        # load best model
        libtorch_best = NeuralNetwork('./models/best_checkpoint.pt', self.libtorch_use_gpu, 12)
        mcts_best = MCTS(libtorch_best, self.num_mcts_threads * 3, \
             self.c_puct, self.num_mcts_sims * 6, self.c_virtual_loss, self.action_size,
             self.max_round, 0, 0, self.step_num_of_round)

        # create bp game
        gamecore = Gamecore(self.max_round, self.action_size, self.consider_pick_only)

        players = ["alpha", None, "human"]
        player_index = 1 if human_first else -1


        while True:
            player = players[player_index + 1]

            # select move
            if player == "alpha":
                prob = mcts_best.get_action_probs(gamecore)
                best_move = int(np.argmax(np.array(list(prob))))
            else:
                time.sleep(1.0)
                prob = mcts_best.get_action_probs(gamecore)
                best_move = int(np.argmax(np.array(list(prob))))

            # execute move
            gamecore.execute_move(best_move)

            # check game status
            ended, winner = gomoku.get_game_status()
            if ended == 1:
                break

            # update tree search
            mcts_best.update_with_move(best_move)

            # next player
            player_index = -player_index

        print("HUMAN WIN" if ((winner==1 and human_first) or (winner==-1 and not human_first))  else "ALPHA ZERO WIN")

    def load_samples(self, folder="models", filename="checkpoint.example"):
        """load self.examples_buffer
        """

        filepath = path.join(folder, filename)
        with open(filepath, 'rb') as f:
            self.examples_buffer = pickle.load(f)

    def save_samples(self, folder="models", filename="checkpoint.example"):
        """save self.examples_buffer
        """

        if not path.exists(folder):
            mkdir(folder)

        filepath = path.join(folder, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.examples_buffer, f, -1)

    def contest_placeholder(self, player1, player2):
        num_contest = self.num_contest
        one_winrate, two_winrate = 0.0, 0.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_train_threads) as executor:
            futures = [executor.submit(\
                self._contest_placeholder, 1 if k <= num_contest // 2 else -1, ((k-1)%8), player1, player2) for k in range(1, num_contest + 1)]
            k = 1
            for f in futures:
                winrate = f.result()
                if k <= num_contest // 2:
                    one_winrate += (winrate/2+0.5)
                    two_winrate += 1-(winrate/2+0.5)
                else:
                    one_winrate += 1-(winrate/2+0.5)
                    two_winrate += (winrate/2+0.5)
                print(k, k <= num_contest//2, winrate)
                k += 1
        print(one_winrate/num_contest, two_winrate/num_contest)
    
    def player_selecter(self, libtorch, name):
        if name == 'DNNMCTS':
            return MCTS(libtorch, self.num_mcts_threads, self.c_puct,
                    self.num_mcts_sims, self.c_virtual_loss, self.action_size,
                    self.max_round, 0, 0, self.step_num_of_round, self.idx_job_map_c)
        elif name == 'greedy':
            return greedy_player(self.idx_job_map_c, self.idx_job_map)
        elif name == 'pureMCTS':
            return MCTS_PURE(self.num_mcts_threads, self.c_puct,
                    self.num_mcts_sims, self.c_virtual_loss, self.action_size,
                    self.max_round, 0, 0, self.step_num_of_round, self.idx_job_map_c)
        elif name == 'random':
            return random_player(self.idx_job_map_c)
    
    def _contest_placeholder(self, first_player, order, player1, player2):
        # create MCTS
        libtorch = predict_model('./models/checkpoint.pt', self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2, order)
        s_lib = single_lib('./models/checkpoint_single.pt', self.num_mcts_threads, order)
        player1 = self.player_selecter(libtorch, player1)
        player2 = self.player_selecter(libtorch, player2)

        # prepare
        players_order = [-1]
        for order in self.main_player_order:
            players_order.append(order+1)
        if first_player == 1:
            players = [None, player1, player2]
        else:
            players = [None, player2, player1]
        player_index = first_player
        gamecore = Gamecore(self.max_round, self.action_size, self.consider_pick_only, s_lib)

        # play
        print("game start, first_player:", first_player)
        episode_step = 0 
        while True:
            episode_step += 1
            start_time = time.time()
            player = players[players_order[episode_step]]

            # select best move
            prob =  list(player.get_action_probs(gamecore, self.temp))
            legal_moves = list(gamecore.get_legal_moves(self.idx_job_map_c))
            for i in range(len(legal_moves)):
                if legal_moves[i] == 0:
                    prob[i] = 0.0
            if episode_step == 1:
                noise = 0.01 * np.random.dirichlet(self.dirichlet_alpha * np.ones(np.count_nonzero(legal_moves)))
                j = 0
                for i in range(len(legal_moves)):
                    if legal_moves[i] == 1:
                        prob[i] += noise[j]
                        j += 1 
            best_move = int(np.argmax(np.array(prob)))
            # execute move
            gamecore.execute_move(best_move)
            #check game status
            ended, winrate, is_round_ended = gamecore.get_game_status()
            end_time = time.time()
            print("player_index:%d"%(players_order[episode_step]) + " select_hero:%d"%(self.idx_hero_map[best_move]) + "process_time: %f" %(end_time - start_time))
            if ended == 1:
                winrates_list = gamecore.get_past_winrates()
                final_winrate = 0.0
                for i in range(self.max_round):
                    if self.main_round_order[i] == 0:
                        final_winrate += winrates_list[i]
                    else:
                        final_winrate += -winrates_list[i]
                final_winrate /= self.max_round
                del libtorch
                del s_lib
                return winrate

            # update search tree
            player1.update_with_move(best_move)
            player2.update_with_move(best_move)
            # next player
            player_index = -player_index
