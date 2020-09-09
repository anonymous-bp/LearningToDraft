# coding: utf-8
import sys
sys.path.append('..')
sys.path.append('../src')

import learner
import config

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ["placeholder","train", "play", "test_mcts", "test_greedy"]:
        print("[USAGE] python leaner_test.py train|play|test_mcts")
        exit(1)

    alpha_zero = learner.Leaner(config.config)

    if sys.argv[1] == "train":
        alpha_zero.learn()
    elif sys.argv[1] == "play":
        for i in range(10):
            print("GAME: {}".format(i + 1))
            alpha_zero.play_with_human(human_first=i % 2)
    elif sys.argv[1] == "test_greedy":
        for i in range(1):
            print("GAME: {}".format(i + 1))
            alpha_zero.contest_with_greedy()
    elif sys.argv[1] == "test_mcts":
        for i in range(1):
            print("GAME: {}".format(i + 1))
            alpha_zero.contest_with_pure_mcts()
    elif sys.argv[1] == "placeholder":
        player1 = sys.argv[2]
        player2 = sys.argv[3]
        print("player1: ", player1, "player2: ", player2)
        for i in range(1):
            print("GAME: {}".format(i+1))
            alpha_zero.contest_placeholder(player1, player2)
