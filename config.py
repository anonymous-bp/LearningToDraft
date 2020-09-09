config = {
    # banpick settings
    'ban_pick_order': 'b1-b1-b1-b1-p1-p2-p2-p1-b1-b1-b1-b1-p1-p2-p1',   # ban_pick order
    'main_player_order': [0,1,1,0,0,1,1,0,0,1], #player1:0, player2:1
    'main_round_order': [0],                                         #player1'camp
    'game_number': 1,                                                   # best of game_number
    'reuse_hero_allowed': False,                                        # if allowed to reuse hero in the series of games
    'final_game_setting': False,                                        # if all heroes can be used in the final game
    'hero_numbers': 98,                                                 # hero numbers
    'hero_job_file': '../hero_job.txt.all',
    'step_num_of_round': 10,
    # mcts
    'libtorch_use_gpu': True,                                           # libtorch use cuda
    'num_mcts_threads': 56,                                             # mcts threads number
    'num_mcts_sims': 6400,                                              # mcts simulation times
    'c_puct': 5,                                                        # puct coeff
    'c_virtual_loss': 3,                                                # virtual loss coeff

    # neural networks                                                   # use FC for base model
    'batch_size': 512,
    'epochs': 2,
    'train_use_gpu': True,                                              # train NN using cuda
    'embedding_size': 32,                                               # from hero id to embedding dims
    'activation_function': 'relu',                                      # activation function choice: [relu, leaky_relu, tanh]
    'optimizer': 'adam',                                                # optimizer
    'pick_block_dim': [64, 32],                                         # pick_block_dim
    'ban_block_dim': [64, 32],                                          # ban_block_dim
    'current_block_dim': [64, 32],
    'hidden_block_dim': [128],                                          # hidden block after concat
    'policy_output_dim': [128],                                         # policy output block
    'value_output_dim': [128],                                          # value output block
    'ban_considered': False,                                            # consider ban in predictor?
    'consider_pick_only': True,

    # network training parameters
    'learning_rate': 1e-3,                                              # learning rate
    'l2': 1e-4,                                                         # l2 penalty
    'training_epoches': 5,                                              # training epoches
    'training_batch_size': 1024,                                        # training batch_size

    # alphazero algorithm related parameters
    'num_iters': 10000,                                                 # train iterations
    'num_eps': 84,                                                      # self play times in per iter
    'num_train_threads': 28,                                            # self play in parallel
    'num_explore': 5,                                                   # explore step in a game
    'temp': 1,                                                          # temperature
    'dirichlet_alpha': 0.3,                                             # action noise in self play games
    'update_threshold': 0.55,                                           # update model threshold
    'num_contest': 100,                                                  # new/old model compare times
    'check_freq': 20,                                                   # test model frequency
    'examples_buffer_max_len': 20,                                      # max length of examples buffer

}
