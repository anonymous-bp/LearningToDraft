# Learn to draft
A drafting method based on MCTS and nerual network. This code is provied for testing and training on single machine.

## Args
Edit config.py

## Environment

* Python 3.6+
* PyTorch 1.0+
* LibTorch 1.0+
* MSVC14.0/GCC6.0+
* CMake 3.8+
* SWIG 3.0.12+

## Run
```
# Add LibTorch/SWIG to environment variable $PATH

# Compile Python extension
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=path/to/libtorch -DCMAKE_BUILD_TYPE=Release
cmake --build

# Run
cd ../test
python3.6 learner_test.py train # train model
python3.6 leaner_test.py placeholder method1 method2 #play between different methods(DNNMCTS, pureMCTS, greedy, random)
```


## References
1. Silver D, Schrittwieser J, Simonyan K, et al. Mastering the game of go without human knowledge[J]. nature, 2017, 550(7676): 354-359.
2. Silver D, Hubert T, Schrittwieser J, et al. Mastering chess and shogi by self-play with a general reinforcement learning algorithm[J]. arXiv preprint arXiv:1712.01815, 2017.
3. Silver D, Huang A, Maddison C J, et al. Mastering the game of Go with deep neural networks and tree search[J]. nature, 2016, 529(7587): 484-489.
4. Chaslot G M J B, Winands M H M, van Den Herik H J. Parallel monte-carlo tree search[C]//International Conference on Computers and Games. Springer, Berlin, Heidelberg, 2008: 60-71.
5. Mirsoleimani S A, Plaat A, van den Herik H J, et al. An Analysis of Virtual Loss in Parallel MCTS[C]//ICAART (2). 2017: 648-652.
6. Enzenberger M, Müller M. A lock-free multithreaded Monte-Carlo tree search algorithm[C]//Advances in Computer Games. Springer, Berlin, Heidelberg, 2009: 14-20.
7. github.com/hijkzzz/alpha-zero-gomoku
8. github.com/suragnair/alpha-zero-general
