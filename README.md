# Single Round BanPick
A drafting method based on MCTS and nerual network. This code is provied for testing and training on single machine.

## Args
Edit config.py

## Environment

* Python 3.6+
* PyGame 1.9+
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
1. Mastering the Game of Go without Human Knowledge
2. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
3. Parallel Monte-Carlo Tree Search
4. An Analysis of Virtual Loss in Parallel MCTS
5. A Lock-free Multithreaded Monte-Carlo Tree Search Algorithm
6. https://github.com/hijkzzz/alpha-zero-gomoku
7. github.com/suragnair/alpha-zero-general
