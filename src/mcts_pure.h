#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <random>

#include <gamecore.h>
#include <thread_pool.h>

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator* g)
{
    std::uniform_int_distribution<> dis(0, std::distance(start, end) -1);
    std::advance(start, dis(*g));
    return start;
}
template<typename Iter>
Iter select_randomly(Iter start, Iter end)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, &gen);
}

class TreeNodePure {
 public:
  // friend class can access private variables
  friend class MCTS_PURE;

  TreeNodePure();
  TreeNodePure(const TreeNodePure &node);
  TreeNodePure(TreeNodePure *parent, double p_sa, unsigned int action_size, int max_round, int current_round, int current_step, int step_num_one_round);

  TreeNodePure &operator=(const TreeNodePure &p);
  

  unsigned int select(double c_puct, double c_virtual_loss);
  unsigned int select_pure(double c_puct, double c_virtual_loss);
  void expand(const std::vector<double> &action_priors);
  //void backup(double leaf_value, int leaf_round, std::vector<int>& past_winners);
  void backup(double value);

  void update_winrate(float winrate);
  double get_value(double c_puct, double c_virtual_loss,
                   unsigned int sum_n_visited) const;
  inline bool get_is_leaf() const { return this->is_leaf; }
  inline int get_current_round() const {return this->current_round;} 
  inline int get_current_step() const {return this->current_step;} 
  int max_round;
  int step_num_one_round;
 private:
  // store tree
  TreeNodePure *parent;
  std::vector<TreeNodePure *> children;
  bool is_leaf;
  std::mutex lock;

  std::atomic<int> virtual_loss;
  std::atomic<unsigned int> n_visited;
  double p_sa;
  double q_sa;
  float winrate;
  int action_size;
  //
  int current_round;
  int current_step;
};

class MCTS_PURE{
 public:
  MCTS_PURE(unsigned int thread_num, double c_puct,
       unsigned int num_mcts_sims, double c_virtual_loss, unsigned int action_size,
       int max_round, int current_round, int current_step, int step_num_one_round, std::map<int,std::string>& hero_job_map);
  std::vector<double> get_action_probs(Gamecore *gamecore, double temp = 1e-3);
  void update_with_move(int last_move);

 private:
  void simulate(std::shared_ptr<Gamecore> game);
  static void tree_deleter(TreeNodePure *t);

  double c_puct;
  unsigned int num_mcts_sims;
  double c_virtual_loss;
  unsigned int action_size;
  int max_round;
  int step_num_one_round;
  std::map<int,std::string> hero_job_map;
  
  // variables
  std::unique_ptr<ThreadPool> thread_pool;
  std::unique_ptr<TreeNodePure, decltype(MCTS_PURE::tree_deleter) *> root;

};
