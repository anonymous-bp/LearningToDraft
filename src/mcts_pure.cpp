#include <math.h>
#include <float.h>
#include <numeric>
#include <iostream>
#include <chrono>
#include <mcts_pure.h>
//TreeNodePurePure
TreeNodePure::TreeNodePure()
    : 
      max_round(1),
      step_num_one_round(10),
      parent(nullptr),
      is_leaf(true),
      virtual_loss(0),
      n_visited(0),
      p_sa(0),
      q_sa(0),
      winrate(0),
      action_size(0),
      current_round(0),
      current_step(0) {}

TreeNodePure::TreeNodePure(TreeNodePure *parent, double p_sa, unsigned int action_size, int max_round, int current_round, int current_step, int step_num_one_round)
    : 
      max_round(max_round),
      step_num_one_round(step_num_one_round),
      parent(parent),
      children(action_size, nullptr),
      is_leaf(true),
      virtual_loss(0),
      n_visited(0),
      p_sa(p_sa),
      q_sa(0),
      winrate(0),
      action_size(action_size),
      current_round(current_round),
      current_step(current_step){}

TreeNodePure::TreeNodePure(
    const TreeNodePure &node) {  // because automic<>, define copy function
  // struct
  this->parent = node.parent;
  this->children = node.children;
  this->is_leaf = node.is_leaf;

  this->n_visited.store(node.n_visited.load());
  this->p_sa = node.p_sa;
  this->q_sa = node.q_sa;
  this->winrate = node.winrate;

  this->virtual_loss.store(node.virtual_loss.load());
  
  this->action_size = node.action_size;
  this->max_round = node.max_round;
  this->current_round = node.current_round;
  this->current_step = node.current_step;
  this->step_num_one_round = node.step_num_one_round;
}

TreeNodePure &TreeNodePure::operator=(const TreeNodePure &node) {
  if (this == &node) {
    return *this;
  }

  // struct
  this->parent = node.parent;
  this->children = node.children;
  this->is_leaf = node.is_leaf;

  this->n_visited.store(node.n_visited.load());
  this->p_sa = node.p_sa;
  this->q_sa = node.q_sa;
  this->winrate = node.winrate;
  this->virtual_loss.store(node.virtual_loss.load());

  this->action_size = node.action_size;
  this->max_round = node.max_round;
  this->current_round = node.current_round;
  this->current_step = node.current_step;
  this->step_num_one_round = node.step_num_one_round;
  
  return *this;
}
void TreeNodePure::update_winrate(float winrate)
{
    this->winrate = winrate;
}
unsigned int TreeNodePure::select(double c_puct, double c_virtual_loss) {
  unsigned int best_move = 0;
  TreeNodePure *best_node;
  double max_value = -DBL_MAX;
  for (unsigned int i = 0; i < this->children.size(); i++) {
    // empty node
    if (children[i] == nullptr) {
      continue;
    }

    unsigned int sum_n_visited = this->n_visited.load() + 1;
    double cur_value =
        children[i]->get_value(c_puct, c_virtual_loss, sum_n_visited);
    if (cur_value > max_value) {
      max_value = cur_value;
      best_move = i;
      best_node = children[i];
    }
  }

  // add vitural loss
  best_node->virtual_loss++;

  return best_move;
}

unsigned int TreeNodePure::select_pure(double c_puct, double c_virtual_loss) {
  unsigned int best_move = 0;
  TreeNodePure *best_node;
  double max_value = -DBL_MAX;
  std::vector<int> best_move_list;
  std::vector<double> cur_value_list;
  std::vector<double> cur_move_list;
  for (unsigned int i = 0; i < this->children.size(); i++) {
    // empty node
    if (children[i] == nullptr) {
      continue;
    }

    unsigned int sum_n_visited = this->n_visited.load() + 1;
    double cur_value =
        children[i]->get_value(c_puct, c_virtual_loss, sum_n_visited);

    cur_value_list.push_back(cur_value);
    cur_move_list.push_back(i);
    if (cur_value > max_value) {
      max_value = cur_value;
      best_move = i;
      best_node = children[i];
    }
  }
  for (unsigned int i = 0; i < cur_value_list.size(); i++)
  {
    if (std::abs(cur_value_list[i] - max_value) < 1e-6)
    {
        best_move_list.push_back(cur_move_list[i]);
    }
  }
  best_move = *select_randomly(best_move_list.begin(), best_move_list.end());
  best_node = this->children[best_move];
  //std::cout << "best_move_size:" << best_move_list.size() << "best_move:" << best_move << std::endl;
  // add vitural loss
  best_node->virtual_loss++;

  return best_move;
}

void TreeNodePure::expand(const std::vector<double> &action_priors) {
  {
    // get lock
    std::lock_guard<std::mutex> lock(this->lock);

    if (this->is_leaf) {
      unsigned int action_size = this->children.size();
      int max_round = this->max_round;
      int child_current_round = this->current_round;
      int child_current_step = this->current_step;
      int child_step_num_one_round = this->step_num_one_round;
      child_current_step++;
      if (child_current_step == step_num_one_round)
      {
        child_current_step = 0;
        child_current_round++;
      }
      for (unsigned int i = 0; i < action_size; i++) {
        // illegal action
        if (abs(action_priors[i] - 0) < FLT_EPSILON) {
          continue;
        }
        this->children[i] = new TreeNodePure(this, action_priors[i], action_size, max_round, child_current_round, child_current_step, child_step_num_one_round);
      }

      // not leaf
      this->is_leaf = false;
    }
  }
}

void TreeNodePure::backup(double value) {
  // If it is not root, this node's parent should be updated first
  if (this->parent != nullptr) {
    this->parent->backup(value);
  }

  // remove vitural loss
  this->virtual_loss--;

  // update n_visited
  unsigned int n_visited = this->n_visited.load();
  this->n_visited++;
  
  if (this->current_step == 0)
  {
    return;
  }
  // update q_sa
  {
    std::lock_guard<std::mutex> lock(this->lock);

    if(MAIN_CAMP_LIST[this->current_step-1] == 0)
        this->q_sa = (n_visited * this->q_sa + value) / (n_visited + 1);
    else
        this->q_sa = (n_visited * this->q_sa - value) / (n_visited + 1);
  }
}

double TreeNodePure::get_value(double c_puct, double c_virtual_loss,
                           unsigned int sum_n_visited) const {
  // u
  auto n_visited = this->n_visited.load();
  double u = (c_puct * this->p_sa * sqrt(sum_n_visited) / (1 + n_visited));

  // virtual loss
  double virtual_loss = c_virtual_loss * this->virtual_loss.load();
  // int n_visited_with_loss = n_visited - virtual_loss;

  if (n_visited <= 0) {
    return u;
  } else {
        return u + (this->q_sa * n_visited - virtual_loss) / n_visited;
  }
}


// MCTS_PURE
MCTS_PURE::MCTS_PURE(unsigned int thread_num, double c_puct,
           unsigned int num_mcts_sims, double c_virtual_loss, unsigned int action_size,
           int max_round, int current_round, int current_step, int step_num_one_round, std::map<int, std::string>& hero_job_map)
    : c_puct(c_puct),
      num_mcts_sims(num_mcts_sims),
      c_virtual_loss(c_virtual_loss),
      action_size(action_size),
      max_round(max_round),
      step_num_one_round(step_num_one_round),
      hero_job_map(hero_job_map),
      thread_pool(new ThreadPool(thread_num)),
      root(new TreeNodePure(nullptr, 1., action_size, max_round, current_round, current_step, step_num_one_round), MCTS_PURE::tree_deleter){}

void MCTS_PURE::update_with_move(int last_action) {
  auto old_root = this->root.get();

  // reuse the child tree
  if (last_action >= 0 && old_root->children[last_action] != nullptr) {
    // unlink
    TreeNodePure *new_node = old_root->children[last_action];
    old_root->children[last_action] = nullptr;
    new_node->parent = nullptr;

    this->root.reset(new_node);
  } else {
    int new_round = old_root->get_current_round();
    int new_step = old_root->get_current_step()+1;
    if (new_step == this->step_num_one_round)
    {
        new_step = 0;
        new_round++;
    }
    this->root.reset(new TreeNodePure(nullptr, 1., this->action_size, this->max_round, new_round, new_step, this->step_num_one_round));
  }
}

void MCTS_PURE::tree_deleter(TreeNodePure *t) {
  if (t == nullptr) {
    return;
  }

  // remove children
  for (unsigned int i = 0; i < t->children.size(); i++) {
    if (t->children[i]) {
      tree_deleter(t->children[i]);
    }
  }

  // remove self
  delete t;
}

std::vector<double> MCTS_PURE::get_action_probs(Gamecore *gamecore, double temp) { 
  // submit simulate tasks to thread_pool
  std::vector<std::future<void>> futures;
  //auto start_time = std::chrono_literals::system_clock::now();
  for (unsigned int i = 0; i < this->num_mcts_sims; i++) {
    // copy gamecore
    auto game = std::make_shared<Gamecore>(*gamecore);
    /*
    if (i == 0 && game->get_current_step() == 9)
    {
        game->print_game_info();
    }
    */
    auto future =
        this->thread_pool->commit(std::bind(&MCTS_PURE::simulate, this, game));
    // future can't copy
    futures.emplace_back(std::move(future));
  }
  //auto end = std::chrono_literals::system_clock::now();
  //auto duration = std::chrono::duration_cast<milliseconds>(end - start);
  //std::cout<<"commit simulate and copy gamecore time spend: "<<duration.count()<<"ms\n";
  // wait simulate
  for (unsigned int i = 0; i < futures.size(); i++) {
    futures[i].wait();
  }
  
  // calculate probs
  std::vector<double> action_probs(gamecore->get_action_size(), 0);
  const auto &children = this->root->children;

  // greedy
  if (temp - 1e-3 < FLT_EPSILON) {
    unsigned int max_count = 0;
    unsigned int best_action = 0;

    for (unsigned int i = 0; i < children.size(); i++) {
      if (children[i] && children[i]->n_visited.load() > max_count) {
        max_count = children[i]->n_visited.load();
        best_action = i;
      }
    }

    action_probs[best_action] = 1.;
    return action_probs;

  } else {
    // explore
    double sum = 0;
    for (unsigned int i = 0; i < children.size(); i++) {
      if (children[i] && children[i]->n_visited.load() > 0) {
        action_probs[i] = pow(children[i]->n_visited.load(), 1 / temp);
        sum += action_probs[i];
      }
    }

    // renormalization
    std::for_each(action_probs.begin(), action_probs.end(),
                  [sum](double &x) { x /= sum; });
    /*
    if (gamecore->get_current_step() >= 0)
    {
        std::string output_str = std::to_string(gamecore->get_current_step()) + "nan:";
        for (unsigned int i = 0; i < action_probs.size(); i++)
        {
            output_str += std::to_string(action_probs[i]);
            output_str += ";";
        }
        std::cout << output_str << std::endl;
    }
    */
    return action_probs;
  }
}

void MCTS_PURE::simulate(std::shared_ptr<Gamecore> game) {
  // execute one simulation
  auto node = this->root.get();

  while (true) {
    if (node->get_is_leaf()) {
      break;
    }

    // select
    auto action = node->select_pure(this->c_puct, this->c_virtual_loss);
    game->execute_move(action);
    node = node->children[action];
  }

  // get game status
  auto status = game->get_game_status();
  
  double value = 0;

  // not end
  if (status[2] == 0) 
  {
    // predict action_probs and value by neural network
    std::vector<double> action_priors(this->action_size, 0);

    // mask invalid actions
    auto legal_moves = game->get_legal_moves(this->hero_job_map);
    value = game->get_simulation_value(this->hero_job_map);
    
    double sum = 0.0;
    for (unsigned int i = 0; i < legal_moves.size(); i++)
    {
       sum += legal_moves[i];
    }
    for (unsigned int i = 0; i < action_priors.size(); i++) {
      if (legal_moves[i] == 1) {
        action_priors[i] = 1.0;
      } else {
        action_priors[i] = 0;
      }
    }
    
    // renormalization
    if (sum > FLT_EPSILON) {
      std::for_each(action_priors.begin(), action_priors.end(),
                    [sum](double &x) { x /= sum; });
    } else {
      // all masked

      // NB! All valid moves may be masked if either your NNet architecture is
      // insufficient or you've get overfitting or something else. If you have
      // got dozens or hundreds of these messages you should pay attention to
      // your NNet and/or training process.
      std::cout << "All valid moves were masked, do workaround." << std::endl;
      std::string action_priors_str = "";
      std::string mask_str = "";
      for (unsigned int i = 0; i < action_priors.size(); i++)
      {
        action_priors_str += std::to_string(action_priors[i]);
      }
      for (unsigned int i = 0; i < legal_moves.size(); i++)
      {
        mask_str += std::to_string(legal_moves[i]);
      }
      std::cout << "action_priors:" << action_priors_str << "mask_str:" << mask_str << std::endl;
      sum = std::accumulate(legal_moves.begin(), legal_moves.end(), 0);
      for (unsigned int i = 0; i < action_priors.size(); i++) {
        action_priors[i] = legal_moves[i] / sum;
      }
    }

    // expand
    node->expand(action_priors);

  }
  else {
    // end
    value = status[1];
  }
  /*
  if (node->get_current_round() == 0)
  {
    std::cout<<value<<"!!!!!!!!!"<<node->get_current_round()<<'\n';
    std::cout<<node->get_current_round() << " " << node->get_current_step() << "past_winners:" << past_winners.size() << "\n";
    std::cout<<game->get_current_round() << " " << game->get_current_step() << "\n";
    std::cout<<this->root->get_current_round() << " " << this->root->get_current_step() << "\n";
  }
  if (node->get_current_round() == 1)
  {
    std::cout<< "current_visit" <<  node->n_visited.load() << std::endl;
  }
  */

  node->backup(value);
  return;
}
