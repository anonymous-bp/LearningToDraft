#pragma once

#include <tuple>
#include <vector>
#include <map>
#include <torch/script.h>
#include <single_lib.h>

const std::vector<int>  MAIN_CAMP_LIST =   {0,1,1,0,0,1,1,0,0,1};//camp1:0, camp2:1
const std::vector<int> MAIN_ROUND_LIST = {0};//player1'camp
const std::vector<int>  MAIN_PLAYER_LIST = {0,1,1,0,0,1,1,0,0,1};//player1:0, player2:1
const std::vector<int>  BAN_OR_PICK_LIST = {1,1,1,1,1,1,1,1,1,1};//ban:0, pick:1
const std::vector<std::string> JOB_LIST = {"shooter", "support", "middle", "jungle", "top"};
const std::map<std::string, int> MAX_HERO_JOB_NUM_MAP = {{"shooter",3}, {"support",2}, {"middle",3}, {"jungle",3}, {"top",3}};
const int ROUND_STEPS = 10;
const int PICK_SIZE = 5;
const int BAN_SIZE = 4;

class Gamecore {
public:
  Gamecore(unsigned int max_rounds, unsigned int max_hero_num, bool consider_pick_only, single_lib *nn);

  std::vector<int> get_legal_moves(std::map<int, std::string>& hero_job_map);
  void execute_move(int move);
  std::vector<int> get_feature();
  std::vector<float> get_game_status();
  float get_simulation_value(std::map<int, std::string>& hero_job_map);
  void print_game_info();
  void randsample(int n, int m, std::vector<int>& sample_idxs);
 // float get_final_winrates();

  inline unsigned int get_hero_num() const { return this->max_hero_num; }
  inline unsigned int get_action_size() const { return this->max_hero_num; }
  inline int get_last_move() const { return this->last_move; }
  inline int get_current_camp() const { return this->current_camp; }
  inline int get_current_round() const { return this->current_round; }
  inline int get_max_round() const {return this->max_rounds;}
  inline int get_current_step() const { return this->current_step; }
  inline int get_current_total_step() const { return (this->current_step + ROUND_STEPS * this->current_round); }
  inline std::vector<float> get_past_winrates() const {return this->past_winrates;}

private:

  unsigned int max_rounds;
  unsigned int max_hero_num;
  bool consider_pick_only;
  unsigned int min_job_num;
  //heroes selected at past rounds.
  std::vector<std::vector<std::vector<int>>> all_camp_selected_heroes;  
  std::vector<std::vector<std::vector<int>>> all_camp_baned_heroes;
  std::vector<float> past_winrates;

  //current round, for state
  std::vector<std::vector<int>> all_camp_ban_list;
  std::vector<int> total_ban_list;
  std::vector<std::vector<int>> all_camp_pick_list;
  std::vector<int> total_pick_list;
  

  int current_round;
  int current_camp;
  int current_step;//for current round, true_step = current_round*main_camp_list.size()+current_step
  int last_move;
  single_lib *nn;
  //std::shared_ptr<torch::jit::script::Module> module;     // torch module for single game winrate predictor
};
