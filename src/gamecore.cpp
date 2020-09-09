#include <math.h>
#include <iostream>

#include <gamecore.h>
#include <cstdlib>
#include <time.h>
#include <unistd.h>
#include <single_lib.h>
Gamecore::Gamecore(unsigned int max_rounds, unsigned int max_hero_num, bool consider_pick_only, single_lib *nn)
    : max_rounds(max_rounds), 
      max_hero_num(max_hero_num), 
      consider_pick_only(consider_pick_only),
      all_camp_selected_heroes(max_rounds, std::vector<std::vector<int>>(2, std::vector<int>())),
      all_camp_baned_heroes(max_rounds, std::vector<std::vector<int>>(2, std::vector<int>())),
      all_camp_ban_list(2, std::vector<int>()),
      all_camp_pick_list(2, std::vector<int>()),
      nn(nn){
      //module(std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path.c_str()))){
     // module(nullptr){
  //this->module->to(at::kCUDA);
  this->current_round = 0;
  this->current_camp = 0;
  this->current_step = 0;
  this->last_move = -1;
}
void Gamecore::print_game_info()
{
    std::string output_str = "";
    output_str += "camp:" + std::to_string(this->get_current_camp());
    output_str += " last_move:" + std::to_string(this->get_last_move());
    output_str += "select:";
    for (unsigned int ii = 0; ii < 2; ii++)
    {
        for (unsigned int jj = 0; jj < this->all_camp_pick_list[ii].size(); jj++)
        {
            output_str += std::to_string(this->all_camp_pick_list[ii][jj]) + ";";
        }
    }
    std::cout << output_str << std::endl;
}

float Gamecore::get_simulation_value(std::map<int,std::string>& hero_job_map)
{
 //seed
 struct timespec ts;
 clock_gettime(CLOCK_MONOTONIC, &ts);
 srand((unsigned)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000));

  std::vector<int> camp1_pick_list((this->all_camp_pick_list[0]).begin(), (this->all_camp_pick_list[0]).end());
  std::vector<int> camp2_pick_list((this->all_camp_pick_list[1]).begin(), (this->all_camp_pick_list[1]).end());
  auto max_hero_num = this->get_hero_num();
  auto current_round = this->get_current_round();
  std::vector<int> legal_moves1(max_hero_num, 1);
  std::vector<int> legal_moves2(max_hero_num, 1);
  std::vector<int> left_hero_pool_camp1;
  std::vector<int> left_hero_pool_camp2;
  //get illegal actions
  if (!consider_pick_only)
  {
    for(uint32_t i = 0; i < total_ban_list.size(); i++)
    {
        legal_moves1[total_ban_list[i]] = 0;
        legal_moves2[total_ban_list[i]] = 0;
    }
  }
  
  for(uint32_t i = 0; i < total_pick_list.size(); i++)
  {
    legal_moves1[total_pick_list[i]] = 0;
    legal_moves2[total_pick_list[i]] = 0;
  }
  for(uint32_t i = 0; i < current_round; i++)
  {
    for(uint32_t j = 0; j < all_camp_selected_heroes[i][0].size(); j++)
    {
        if (MAIN_ROUND_LIST[current_round] == 0)
        {
            legal_moves1[all_camp_selected_heroes[i][0][j]] = 0;
        }
        else
        {
            legal_moves2[all_camp_selected_heroes[i][0][j]] = 0;
        }
    }
    for(uint32_t j = 0; j < all_camp_selected_heroes[i][1].size(); j++)
    {
        if (MAIN_ROUND_LIST[current_round] == 1)
        {
            legal_moves1[all_camp_selected_heroes[i][1][j]] = 0;
        }
        else
        {
            legal_moves2[all_camp_selected_heroes[i][1][j]] = 0;
        }
    }
  }

  //lubandashi daqiao yuange illegal
  legal_moves1[19] = 0;
  legal_moves1[69] = 0;
  legal_moves1[95] = 0;
  
  legal_moves2[19] = 0;
  legal_moves2[69] = 0;
  legal_moves2[95] = 0;


  for (uint32_t i = 0; i < legal_moves1.size(); i++)
  {
    if (legal_moves1[i])
    {
        left_hero_pool_camp1.push_back(i);
    }
  }
  for (uint32_t i = 0; i < legal_moves2.size(); i++)
  {
    if (legal_moves2[i])
    {
        left_hero_pool_camp2.push_back(i);
    }
  }
  int rz = 0; 
  while(rz++ < 500 && camp1_pick_list.size() < 5)
  {
    std::vector<int> camp1_sample_idxs;
    randsample(left_hero_pool_camp1.size(), 5-camp1_pick_list.size(), camp1_sample_idxs);
    std::vector<int> tmp_left_camp1;
    bool is_legal_camp = true;
    for (uint32_t i = 0; i < camp1_sample_idxs.size(); i++)
    {
        tmp_left_camp1.push_back(left_hero_pool_camp1[camp1_sample_idxs[i]]);
    }
    std::vector<int> tmp_camp1(tmp_left_camp1.begin(), tmp_left_camp1.end());
    tmp_camp1.insert(tmp_camp1.end(), camp1_pick_list.begin(), camp1_pick_list.end());
    std::map<std::string,int> current_job_num_map = {{"top",0},{"middle",0},{"shooter",0},{"support",0},{"jungle",0}};
    
    for(uint32_t i = 0; i < tmp_camp1.size(); i++)
    {
        current_job_num_map[hero_job_map[tmp_camp1[i]]] ++;
    }
    for (uint32_t i = 0; i < tmp_camp1.size(); i++)
    {
        if (current_job_num_map[hero_job_map[tmp_camp1[i]]] > MAX_HERO_JOB_NUM_MAP.at(hero_job_map[tmp_camp1[i]]))
        {
           is_legal_camp = false;
           break;
        }
    }
    if (is_legal_camp || rz >= 499)
    {   
        //std::cout << "tmp_left_camp1_size:" << tmp_left_camp1.size() << "camp1_sample_idxs_size:" << camp1_sample_idxs.size() << "left_hero_pool_camp1" << left_hero_pool_camp1.size() << "camp1_pick_list" <<camp1_pick_list.size()  << std::endl;
        camp1_pick_list.insert(camp1_pick_list.end(), tmp_left_camp1.begin(), tmp_left_camp1.end());
        //std::cout << "camp1_size:" << camp1_pick_list.size() << std::endl;
        break;
    }
  }
  rz = 0;
  while(rz++ < 500 && camp2_pick_list.size() < 5)
  {
    std::vector<int> camp2_sample_idxs;
    randsample(left_hero_pool_camp2.size(), 5-camp2_pick_list.size(), camp2_sample_idxs);
    std::vector<int> tmp_left_camp2;
    bool is_legal_camp = true;
    for (uint32_t i = 0; i < camp2_sample_idxs.size(); i++)
    {
        tmp_left_camp2.push_back(left_hero_pool_camp2[camp2_sample_idxs[i]]);
    }
    std::vector<int> tmp_camp2(tmp_left_camp2.begin(), tmp_left_camp2.end());
    tmp_camp2.insert(tmp_camp2.end(), camp2_pick_list.begin(), camp2_pick_list.end());
    std::map<std::string,int> current_job_num_map = {{"top",0},{"middle",0},{"shooter",0},{"support",0},{"jungle",0}};
    
    for(uint32_t i = 0; i < tmp_camp2.size(); i++)
    {
        current_job_num_map[hero_job_map[tmp_camp2[i]]] ++;
    }
    for (uint32_t i = 0; i < tmp_camp2.size(); i++)
    {
        if (current_job_num_map[hero_job_map[tmp_camp2[i]]] > MAX_HERO_JOB_NUM_MAP.at(hero_job_map[tmp_camp2[i]]))
        {
           is_legal_camp = false;
           break;
        }
    }
    if (is_legal_camp || rz >= 499)
    {   
        //std::cout << "tmp_left_camp2_size:" << tmp_left_camp2.size() << "camp2_sample_idxs_size:" << camp2_sample_idxs.size() << "left_hero_pool_camp2" << left_hero_pool_camp2.size() << "camp2_pick_list" <<camp2_pick_list.size()  << std::endl;
        camp2_pick_list.insert(camp2_pick_list.end(), tmp_left_camp2.begin(), tmp_left_camp2.end());
        //std::cout << "camp2_size:" << camp2_pick_list.size() << std::endl;
        break;
    }
  }
  std::sort(camp1_pick_list.begin(), camp1_pick_list.end());
  std::sort(camp2_pick_list.begin(), camp2_pick_list.end());
  camp1_pick_list.insert(camp1_pick_list.end(), camp2_pick_list.begin(), camp2_pick_list.end());
  if (camp1_pick_list.size() != 10)
  {
    std::cout <<"error size:" << camp1_pick_list.size() << std::endl;
  }
  auto future = this->nn->commit(camp1_pick_list);
  auto result = future.get();
  float winrate = result;
  return (winrate-0.5)*2;
}

std::vector<int> Gamecore::get_legal_moves(std::map<int,std::string>& hero_job_map) {
  auto max_hero_num = this->get_hero_num();
  auto current_round = this->get_current_round();
  auto current_camp = this->get_current_camp();
  auto current_total_step = this->current_step + this->current_round*ROUND_STEPS;
  auto current_player = MAIN_PLAYER_LIST[current_total_step];
  std::vector<int> legal_moves(max_hero_num, 1);

  //lubandashi daqiao yuange illegal
  legal_moves[19] = 0;
  legal_moves[69] = 0;
  legal_moves[95] = 0;

 
  //get illegal actions
  if (!consider_pick_only)
  {
    for(uint32_t i = 0; i < total_ban_list.size(); i++)
    {
        legal_moves[total_ban_list[i]] = 0;
    }
  }
  
  for(uint32_t i = 0; i < total_pick_list.size(); i++)
  {
    legal_moves[total_pick_list[i]] = 0;
  }
  
  for(uint32_t i = 0; i < current_round; i++)
  {
    for(uint32_t j = 0; j < all_camp_selected_heroes[i][current_player].size(); j++)
    {
        legal_moves[all_camp_selected_heroes[i][current_player][j]] = 0;
    }
  }
  
  //delete some camps by job
  std::map<std::string,int> current_job_num_map = {{"top",0},{"middle",0},{"shooter",0},{"support",0},{"jungle",0}};
  
  for(uint32_t i = 0; i < all_camp_pick_list[current_camp].size(); i++)
  {
    current_job_num_map[hero_job_map[all_camp_pick_list[current_camp][i]]] ++;
  }
  for (uint32_t i = 0; i < legal_moves.size(); i++)
  {
    if (!legal_moves[i]) continue;
    if (current_job_num_map[hero_job_map[i]] >= MAX_HERO_JOB_NUM_MAP.at(hero_job_map[i]))
    {
        legal_moves[i] = 0;
    }
  }
  
  return legal_moves;
}


void Gamecore::execute_move(int move) {
  auto current_camp = this->get_current_camp();
  auto current_step = this->get_current_step();
  
  //if move is illegal_action
  /*
  if (false) {
    throw std::runtime_error("execute_move illgeal!");
  }
  */

  //update status
  if(consider_pick_only)
  {
    this->all_camp_pick_list[current_camp].push_back(move);
    this->total_pick_list.push_back(move);
  }
  else
  {
    int ban_or_pick = BAN_OR_PICK_LIST[current_step];
    if(ban_or_pick)
    {
        this->all_camp_pick_list[current_camp].push_back(move); 
        this->total_pick_list.push_back(move);
    }
    else
    {
        this->all_camp_ban_list[current_camp].push_back(move); 
        this->total_ban_list.push_back(move);
    }
  }
  
  
  //update infos
  this->last_move = move;
  this->current_step++;
  if (this->current_step == MAIN_CAMP_LIST.size())
  {
    all_camp_baned_heroes[this->current_round][0].insert(all_camp_baned_heroes[this->current_round][0].begin(), this->all_camp_ban_list[0].begin(), this->all_camp_ban_list[0].end());
    all_camp_baned_heroes[this->current_round][1].insert(all_camp_baned_heroes[this->current_round][1].begin(), this->all_camp_ban_list[1].begin(), this->all_camp_ban_list[1].end());
    std::sort(all_camp_baned_heroes[this->current_round][0].begin(), all_camp_baned_heroes[this->current_round][0].end());
    std::sort(all_camp_baned_heroes[this->current_round][1].begin(), all_camp_baned_heroes[this->current_round][1].end());

    if (MAIN_ROUND_LIST[this->current_round] == 0)
    {
        all_camp_selected_heroes[this->current_round][0].insert(all_camp_selected_heroes[this->current_round][0].end(), this->all_camp_pick_list[0].begin(), this->all_camp_pick_list[0].end());
        all_camp_selected_heroes[this->current_round][1].insert(all_camp_selected_heroes[this->current_round][1].end(), this->all_camp_pick_list[1].begin(), this->all_camp_pick_list[1].end());
        std::sort(all_camp_selected_heroes[this->current_round][0].begin(), all_camp_selected_heroes[this->current_round][0].end());
        std::sort(all_camp_selected_heroes[this->current_round][1].begin(), all_camp_selected_heroes[this->current_round][1].end());
    }
    else
    {
        all_camp_selected_heroes[this->current_round][1].insert(all_camp_selected_heroes[this->current_round][1].end(), this->all_camp_pick_list[0].begin(), this->all_camp_pick_list[0].end());
        all_camp_selected_heroes[this->current_round][0].insert(all_camp_selected_heroes[this->current_round][0].end(), this->all_camp_pick_list[1].begin(), this->all_camp_pick_list[1].end());
        std::sort(all_camp_selected_heroes[this->current_round][0].begin(), all_camp_selected_heroes[this->current_round][0].end());
        std::sort(all_camp_selected_heroes[this->current_round][1].begin(), all_camp_selected_heroes[this->current_round][1].end());
    }
    this->all_camp_ban_list[0].clear();
    this->all_camp_ban_list[1].clear();
    this->total_ban_list.clear();
    
    this->all_camp_pick_list[0].clear();
    this->all_camp_pick_list[1].clear();
    this->total_pick_list.clear();
    

    this->current_step = 0;
    this->current_round ++;
  }
  this->current_camp = MAIN_CAMP_LIST[this->current_step];

}
  
std::vector<float> Gamecore::get_game_status() {
  // return (is ended, winrate)
  float is_end = 0.0;
  float value = 0.0;
  float is_round_end = 0.0;
  if(this->current_round >= this->max_rounds && this->current_step == 0)
  {
    is_end = 1.0;
  }
  //predict winrate

  if (this->current_step == 0 && this->current_round>0)
  {
    is_round_end = 1.0;

    std::vector<int> predict_feature;
    std::vector<int> camp1_feature;
    std::vector<int> camp2_feature;
    for(uint32_t i = 0; i < all_camp_selected_heroes[this->current_round-1][0].size(); i++)
    {   
        if (MAIN_ROUND_LIST[this->current_round-1] == 0)
            camp1_feature.push_back(all_camp_selected_heroes[this->current_round-1][0][i]);
        else
            camp2_feature.push_back(all_camp_selected_heroes[this->current_round-1][0][i]);
    }
    for(uint32_t i = 0; i < all_camp_selected_heroes[this->current_round-1][1].size(); i++)
    {
        if (MAIN_ROUND_LIST[this->current_round-1] == 0)
            camp2_feature.push_back(all_camp_selected_heroes[this->current_round-1][1][i]);
        else
            camp1_feature.push_back(all_camp_selected_heroes[this->current_round-1][1][i]);
    }
    std::sort(camp1_feature.begin(), camp1_feature.end());
    std::sort(camp2_feature.begin(), camp2_feature.end());
    predict_feature.insert(predict_feature.end(), camp1_feature.begin(), camp1_feature.end());
    predict_feature.insert(predict_feature.end(), camp2_feature.begin(), camp2_feature.end());
    // predict here
    //torch::Tensor input_tensor = torch::from_blob(&predict_feature[0], {1, predict_feature.size()}, torch::dtype(torch::kInt32)).cuda();
    //std::vector<torch::jit::IValue> inputs {input_tensor.toType(torch::kLong).to(at::kCUDA)};
    //auto result = this->module->forward(inputs);
    // convert to tensor
    //torch::Tensor wr = result.toTensor()
    //                        .toType(torch::kFloat32)
    //                        .to(at::kCPU);
    //std::srand((unsigned)std::time(NULL));
    //float random_number = (float) std::rand()/RAND_MAX;
    //float winrate = wr.item<float>();
    //float winrate = 0.5;
    auto future = this->nn->commit(predict_feature);
    auto result = future.get();
    /*
    //log here
    std::string output_str = "";
    for (uint32_t i = 0; i < predict_feature.size(); i++)
    {
        output_str += std::to_string(predict_feature[i]);
        output_str += ",";
    }
    output_str += std::to_string(result);
    std::cout << "camp_feature:" << output_str << std::endl;
    */
    float winrate = (result-0.5)*2;
    /*
    int winlose;
    if (winrate >= random_number) {
        winlose = 1;
    }
    else {
        winlose = -1;
    }
    past_winners.push_back(winlose);
    value = static_cast<float>(winlose); 
    */
    past_winrates.push_back(winrate);
    value = winrate;
  }
  return {is_end, value, is_round_end};
}
/*
float Gamecore::get_final_winrates() {
  if(!(this->current_round >= this->max_rounds && this->current_step == 0))
  {
    std::cout << "not end error!\n";
  }
  //predict winrate
  std::vector<int> predict_feature;
  std::vector<int> camp1_feature;
  std::vector<int> camp2_feature;
  int r = 0;
  for(uint32_t i = 0; i < all_camp_selected_heroes[r][0].size(); i++)
  {
    camp1_feature.push_back(all_camp_selected_heroes[r][0][i]);
  }
  for(uint32_t i = 0; i < all_camp_selected_heroes[r][1].size(); i++)
  {
    camp2_feature.push_back(all_camp_selected_heroes[r][1][i]);
  }
  std::sort(camp1_feature.begin(), camp1_feature.end());
  std::sort(camp2_feature.begin(), camp2_feature.end());
  predict_feature.insert(predict_feature.end(), camp1_feature.begin(), camp1_feature.end());
  predict_feature.insert(predict_feature.end(), camp2_feature.begin(), camp2_feature.end());
  // predict here
  torch::Tensor input_tensor = torch::from_blob(&predict_feature[0], {1, predict_feature.size()}, torch::dtype(torch::kInt32)).cuda();
  std::vector<torch::jit::IValue> inputs {input_tensor.toType(torch::kLong)};
  auto result = this->module->forward(inputs);
  torch::Tensor wr = result.toTensor()
                           .toType(torch::kFloat32)
                           .to(at::kCPU);
  float winrate = wr.item<float>();
  return winrate;
}
*/

std::vector<int> Gamecore::get_feature() {
  //whether need to treat current camp as feature
  auto max_hero_num = this->get_hero_num();
  int current_total_step = this->current_step + this->current_round*ROUND_STEPS;
  std::vector<int> features;
  //current round featue
  {
    
    if (!consider_pick_only)
    {
        std::vector<int> camp1_feature(BAN_SIZE, max_hero_num);
        std::vector<int> camp2_feature(BAN_SIZE, max_hero_num);
        for (uint32_t i = 0; i < this->all_camp_ban_list[0].size(); i++)
        {
            camp1_feature[i] = this->all_camp_ban_list[0][i];
        }
        for (uint32_t i = 0; i < this->all_camp_ban_list[1].size(); i++)
        {
            camp2_feature[i] = this->all_camp_ban_list[1][i];
        }
        std::sort(camp1_feature.begin(), camp1_feature.end());
        std::sort(camp2_feature.begin(), camp2_feature.end());
        if (MAIN_PLAYER_LIST[current_total_step] == 0)
        {
            features.insert(features.end(), camp1_feature.begin(), camp1_feature.end());
            features.insert(features.end(), camp2_feature.begin(), camp2_feature.end());
        }
        else
        {
            features.insert(features.end(), camp2_feature.begin(), camp2_feature.end());
            features.insert(features.end(), camp1_feature.begin(), camp1_feature.end());
        }
    }
    {
        std::vector<int> camp1_feature(PICK_SIZE, max_hero_num);
        std::vector<int> camp2_feature(PICK_SIZE, max_hero_num);
        for (uint32_t i = 0; i < this->all_camp_pick_list[0].size(); i++)
        {
            camp1_feature[i] = this->all_camp_pick_list[0][i];
        }
        for (uint32_t i = 0; i < this->all_camp_pick_list[1].size(); i++)
        {
            camp2_feature[i] = this->all_camp_pick_list[1][i];
        }
        std::sort(camp1_feature.begin(), camp1_feature.end());
        std::sort(camp2_feature.begin(), camp2_feature.end());
        if (MAIN_PLAYER_LIST[current_total_step] == 0)
        {
            features.insert(features.end(), camp1_feature.begin(), camp1_feature.end());
            features.insert(features.end(), camp2_feature.begin(), camp2_feature.end());
        }
        else
        {
            features.insert(features.end(), camp2_feature.begin(), camp2_feature.end());
            features.insert(features.end(), camp1_feature.begin(), camp1_feature.end());
        }
    }
    
  }
  /*
  //ban feature
  if (!consider_pick_only)
  {
    std::vector<int> ban_feature;
    int ban_feature_size = this->max_rounds * BAN_SIZE * 2;
    for(uint32_t i = 0; i < this->current_round; i++)
    {
        for(uint32_t j = 0; j < 2; j++)
        {
            for(uint32_t k = 0; k < this->all_camp_baned_heroes[i][j].size(); k++)
            {
                ban_feature.push_back(this->all_camp_baned_heroes[i][j][k]);
            }
        }
    }
    std::vector<int> camp1_feature(BAN_SIZE, max_hero_num);
    std::vector<int> camp2_feature(BAN_SIZE, max_hero_num);
    for (uint32_t i = 0; i < this->all_camp_ban_list[0].size(); i++)
    {
        camp1_feature[i] = this->all_camp_ban_list[0][i];
    }
    for (uint32_t i = 0; i < this->all_camp_ban_list[1].size(); i++)
    {
        camp2_feature[i] = this->all_camp_ban_list[1][i];
    }
    ban_feature.insert(ban_feature.end(), camp1_feature.begin(), camp1_feature.end());
    ban_feature.insert(ban_feature.end(), camp2_feature.begin(), camp2_feature.end());
    int left_feature_size = ban_feature_size - ban_feature.size();
    for(uint32_t i = 0; i < left_feature_size; i++)
    {
        ban_feature.push_back(max_hero_num);
    }
    features.insert(features.end(), ban_feature.begin(), ban_feature.end());
  }

  //pick feature
  {
    std::vector<int> pick_feature;
    int pick_feature_size = this->max_rounds * PICK_SIZE * 2;
    for(uint32_t i = 0; i < this->current_round; i++)
    {
        if (MAIN_PLAYER_LIST[current_total_step] == 0)
        {
            pick_feature.insert(pick_feature.end(), (this->all_camp_selected_heroes[i][0]).begin(),(this->all_camp_selected_heroes[i][0]).end());
            pick_feature.insert(pick_feature.end(), (this->all_camp_selected_heroes[i][1]).begin(),(this->all_camp_selected_heroes[i][1]).end());
        }
        else
        {
            pick_feature.insert(pick_feature.end(), (this->all_camp_selected_heroes[i][1]).begin(),(this->all_camp_selected_heroes[i][1]).end());
            pick_feature.insert(pick_feature.end(), (this->all_camp_selected_heroes[i][0]).begin(),(this->all_camp_selected_heroes[i][0]).end());
        }
    }
    std::vector<int> camp1_feature(PICK_SIZE, max_hero_num);
    std::vector<int> camp2_feature(PICK_SIZE, max_hero_num);
    for (uint32_t i = 0; i < this->all_camp_pick_list[0].size(); i++)
    {
        camp1_feature[i] = this->all_camp_pick_list[0][i];
    }
    for (uint32_t i = 0; i < this->all_camp_pick_list[1].size(); i++)
    {
        camp2_feature[i] = this->all_camp_pick_list[1][i];
    }
    pick_feature.insert(pick_feature.end(), camp1_feature.begin(), camp1_feature.end());
    pick_feature.insert(pick_feature.end(), camp2_feature.begin(), camp2_feature.end());
    int left_feature_size = pick_feature_size - pick_feature.size();
    for(uint32_t i = 0; i < left_feature_size; i++)
    {
        pick_feature.push_back(max_hero_num);
    }
    features.insert(features.end(), pick_feature.begin(), pick_feature.end());
  }
  */

  //current round, current_camp, current_player
  {
    //current_round
    /*
    int round_feature_size = this->max_rounds*1;
    std::vector<int> round_feature(round_feature_size, 0);
    round_feature[this->current_round] = 1;
    features.insert(features.end(), round_feature.begin(), round_feature.end());
    
    //current_camp
    std::vector<int> camp_feature(2, 0);
    camp_feature[this->current_camp] = 1;
    features.insert(features.end(), camp_feature.begin(), camp_feature.end());
    */

    //current_player
    std::vector<int> player_feature(2,0);
    player_feature[MAIN_PLAYER_LIST[current_total_step]] = 1;
    features.insert(features.end(), player_feature.begin(), player_feature.end());
    
  }
    
  /*
  //winrate feature, unkonwn:0
  {
    int winlose_feature_size = this->max_rounds * 1;
    std::vector<int> winlose_feature(winlose_feature_size, 0);
    for (uint32_t i = 0; i < this->past_winners.size(); i++)
    {
        winlose_feature[i] = this->past_winners[i];
    }
    features.insert(features.end(), winlose_feature.begin(), winlose_feature.end());
  }
  */
  return features;
}
void Gamecore::randsample(int n, int m, std::vector<int>& sample_idxs)
{
    for(int i = 0; i < n; ++i)
    {
        if(rand() % (n-i) < m)
        {
            sample_idxs.push_back(i);
            --m;
        }
    }
}
