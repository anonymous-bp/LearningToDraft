#pragma once

#include <torch/script.h>
#include <future>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

class single_lib {
public:
    using return_type = double;

    single_lib(std::string model_path, unsigned int batch_size, int gpu_selected);
    ~single_lib();

    std::future<return_type> commit(std::vector<int> input);
    
    void set_batch_size(unsigned int batch_size) {
        this->batch_size = batch_size;
    }

private:
  using task_type = std::pair<std::vector<int>, std::promise<return_type>>;

  void infer();
  int gpu_selected;
  unsigned int batch_size;

  bool running;
  std::unique_ptr<std::thread> loop;

  std::queue<task_type> tasks;
  std::mutex lock;
  std::condition_variable cv;

  std::shared_ptr<torch::jit::script::Module> module;
};


