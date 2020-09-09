#include <single_lib.h>

#include <iostream>

using namespace std::chrono_literals;

single_lib::single_lib(std::string model_path, unsigned int batch_size, int gpu_selected)
        :batch_size(batch_size),
        running(true),
        loop(nullptr),
        gpu_selected(gpu_selected),
        module(std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path.c_str(), torch::Device(torch::DeviceType::CUDA, gpu_selected)))) {
    this->loop = std::make_unique<std::thread>([this] {
        while (this->running) {
            this->infer();
        }
    });      
}

single_lib::~single_lib() {
    this->running = false;
    this->loop->join();
}

std::future<single_lib::return_type> single_lib::commit(std::vector<int> input_feature) {
    std::promise<return_type> promise;
    std::vector<int> inputs = input_feature;
    auto ans = promise.get_future();
    
    {
        std::lock_guard<std::mutex> lock(this->lock);
        tasks.emplace(std::make_pair(inputs, std::move(promise)));
    }
    
    this->cv.notify_all();
    return ans;
} 

void single_lib::infer() {
    std::vector<torch::Tensor> states;
    std::vector<std::promise<return_type> > promises;

    bool timeout = false;

    while (states.size() < this->batch_size && !timeout) {
        std::unique_lock<std::mutex> lock(this->lock);
        if (this->cv.wait_for(lock, 1ms, [this]{return this->tasks.size()>0;})) {
            auto task = std::move(this->tasks.front());
            torch::Tensor t = torch::from_blob(task.first.data(), {1, task.first.size()}, torch::dtype(torch::kInt32)).to(torch::Device(torch::kCUDA, this->gpu_selected));
            states.push_back(t);
            promises.emplace_back(std::move(task.second));

            this->tasks.pop();
        } 
       else {
        timeout = true;
       }
    }

   if (states.size() == 0) {
       return;
   }

   std::vector<torch::Tensor> new_states;
   for (auto it=states.begin(); it!=states.end(); it++) {
    new_states.push_back((*it).toType(torch::kInt32));
   }

   std::vector<torch::jit::IValue> inputs{torch::cat(states, 0).toType(torch::kLong)};
   auto result = this->module->forward(inputs);
   torch::Tensor winrate_batch = result.toTensor().toType(torch::kFloat32).to(at::kCPU);
   std::vector<float> winrates;
   for (unsigned int i=0; i<promises.size(); ++i) {
        float tmp_winrate = winrate_batch[i][0].item<float>();
        promises[i].set_value(std::move(tmp_winrate));
   }
}
