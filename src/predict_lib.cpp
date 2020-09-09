#include <predict_lib.h>

#include <iostream>


// define functions
using namespace std::chrono_literals;

predict_model::predict_model(std::string model_path,
                            bool use_gpu, 
                            unsigned int batch_size,
                            int gpu_selected)
            :use_gpu(use_gpu),
             batch_size(batch_size),
             running(true),
             loop(nullptr),
             gpu_selected(gpu_selected),
             module(std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path.c_str(), torch::Device(torch::DeviceType::CUDA, gpu_selected)))){
    if (this->use_gpu) {
        this->module->to(torch::Device(torch::kCUDA, this->gpu_selected));
    }

    // loop start and create infer thread
    // lambda function here
    this->loop = std::make_unique<std::thread>([this] {
        while (this->running) {
            this->infer();
        }
    });
}

predict_model::~predict_model() {
    this->running = false;
    this->loop->join();
}

// commit interface is for python
std::future<predict_model::return_type> predict_model::commit(Gamecore *gamecore) {
    // process game info for input to network
        
    // get feature from gamecore
    auto game_states = gamecore->get_feature();
    
    // get size from vector (directly calculating will be better)
    //int feature_shape = game_states.size();
    
    // emplace task
    std::promise<return_type> promise;
    auto ans = promise.get_future();
     
    {
    // lock to protect task queue
    std::lock_guard<std::mutex> lock(this->lock);  // lock_guard will auto unlock when destruction
    
    // add tasks in queue
    tasks.emplace(std::make_pair(game_states, std::move(promise)));
    } 
    // wake up infer thread
    this->cv.notify_all(); 
    return ans;
}

// loop running
void predict_model::infer() {
    // create container for states and promises
    std::vector<torch::Tensor> states;
    std::vector<std::promise<return_type> > promises;
    
    bool timeout = false;
    
    // block loop infer until batch is full or wait for 1 ms
    while (states.size() < this->batch_size && !timeout) {
        // lock before add a task
        std::unique_lock<std::mutex> lock(this->lock);
        if (this->cv.wait_for(lock, 1ms, [this] {return this->tasks.size()>0;})) {
            auto task = std::move(this->tasks.front());
            torch::Tensor t = torch::from_blob(task.first.data(), {1, task.first.size()}, torch::dtype(torch::kInt32)).to(torch::Device(torch::kCUDA, this->gpu_selected));
            states.push_back(t);
            promises.emplace_back(std::move(task.second));   
            
            this->tasks.pop();
        } else {
            // this branch means infer thread wait for a too long time.
            timeout = true;
        }
    }

    // if no commit. end function
    if (states.size() == 0) {
        return;
    }
    std::vector<torch::Tensor> new_states;
    for (auto it=states.begin();it!=states.end();it++) {
       new_states.push_back((*it).toType(torch::kInt32));
    }
    
    // infer
    std::vector<torch::jit::IValue> inputs{torch::cat(states, 0).toType(torch::kLong)};
    
    auto result = this->module->forward(inputs).toTuple();
    
    torch::Tensor p_batch = result->elements()[1]
                              .toTensor()
                              .toType(torch::kFloat32)
                              .to(at::kCPU);
    torch::Tensor v_batch =
        result->elements()[0].toTensor().toType(torch::kFloat32).to(at::kCPU);
        
    // set promise to return value for a task
    for (unsigned int i = 0; i < promises.size(); ++i) {
        // dealt with returned value from the network
        torch::Tensor p = p_batch[i];
        torch::Tensor v = v_batch[i];

        std::vector<double> prob(static_cast<float*>(p.data_ptr()), static_cast<float*>(p.data_ptr())+p.size(0));
        std::vector<double> value{v.item<float>()};
        return_type temp{std::move(prob), std::move(value)};
        promises[i].set_value(std::move(temp));
        
    } 
}
