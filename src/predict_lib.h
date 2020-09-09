#pragma once

#include <torch/script.h>

#include <future>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <gamecore.h>

class predict_model {
public:
    // return type: policy or value
    using return_type = std::vector<std::vector<double> >;
    
    // constructor function of predict model
    predict_model(std::string model_path, bool use_gpu, unsigned int batch_size, int gpu_selected);
    // destructor function of predict model
    ~predict_model();

    // commit task to queue
    std::future<return_type> commit(Gamecore *gamecore);
    // set batch size of input and result
    void set_batch_size(unsigned int batch_size) {
        this->batch_size = batch_size;
    }

private:
    // define task type for mempool
    using task_type = std::pair<std::vector<int>, std::promise<return_type>>;

    // key function infer, with offerred model
    void infer();
    int gpu_selected;
    bool use_gpu;                           // use gpu or not
    unsigned int batch_size;                // batch size of input and result

    bool running;                           // the lib is ruunning?
    std::unique_ptr<std::thread> loop;      // loop thread

    std::queue<task_type> tasks;            // task queue
    std::mutex lock;                        // lock for queue
    std::condition_variable cv;             // conditioinal variable

    std::shared_ptr<torch::jit::script::Module> module;     // torch module
};
