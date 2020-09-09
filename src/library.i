%module(threads="12") library

%{
#include "gamecore.h"
#include "predict_lib.h"
#include "mcts.h"
#include "mcts_pure.h"
#include "single_lib.h"
%}

%include "std_vector.i"
%include "std_map.i"
%include "std_string.i"
namespace std {
  %template(IntVector) vector<int>;
  %template(IntVectorVector) vector<vector<int>>;
  %template(IntVectorVectorVector) vector<vector<vector<int>>>;
  %template(DoubleVector) vector<double>;
  %template(DoubleVectorVector) vector<vector<double>>;
  %template(Int2StrMap) map<int,string>;
  %template(FloatVector) vector<float>;
  %template(FloatVectorVector) vector<vector<float>>;
}


%include "gamecore.h"
%include "mcts.h"
%include "mcts_pure.h"


class predict_model {
 public:
  predict_model(std::string model_path, bool use_gpu, unsigned int batch_size, int gpu_selected);
  ~predict_model();
  void set_batch_size(unsigned int batch_size);
};

class single_lib {
public:
  single_lib(std::string model_path, unsigned int batch_size, int gpu_selected);
  ~single_lib();
  void set_batch_size(unsigned int batch_size);
};
