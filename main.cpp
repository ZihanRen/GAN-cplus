// #include <torch/script.h> // One-stop header.
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <vector>
// #include <opencv2/opencv.hpp>

// load torch model
int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
    std::cout<< "ScriptModule loaded sucessfully" << std::endl;
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // create input for module - torch jit IValue
  // if you want to have tensor operation, you need to convert torch::jit::IValue into 
  // torch::tensor

  std::vector<torch::jit::IValue> input_gen;  
  input_gen.push_back(torch::randn({4,100}));
  input_gen.push_back(torch::ones({4,1})*0.2);

  at::Tensor output = module.forward(input_gen).toTensor();
  std::cout << "Input noise shape is " << output.sizes() << std::endl;
  
}