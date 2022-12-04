// #include <torch/script.h> // One-stop header.
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat TensorToCVMat(torch::Tensor tensor)
{
    std::cout << "converting tensor to cvmat\n";
    tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
    tensor = tensor.to(torch::kCPU);
    int64_t height = tensor.size(0);
    int64_t width = tensor.size(1);
    cv::Mat mat(width, height, CV_8UC3);
    std::memcpy((void *)mat.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());
    return mat.clone();
}



// load torch model
int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

  // initialize some parameters
  torch::jit::script::Module module;
  torch::manual_seed(7);

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
  // if you want to have tensor operation, you need to convert torch::jit 
  // into torch::tensor
  std::vector<torch::jit::IValue> input_gen;
  input_gen.push_back(torch::randn({4,100}));
  input_gen.push_back(torch::ones({4,1})*0.2);

  at::Tensor img_fake = module.forward(input_gen).toTensor();
  std::cout << "Fake image shape is " << img_fake.sizes() << std::endl;
  int sample_idx {0};
  int thin_sec_idx {0};
  int sample_idx_z {1};

  torch::Tensor img_fake_reshape = img_fake.select(0,sample_idx).select(sample_idx_z,0);
  std::cout << "After reshape, fake image size is " << img_fake_reshape.sizes() << std::endl;

  // feed tensor into opencv format and display
  img_fake_reshape = img_fake_reshape.detach().permute({1,2,0});
  img_fake_reshape = img_fake_reshape.mul(255).clamp(0,255).to(torch::kU8);
  img_fake_reshape = img_fake_reshape.to(torch::kCPU);
  int height,width;
  height = img_fake_reshape.size(0);
  width = img_fake_reshape.size(1);
  cv::Mat imgbin(cv::Size(width,height), CV_16U, img_fake_reshape.data_ptr());
  std::cout << "The size of image is: " << imgbin.size()<< std::endl;

  cv::imshow("Display window",imgbin);

  // cv::imwrite("genimg.png",imgbin);
  // cv::imwrite("genimg.png",imgbin);


  






  // convert torch tensor to opencv mat
  // img_fake = img_fake.squeeze().detach().permute({1,2,0})
  
}