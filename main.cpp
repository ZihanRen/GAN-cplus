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


// print out tensor for debug
void print_t(torch::Tensor input){
    float* ptr = (float*)input.data_ptr();
    std::cout << "\nBegin Printing 3D tensor:" << std::endl;
    for (int i = 0; i < input.sizes()[0]; ++i)
      {
        for (int j = 0; j < input.sizes()[1]; ++j)
        {
          for (int k=0; k<input.sizes()[2];++k){
          std::cout << " " << *ptr++ << " ";
          }
          }
    std::cout << std::endl;
}
    std::cout<<"\n";
}

// tensor to opencv mat
cv::Mat TensorToMat(torch::Tensor tensor){
  int img_type = CV_32FC1;
  int64_t height,width;
  height = tensor.size(0);
  width = tensor.size(1);
  cv::Mat image;

  try{
    image = cv::Mat(cv::Size(width, height), img_type, tensor.data_ptr<float>());
    std::cout << "The size of image is: " << image.size()<< std::endl;}
    catch (const c10::Error& e) {
    std::cerr << "error convert to mat model\n";
  }

  // convert to binary
  cv::Mat binary_image;
  cv::threshold(image,binary_image,105,255,cv::THRESH_BINARY);
  return binary_image;
}

// main func
int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }
  // make sure the model is reproducible
  torch::jit::script::Module module;
  torch::manual_seed(7);

  try {
    // load the model
    module = torch::jit::load(argv[1]);
    std::cout<< "ScriptModule loaded sucessfully" << std::endl;
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // create input for module - torch jit IValue
  std::vector<torch::jit::IValue> input_gen;
  input_gen.push_back(torch::randn({4,100}));
  input_gen.push_back(torch::ones({4,1})*0.2);

  at::Tensor img_fake = module.forward(input_gen).toTensor();
  std::cout << "Fake image shape is " << img_fake.sizes() << std::endl;
  int sample_idx {0};
  int thin_sec_idx {0};
  int sample_idx_z {1};

  // slice the iamge
  torch::Tensor img_fake_reshape = img_fake.index(
    {sample_idx,torch::indexing::Slice(),thin_sec_idx,torch::indexing::Slice() ,torch::indexing::Slice() }
    );
  std::cout << "After reshape, fake image size is " << img_fake_reshape.sizes() << std::endl;

  // feed tensor into opencv format and display
  img_fake_reshape = img_fake_reshape.to(torch::kCPU).to(torch::kFloat32).detach();
  img_fake_reshape = img_fake_reshape.permute({1,2,0}).contiguous();
  // normalize image
  auto max_img = torch::max(img_fake_reshape);
  auto min_img = torch::min(img_fake_reshape);
  torch::Tensor norm_img = (img_fake_reshape - min_img) / (max_img-min_img);
  norm_img = norm_img.mul(255.0).clamp(0,255);

  cv::Mat binary_image;
  binary_image = TensorToMat(norm_img);
  std::cout << "Size of function output is: " << binary_image.size() << std::endl;

  // define image parameter
  // int img_type = CV_32FC1;
  // int64_t height,width;
  // height = norm_img.size(0);
  // width = norm_img.size(1);
  // cv::Mat imgbin;

  // try{
  //   imgbin = cv::Mat(cv::Size(width, height), img_type, norm_img.data_ptr<float>());
  //   std::cout << "The size of image is: " << imgbin.size()<< std::endl;}

  //   catch (const c10::Error& e) {
  //   std::cerr << "error convert to mat model\n";
  //   return -1;
  // }

  // // convert to binary
  // cv::Mat binary_image;
  // cv::threshold(imgbin,binary_image,105,255,cv::THRESH_BINARY);

  // Display image
  cv::namedWindow("Display window",cv::WINDOW_AUTOSIZE);
  cv::imshow("Display window",binary_image);
  cv::waitKey(0);
  // cv::imwrite("imggen.png",binary_image);

}