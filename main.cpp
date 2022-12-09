#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <string>
#include <chrono>


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

torch::Tensor fake_gen(torch::jit::script::Module module, const float phi, int sample_num){
  // evaluate program executation time
  
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<torch::jit::IValue> input_gen;
  input_gen.push_back(torch::randn({sample_num,100}));
  input_gen.push_back(torch::ones({sample_num,1})*phi);
  at::Tensor img_fake = module.forward(input_gen).toTensor();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
  std::cout << "The image inference time is: " << duration.count() << " milliseconds" << std::endl;
  return img_fake;
}


torch::Tensor image_proc(torch::Tensor image, const int sample_idx, const int z_idx){
    
    // slice the iamge
    torch::Tensor img_fake_reshape = image.index({
        sample_idx,
        torch::indexing::Slice(),
        z_idx,
        torch::indexing::Slice(),
        torch::indexing::Slice()
        });
    
    // permute and normalize iamge
    img_fake_reshape = img_fake_reshape.to(torch::kCPU).to(torch::kFloat32).detach();
    img_fake_reshape = img_fake_reshape.permute({1,2,0}).contiguous();
    
    // normalize image
    auto max_img = torch::max(img_fake_reshape);
    auto min_img = torch::min(img_fake_reshape);
    torch::Tensor norm_img = (img_fake_reshape - min_img) / (max_img-min_img);
    norm_img = norm_img.mul(255.0).clamp(0,255);

    std::cout <<
     "After image processing and slicing, generated image size is "<<
    img_fake_reshape.sizes() << std::endl;

    return norm_img;

}


cv::Mat ModuletoMat(
  torch::jit::script::Module module, 
  float phi,
  const int sample_idx,
  const int z_idx,
  int sample_num
  ){
  // imagle generation
  at::Tensor img_fake = fake_gen(module,phi,sample_num);
  std::cout << "Generated image shape is " << img_fake.sizes() << std::endl;
  // image processing and slicing
  torch::Tensor norm_img = image_proc(img_fake,sample_idx,z_idx);
  // convert image to opencv mat
  cv::Mat binary_image = TensorToMat(norm_img);
  return binary_image;
  }


// main func
int main(int argc, const char* argv[]) {
  torch::jit::getBailoutDepth() = 1;

  // Parameters initialization
  // Need to be gloabl to fit trackbar
  torch::jit::script::Module module;
  torch::manual_seed(7);
  int sample_idx {0};
  int thin_sec_idx {0};

    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

  try {
    // load the model
    module = torch::jit::load(argv[1]);
    std::cout<< "GAN model loaded sucessfully" << std::endl;
  }

  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  int sample_num;
  std::cout << "How many iamges you want to generate at each inference?";
  std::cin >> sample_num;

  for (int i{10};i<51;i+=5){
    float phi = float(i)/100.0;
    std::ostringstream ss;
    ss << phi;
    std::string phi_name(ss.str());
    phi_name = "Generated image at porosity " + phi_name;


    cv::Mat binary_image = ModuletoMat(module,phi,sample_idx,thin_sec_idx,sample_num);


    cv::namedWindow(phi_name,cv::WINDOW_NORMAL);
    // cv::setWindowProperty(phi_name,cv::WINDOW_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    cv::resizeWindow(phi_name,900,900);
    cv::moveWindow(phi_name,50,50);
    cv::imshow(phi_name,binary_image);
    cv::waitKey(3000);
  }
  cv::waitKey(0);
  return 0; 
}