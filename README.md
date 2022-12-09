Wrapping trained PyTorch GAN model in c++ for frontend display
============

Table of contents
* c++ Project build
* PyThon code walkthrough
* Benchmark results

## Project build
### Prerequisite
* Libtorch 1.13.0 cxx11 ABI version
* OpenCV 4.6.0 from source

### CMake files
Before building your project, you need to modify cmake files under folder `GAN-cplus/CMakeLists.txt`. Replace list PATH of OpenCV and Libtorch with your own build PATH.

### C++ project build
Run following commands to build your project:
`cd GAN-cplus/build`
`cmake ..`
`make`
If no errors occured, c++ module should be built sucessfully. Run following commands to execute program:
`./GAN-cplus ../gan-py/gen-1.pt`
File `gan-py/gen-1.pt` is pytorch Torch Script file by tracking original PyTorch model. This will be read by our main.cpp program.

## Python code walkthrough
I also add some python scripts for benchmarking under folder `gan-py`. cgan0620.pth is a original GAN model trained using pytorch for generation of 3D Micro-CT image of rocks. `model.py` is the script for the architecture of conditional GAN, whereas `load.ipynb` is the notebook displaying how to use python to read the model and display the time for generating 3D image at each epoch, for benchmarking.

### Benchmark result
The result shows that pytorch runs much faster than pytorch. This is may due to the reason of lack of parallelism on libtorch.