#include "caffe/caffe.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
#ifndef CPU_ONLY
  cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif
}

#ifndef CPU_ONLY
using caffe::CAFFE_TEST_CUDA_PROP;
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
#ifndef CPU_ONLY
  // Before starting testing, let's first print out a few cuda defice info.
  int device;
  cudaGetDeviceCount(&device);
  if (device == 0) {
    cout << "Try to running with GPU, but no GPU is available." << endl;
    return -1;
  }
  cout << "Cuda number of devices: " << device << endl;
  if (argc > 1) {
    device = atoi(argv[1]);
  } else if (CUDA_TEST_DEVICE >= 0) {
    device = CUDA_TEST_DEVICE;
  }
  cout << "Using device id: " << device << endl;
  caffe::Caffe::SetDevice(device);
  cudaGetDeviceProperties(&CAFFE_TEST_CUDA_PROP, device);
  cout << "Device name: " << CAFFE_TEST_CUDA_PROP.name << endl;
#endif
  // invoke the test.
  return RUN_ALL_TESTS();
}
