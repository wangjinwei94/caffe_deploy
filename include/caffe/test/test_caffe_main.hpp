// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
#define CAFFE_TEST_TEST_CAFFE_MAIN_HPP_

#include "caffe/util/glog_deploy.hpp"
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"

using std::cout;
using std::endl;

#ifdef CMAKE_BUILD
  #include "caffe_config.h"
#else
  #define CUDA_TEST_DEVICE -1
  #define CMAKE_SOURCE_DIR "src/"
  #define EXAMPLES_SOURCE_DIR "examples/"
  #define CMAKE_EXT ""
#endif

int main(int argc, char** argv);

namespace caffe {

template <typename TypeParam>
class MultiDeviceTest : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MultiDeviceTest() {
    Caffe::set_mode(TypeParam::device);
  }
  virtual ~MultiDeviceTest() {}
};

// typedef ::testing::Types<float, double> TestDtypes;
typedef ::testing::Types<float> TestDtypes;

template <typename TypeParam>
struct CPUDevice {
  typedef TypeParam Dtype;
  static const Caffe::Brew device = Caffe::CPU;
};

template <typename Dtype>
class CPUDeviceTest : public MultiDeviceTest<CPUDevice<Dtype> > {
};

#ifdef CPU_ONLY

// typedef ::testing::Types<CPUDevice<float>,
//                          CPUDevice<double> > TestDtypesAndDevices;
typedef ::testing::Types<CPUDevice<float>> TestDtypesAndDevices;

#else

template <typename TypeParam>
struct GPUDevice {
  typedef TypeParam Dtype;
  static const Caffe::Brew device = Caffe::GPU;
};

template <typename Dtype>
class GPUDeviceTest : public MultiDeviceTest<GPUDevice<Dtype> > {
};

// typedef ::testing::Types<CPUDevice<float>, CPUDevice<double>,
//                          GPUDevice<float>, GPUDevice<double> >
//                          TestDtypesAndDevices;
typedef ::testing::Types<CPUDevice<float>, GPUDevice<float> >
                         TestDtypesAndDevices;

#endif

template <typename Dtype>
static void FlushLayerBlobsDiff(Layer<Dtype>* layer) {
  for(size_t i=0; i<layer->blobs().size(); i++) {
    caffe_set<Dtype>(layer->blobs()[i]->count(), Dtype(0), layer->blobs()[i]->mutable_cpu_diff());
  }
}

}  // namespace caffe

#endif  // CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
