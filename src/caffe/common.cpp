#include <thread>
#include <cmath>
#include <cstdio>
#include <ctime>

#ifdef _MSC_VER
#include <process.h>
#else
#include <unistd.h>
#endif

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

bool GlogDeployLogMessage::enable=false;

namespace caffe {

#if __cplusplus >= 201103L
thread_local static std::shared_ptr<Caffe> thread_instance_;
#else
#error "Thread local Caffe singleton is not available"
static std::shared_ptr<Caffe> thread_instance_;
#endif

struct MemoryNode {
  void* ptr;
  bool used;
  size_t size;
  size_t size_after;
  MemoryNode* next;
};

Caffe& Caffe::Get() {
  if (!thread_instance_.get()) {
    thread_instance_.reset(new Caffe());
  }
  return *(thread_instance_.get());
}

// random seeding
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  LOG(INFO) << "System entropy source not available, "
              "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  pid = getpid();
  s = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}

#ifdef CPU_ONLY  // CPU-only Caffe.

Caffe::Caffe()
    : random_generator_(), mode_(Caffe::CPU), cpu_workspace_(nullptr), cpu_workspace_size_(0) { }

Caffe::~Caffe() {
  for(size_t i=0; i<cpu_memory_list_.size(); i++) {
    free(cpu_memory_list_[i]->ptr);
    vector<MemoryNode*> nodes;
    MemoryNode* node=cpu_memory_list_[i];
    while(node!=nullptr) {
      nodes.push_back(node);
      node=node->next;
    }
    for(const auto& node: nodes) {
      if(node->used) {
        LOG(ERROR) << "Memory leak on CPU memory is detected at " << node->ptr << ", size " << node->size;
      }
      delete node;
    }
  }
  if(cpu_workspace_size_>0) {
    free(cpu_workspace_);
  }
}

void Caffe::set_random_seed(const unsigned int seed) {
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  NO_GPU;
}

void Caffe::DeviceQuery() {
  NO_GPU;
}

bool Caffe::CheckDevice(const int device_id) {
  NO_GPU;
  return false;
}

int Caffe::FindDevice(const int start_id) {
  NO_GPU;
  return -1;
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#else  // Normal GPU + CPU Caffe.

Caffe::Caffe()
    : cublas_handle_(NULL), curand_generator_(NULL), device_set_(false),
#ifdef USE_CUDNN
    cudnn_handle_(NULL),
#endif
    random_generator_(), mode_(Caffe::CPU), gpu_workspace_(nullptr), gpu_workspace_size_(0),
    cpu_workspace_(nullptr), cpu_workspace_size_(0) {
  // // Try to create a cublas handler, and report an error if failed (but we will
  // // keep the program running as one might just want to run CPU code).
  // if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
  //   LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
  // }
  // // Try to create a curand handler.
  // if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
  //     != CURAND_STATUS_SUCCESS ||
  //     curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen())
  //     != CURAND_STATUS_SUCCESS) {
  //   LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
  // }
}

Caffe::~Caffe() {
  if(cublas_handle_) {
    CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  }
  if(curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
#ifdef USE_CUDNN
  if(cudnn_handle_) {
    CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
  }
#endif
  for(size_t i=0; i<gpu_memory_list_.size(); i++) {
    cudaFree(gpu_memory_list_[i]->ptr);
    vector<MemoryNode*> nodes;
    MemoryNode* node=gpu_memory_list_[i];
    while(node!=nullptr) {
      nodes.push_back(node);
      node=node->next;
    }
    for(const auto& node: nodes) {
      if(node->used) {
        LOG(ERROR) << "Memory leak on GPU memory is detected at " << node->ptr << ", size " << node->size;
      }
      delete node;
    }
  }
  for(size_t i=0; i<cpu_memory_list_.size(); i++) {
    cudaFreeHost(cpu_memory_list_[i]->ptr);
    vector<MemoryNode*> nodes;
    MemoryNode* node=cpu_memory_list_[i];
    while(node!=nullptr) {
      nodes.push_back(node);
      node=node->next;
    }
    for(const auto& node: nodes) {
      if(node->used) {
        LOG(ERROR) << "Memory leak on CPU memory is detected at " << node->ptr << ", size " << node->size;
      }
      delete node;
    }
  }
  if(gpu_workspace_size_>0) {
    cudaFree(gpu_workspace_);
  }
  if(cpu_workspace_size_>0) {
    cudaFreeHost(Get().cpu_workspace_);
  }
}

void Caffe::set_random_seed(const unsigned int seed) {
  // Curand seed
  static bool g_curand_availability_logged = false;
  if (Get().curand_generator_) {
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(),
        seed));
    CURAND_CHECK(curandSetGeneratorOffset(curand_generator(), 0));
  } else {
    if (!g_curand_availability_logged) {
        LOG(ERROR) <<
            "Curand not available. Skipping setting the curand seed.";
        g_curand_availability_logged = true;
    }
  }
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  if(!Get().device_set_) {
    Get().device_set_=true;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));
    CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
        CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_,
        cluster_seedgen()));
#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnCreate(&Get().cudnn_handle_));
#endif
  }
  else {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (current_device == device_id) {
      return;
    }
    CUDA_CHECK(cudaSetDevice(device_id));
    if (Get().cublas_handle_) {
      CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
    }
    if (Get().curand_generator_) {
      CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
    }
    CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));
    CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
        CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_,
        cluster_seedgen()));
    ClearGpuBuffer();
    CHECK_EQ(Get().gpu_memory_list_.size(), 0);
#ifdef USE_CUDNN
    if (Get().cudnn_handle_) {
      CUDNN_CHECK(cudnnDestroy(Get().cudnn_handle_));
    }
    CUDNN_CHECK(cudnnCreate(&Get().cudnn_handle_));
#endif
  }
}

void* Caffe::GpuBuffer(size_t size) {
  if(size==0) {
    return nullptr;
  }
  vector<MemoryNode*>& memory_list=Get().gpu_memory_list_;
  for(size_t i=0; i<memory_list.size(); i++) {
    void* ptr=nullptr;
    MemoryNode* node=memory_list[i];
    for(;;) {
      if(!node->used && node->size>=size) {
        ptr=node->ptr;
        node->used=true;
        if(node->size>size) {
          MemoryNode* new_node=new MemoryNode;
          new_node->ptr=static_cast<char*>(node->ptr)+size;
          new_node->used=false;
          new_node->size=node->size-size;
          new_node->size_after=node->size_after-size;
          new_node->next=node->next;
          node->size=size;
          node->next=new_node;
        }
        break;
      }
      else if(node->size_after-node->size>=size) {
        if(node->next!=nullptr) {
          node=node->next;
        }
        else {
          break;
        }
      }
      else {
        break;
      }
    }
    if(ptr!=nullptr) {
      return ptr;
    }
  }
  void* new_ptr;
  cudaError_t err=cudaMalloc(&new_ptr, size);
  if(err==cudaSuccess) {
    caffe_gpu_memset(size, 0, new_ptr);
    MemoryNode* new_node=new MemoryNode;
    new_node->ptr=new_ptr;
    new_node->used=true;
    new_node->size=size;
    new_node->size_after=size;
    new_node->next=nullptr;
    memory_list.push_back(new_node);
    return new_ptr;
  }
  else {
    return nullptr;
  }
}

void Caffe::ReleaseGpuBuffer(const void* buffer) {
  if(buffer==nullptr) {
    return;
  }
  vector<MemoryNode*>& memory_list=Get().gpu_memory_list_;
  for(size_t i=0; i<memory_list.size(); i++) {
    MemoryNode* node=memory_list[i];
    bool released=false;
    for(;;) {
      if(node->ptr==buffer) {
        node->used=false;
        released=true;
        break;
      }
      else if(node->next!=nullptr) {
        node=node->next;
      }
      else {
        break;
      }
    }
    if(released) {
      MemoryNode* node=memory_list[i];
      for(;;) {
        if(node->next!=nullptr) {
          if(!node->used && !node->next->used) {
            node->size+=node->next->size;
            MemoryNode* next=node->next;
            node->next=node->next->next;
            delete next;
          }
          else {
            node=node->next;
          }
        }
        else {
          break;
        }
      }
      break;
    }
  }
}

void Caffe::ClearGpuBuffer(void) {
  vector<MemoryNode*> new_memory_list;
  vector<MemoryNode*>& memory_list=Get().gpu_memory_list_;
  for(size_t i=0; i<memory_list.size(); i++) {
    if(memory_list[i]->next==nullptr && !memory_list[i]->used) {
      CUDA_CHECK(cudaFree(memory_list[i]->ptr));
      delete memory_list[i];
    }
    else {
      new_memory_list.push_back(memory_list[i]);
    }
  }
  memory_list=new_memory_list;
}

void* Caffe::GpuWorkspace(size_t size) {
  if(Get().gpu_workspace_size_>=size) {
    return Get().gpu_workspace_;
  }
  else {
    void* new_ptr;
    cudaError_t err=cudaMalloc(&new_ptr, size);
    if(err==cudaSuccess) {
      CUDA_CHECK(cudaFree(Get().gpu_workspace_));
      Get().gpu_workspace_size_=size;
      Get().gpu_workspace_=new_ptr;
      return new_ptr;
    }
    else {
      return nullptr;
    }
  }
}

void Caffe::DeviceQuery() {
  cudaDeviceProp prop;
  int device;
  if (cudaSuccess != cudaGetDevice(&device)) {
    printf("No cuda device present.\n");
    return;
  }
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  LOG(INFO) << "Device id:                     " << device;
  LOG(INFO) << "Major revision number:         " << prop.major;
  LOG(INFO) << "Minor revision number:         " << prop.minor;
  LOG(INFO) << "Name:                          " << prop.name;
  LOG(INFO) << "Total global memory:           " << prop.totalGlobalMem;
  LOG(INFO) << "Total shared memory per block: " << prop.sharedMemPerBlock;
  LOG(INFO) << "Total registers per block:     " << prop.regsPerBlock;
  LOG(INFO) << "Warp size:                     " << prop.warpSize;
  LOG(INFO) << "Maximum memory pitch:          " << prop.memPitch;
  LOG(INFO) << "Maximum threads per block:     " << prop.maxThreadsPerBlock;
  LOG(INFO) << "Maximum dimension of block:    "
      << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
      << prop.maxThreadsDim[2];
  LOG(INFO) << "Maximum dimension of grid:     "
      << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
      << prop.maxGridSize[2];
  LOG(INFO) << "Clock rate:                    " << prop.clockRate;
  LOG(INFO) << "Total constant memory:         " << prop.totalConstMem;
  LOG(INFO) << "Texture alignment:             " << prop.textureAlignment;
  LOG(INFO) << "Concurrent copy and execution: "
      << (prop.deviceOverlap ? "Yes" : "No");
  LOG(INFO) << "Number of multiprocessors:     " << prop.multiProcessorCount;
  LOG(INFO) << "Kernel execution timeout:      "
      << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
  return;
}

bool Caffe::CheckDevice(const int device_id) {
  // This function checks the availability of GPU #device_id.
  // It attempts to create a context on the device by calling cudaFree(0).
  // cudaSetDevice() alone is not sufficient to check the availability.
  // It lazily records device_id, however, does not initialize a
  // context. So it does not know if the host thread has the permission to use
  // the device or not.
  //
  // In a shared environment where the devices are set to EXCLUSIVE_PROCESS
  // or EXCLUSIVE_THREAD mode, cudaSetDevice() returns cudaSuccess
  // even if the device is exclusively occupied by another process or thread.
  // Cuda operations that initialize the context are needed to check
  // the permission. cudaFree(0) is one of those with no side effect,
  // except the context initialization.
  bool r = ((cudaSuccess == cudaSetDevice(device_id)) &&
            (cudaSuccess == cudaFree(0)));
  // reset any error that may have occurred.
  cudaGetLastError();
  return r;
}

int Caffe::FindDevice(const int start_id) {
  // This function finds the first available device by checking devices with
  // ordinal from start_id to the highest available value. In the
  // EXCLUSIVE_PROCESS or EXCLUSIVE_THREAD mode, if it succeeds, it also
  // claims the device due to the initialization of the context.
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  for (int i = start_id; i < count; i++) {
    if (CheckDevice(i)) return i;
  }
  return -1;
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_.reset(other.generator_.get());
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}

#endif  // CPU_ONLY

void* Caffe::CpuBuffer(size_t size) {
  if(size==0) {
    return nullptr;
  }
  vector<MemoryNode*>& memory_list=Get().cpu_memory_list_;
  for(size_t i=0; i<memory_list.size(); i++) {
    void* ptr=nullptr;
    MemoryNode* node=memory_list[i];
    for(;;) {
      if(!node->used && node->size>=size) {
        ptr=node->ptr;
        node->used=true;
        if(node->size>size) {
          MemoryNode* new_node=new MemoryNode;
          new_node->ptr=static_cast<char*>(node->ptr)+size;
          new_node->used=false;
          new_node->size=node->size-size;
          new_node->size_after=node->size_after-size;
          new_node->next=node->next;
          node->size=size;
          node->next=new_node;
        }
        break;
      }
      else if(node->size_after-node->size>=size) {
        if(node->next!=nullptr) {
          node=node->next;
        }
        else {
          break;
        }
      }
      else {
        break;
      }
    }
    if(ptr!=nullptr) {
      return ptr;
    }
  }
#ifdef CPU_ONLY
  void* new_ptr=malloc(size);
  if(new_ptr!=nullptr) {
#else
  void* new_ptr;
  cudaError_t err=cudaMallocHost(&new_ptr, size);
  if(err==cudaSuccess) {
#endif
    caffe_memset(size, 0, new_ptr);
    MemoryNode* new_node=new MemoryNode;
    new_node->ptr=new_ptr;
    new_node->used=true;
    new_node->size=size;
    new_node->size_after=size;
    new_node->next=nullptr;
    memory_list.push_back(new_node);
    return new_ptr;
  }
  else {
    return nullptr;
  }
}

void Caffe::ReleaseCpuBuffer(const void* buffer) {
  if(buffer==nullptr) {
    return;
  }
  vector<MemoryNode*>& memory_list=Get().cpu_memory_list_;
  for(size_t i=0; i<memory_list.size(); i++) {
    MemoryNode* node=memory_list[i];
    bool released=false;
    for(;;) {
      if(node->ptr==buffer) {
        node->used=false;
        released=true;
        break;
      }
      else if(node->next!=nullptr) {
        node=node->next;
      }
      else {
        break;
      }
    }
    if(released) {
      MemoryNode* node=memory_list[i];
      for(;;) {
        if(node->next!=nullptr) {
          if(!node->used && !node->next->used) {
            node->size+=node->next->size;
            MemoryNode* next=node->next;
            node->next=node->next->next;
            delete next;
          }
          else {
            node=node->next;
          }
        }
        else {
          break;
        }
      }
      break;
    }
  }
}

void Caffe::ClearCpuBuffer(void) {
  vector<MemoryNode*> new_memory_list;
  vector<MemoryNode*>& memory_list=Get().cpu_memory_list_;
  for(size_t i=0; i<memory_list.size(); i++) {
    if(memory_list[i]->next==nullptr && !memory_list[i]->used) {
#ifdef CPU_ONLY
      free(memory_list[i]->ptr);
#else
      CUDA_CHECK(cudaFreeHost(memory_list[i]->ptr));
#endif
      delete memory_list[i];
    }
    else {
      new_memory_list.push_back(memory_list[i]);
    }
  }
  memory_list=new_memory_list;
}

void* Caffe::CpuWorkspace(size_t size) {
  if(Get().cpu_workspace_size_>=size) {
    return Get().cpu_workspace_;
  }
  else {
#ifndef CPU_ONLY
    void* new_ptr;
    cudaError_t err=cudaMallocHost(&new_ptr, size);
    if(err==cudaSuccess) {
      CUDA_CHECK(cudaFreeHost(Get().cpu_workspace_));
#else  // CPU_ONLY
    void* new_ptr=malloc(size);
    if(new_ptr!=nullptr) {
      free(Get().cpu_workspace_);
#endif  // CPU_ONLY
      Get().cpu_workspace_size_=size;
      Get().cpu_workspace_=new_ptr;
      return new_ptr;
    }
    else {
      return nullptr;
    }
  }
}

}  // namespace caffe

#include "caffe/layer_factory.hpp"

namespace caffe {

#define LAYER_REGISTERER(name) \
	extern LayerRegisterer<float> g_creator_f_##name; \
	LayerRegisterer<float>* p_g_creator_f_##name = &g_creator_f_##name;

LAYER_REGISTERER(LSTM);
LAYER_REGISTERER(Exp);
LAYER_REGISTERER(Split);
LAYER_REGISTERER(BatchNorm);
LAYER_REGISTERER(Eltwise);
LAYER_REGISTERER(Filter);
LAYER_REGISTERER(LSTMUnit);
LAYER_REGISTERER(EuclideanLoss);
LAYER_REGISTERER(Flatten);
LAYER_REGISTERER(Crop);
LAYER_REGISTERER(AbsVal);
LAYER_REGISTERER(MultinomialLogisticLoss);
LAYER_REGISTERER(Im2col);
LAYER_REGISTERER(Scale);
LAYER_REGISTERER(RNN);
LAYER_REGISTERER(SoftmaxWithLoss);
LAYER_REGISTERER(SPP);
LAYER_REGISTERER(PReLU);
LAYER_REGISTERER(Power);
LAYER_REGISTERER(Reduction);
LAYER_REGISTERER(Tile);
LAYER_REGISTERER(Input);
LAYER_REGISTERER(HingeLoss);
LAYER_REGISTERER(Reshape);
LAYER_REGISTERER(BatchReindex);
LAYER_REGISTERER(DummyData);
LAYER_REGISTERER(Threshold);
LAYER_REGISTERER(InnerProduct);
LAYER_REGISTERER(ArgMax);
LAYER_REGISTERER(ContrastiveLoss);
LAYER_REGISTERER(Concat);
LAYER_REGISTERER(ELU);
LAYER_REGISTERER(Deconvolution);
LAYER_REGISTERER(Dropout);
LAYER_REGISTERER(Silence);
LAYER_REGISTERER(MVN);
LAYER_REGISTERER(Embed);
LAYER_REGISTERER(Bias);
LAYER_REGISTERER(Accuracy);
LAYER_REGISTERER(Parameter);
LAYER_REGISTERER(InfogainLoss);
LAYER_REGISTERER(Log);
LAYER_REGISTERER(SigmoidCrossEntropyLoss);
LAYER_REGISTERER(BNLL);
LAYER_REGISTERER(Slice);
LAYER_REGISTERER(Convolution);
LAYER_REGISTERER(Pooling);
LAYER_REGISTERER(LRN);
LAYER_REGISTERER(ReLU);
LAYER_REGISTERER(Sigmoid);
LAYER_REGISTERER(Softmax);
LAYER_REGISTERER(TanH);
#ifdef WITH_PYTHON_LAYER
LAYER_REGISTERER(Python);
#endif

}
