#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

void SyncedMemory::Clear() {
  if (cpu_ptr_ && own_cpu_data_) {
    Caffe::ReleaseCpuBuffer(cpu_ptr_);
  }
  cpu_ptr_ = NULL;
  own_cpu_data_ = false;

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    Caffe::ReleaseGpuBuffer(gpu_ptr_);
  }
  gpu_ptr_ = NULL;
  own_gpu_data_ = false;
#endif  // CPU_ONLY
  head_ = UNINITIALIZED;
}

SyncedMemory::~SyncedMemory() {
  Clear();
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    cpu_ptr_=Caffe::CpuBuffer(capacity_);
    // TODO: Jinwei: clear memory may cause bad performance
    // caffe_memset(capacity_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      cpu_ptr_=Caffe::CpuBuffer(capacity_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    gpu_ptr_=Caffe::GpuBuffer(capacity_);
    // TODO: Jinwei: clear memory may cause bad performance
    // caffe_gpu_memset(capacity_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      gpu_ptr_=Caffe::GpuBuffer(capacity_);
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {
  if(capacity_==0) {
    return nullptr;
  }
  else {
    to_cpu();
    return static_cast<const void*>(cpu_ptr_);
  }
}

void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    Caffe::ReleaseCpuBuffer(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
#ifndef CPU_ONLY
  if(capacity_==0) {
    return nullptr;
  }
  else {
    to_gpu();
    return static_cast<const void*>(gpu_ptr_);
  }
#else
  NO_GPU;
  return nullptr;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    Caffe::ReleaseGpuBuffer(gpu_ptr_);
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  if(capacity_==0) {
    return nullptr;
  }
  else {
    to_cpu();
    head_ = HEAD_AT_CPU;
    return cpu_ptr_;
  }
}

void* SyncedMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
  if(capacity_==0) {
    return nullptr;
  }
  else {
    to_gpu();
    head_ = HEAD_AT_GPU;
    return gpu_ptr_;
  }
#else
  NO_GPU;
  return nullptr;
#endif
}

void SyncedMemory::Resize(size_t new_size) {
  size_=new_size;
  if(size_>capacity_) {
    capacity_=size_;
    Clear();
  }
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    gpu_ptr_=Caffe::GpuBuffer(capacity_);
    // TODO: Jinwei: clear memory may cause bad performance
    // caffe_gpu_memset(capacity_, 0, gpu_ptr_);
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

}  // namespace caffe

