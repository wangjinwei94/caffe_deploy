#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  if(Caffe::mode()==Caffe::CPU) {
    return;
  }

  CHECK_EQ(2, this->num_spatial_axes_)
    << "CuDNNConvolution input must have 2 spatial axes "
    << "(e.g., height and width). "
    << "Use 'engine: CAFFE' for general ND convolution.";
  kernel_h_ = this->kernel_shape_.cpu_data()[0];
  kernel_w_ = this->kernel_shape_.cpu_data()[1];
  stride_h_ = this->stride_.cpu_data()[0];
  stride_w_ = this->stride_.cpu_data()[1];
  pad_h_ = this->pad_.cpu_data()[0];
  pad_w_ = this->pad_.cpu_data()[1];
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      kernel_h_, kernel_w_);
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createConvolutionDesc<Dtype>(&conv_desc_);
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
    bias_offset_ = (this->num_output_ / this->group_);
  }
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  if(Caffe::mode()==Caffe::CPU) {
    return;
  }

  if(bottom[0]->shape()!=input_shape_) {
    input_shape_ = bottom[0]->shape();
    bottom_offset_ = this->bottom_dim_ / this->group_;
    top_offset_ = this->top_dim_ / this->group_;
    const int height = bottom[0]->shape(this->channel_axis_ + 1);
    const int width = bottom[0]->shape(this->channel_axis_ + 2);
    const int height_out = top[0]->shape(this->channel_axis_ + 1);
    const int width_out = top[0]->shape(this->channel_axis_ + 2);

    cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, this->num_,
        this->channels_ / this->group_, height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    cudnn::setTensor4dDesc<Dtype>(&top_desc_, this->num_,
        this->num_output_ / this->group_, height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
    cudnn::setConvolutionDesc<Dtype>(&conv_desc_, bottom_desc_,
        filter_desc_, pad_h_, pad_w_, stride_h_, stride_w_);

    // Specify workspace limit for kernels directly until we have a
    // planning strategy and a rewrite of Caffe's GPU memory mangagement
    static const size_t workspace_limit_bytes = 1024*1024*1024;
    void* temp_bottom_data=Caffe::GpuBuffer(bottom[0]->count()*sizeof(Dtype));
    void* temp_weight_data=Caffe::GpuBuffer(this->blobs_[0]->count()*sizeof(Dtype));
    void* temp_top_data=Caffe::GpuBuffer(top[0]->count()*sizeof(Dtype));

    vector<cudnnConvolutionFwdAlgoPerf_t> fwd_perf(CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
    int fwd_perf_cnt=0;
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(Caffe::cudnn_handle(),
      bottom_desc_, temp_bottom_data, filter_desc_, temp_weight_data,
      conv_desc_, top_desc_, temp_top_data, fwd_perf.size(), &fwd_perf_cnt,
      fwd_perf.data(), Caffe::GpuWorkspace(workspace_limit_bytes), workspace_limit_bytes));
    CHECK_GT(fwd_perf_cnt, 0);
    CUDNN_CHECK(fwd_perf[0].status);
    fwd_algo_=fwd_perf[0].algo;
    workspace_fwd_size_=fwd_perf[0].memory;

    // choose backward algorithm for filter
    if (this->phase_ == TRAIN) {
      void* temp_bottom_diff=Caffe::GpuBuffer(bottom[0]->count()*sizeof(Dtype));
      void* temp_weight_diff=Caffe::GpuBuffer(this->blobs_[0]->count()*sizeof(Dtype));
      void* temp_top_diff=Caffe::GpuBuffer(top[0]->count()*sizeof(Dtype));

      vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_perf(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT);
      int bwd_filter_perf_cnt=0;
      CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(Caffe::cudnn_handle(),
        bottom_desc_, temp_bottom_data, top_desc_, temp_top_diff, conv_desc_,
        filter_desc_, temp_weight_diff, bwd_filter_perf.size(), &bwd_filter_perf_cnt,
        bwd_filter_perf.data(), Caffe::GpuWorkspace(workspace_limit_bytes), workspace_limit_bytes));
      CHECK_GT(bwd_filter_perf_cnt, 0);
      CUDNN_CHECK(bwd_filter_perf[0].status);
      bwd_filter_algo_=bwd_filter_perf[0].algo;
      workspace_bwd_filter_size_=bwd_filter_perf[0].memory;

      vector<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_perf(CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT);
      int bwd_data_perf_cnt=0;
      CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithmEx(Caffe::cudnn_handle(),
        filter_desc_, temp_weight_data, top_desc_, temp_top_diff, conv_desc_,
        bottom_desc_, temp_bottom_diff, bwd_data_perf.size(), &bwd_data_perf_cnt,
        bwd_data_perf.data(), Caffe::GpuWorkspace(workspace_limit_bytes), workspace_limit_bytes));
      CHECK_GT(bwd_data_perf_cnt, 0);
      CUDNN_CHECK(bwd_data_perf[0].status);
      bwd_data_algo_=bwd_data_perf[0].algo;
      workspace_bwd_data_size_=bwd_data_perf[0].memory;

      Caffe::ReleaseGpuBuffer(temp_bottom_diff);
      Caffe::ReleaseGpuBuffer(temp_weight_diff);
      Caffe::ReleaseGpuBuffer(temp_top_diff);
    }
    Caffe::ReleaseGpuBuffer(temp_bottom_data);
    Caffe::ReleaseGpuBuffer(temp_weight_data);
    Caffe::ReleaseGpuBuffer(temp_top_data);

    if(this->bias_term_) {
      cudnn::setTensor4dDesc<Dtype>(&bias_desc_, 1, this->num_output_/this->group_, 1, 1);
    }
  }
}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  if (!handles_setup_) {
    return;
  }
  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyConvolutionDescriptor(conv_desc_);
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(),
            cudnn::dataType<Dtype>::one,
            bottom_desc_,
            bottom_data + bottom_offset_ * g,
            filter_desc_,
            weight + this->weight_offset_ * g,
            conv_desc_,
            fwd_algo_,
            // warning: GpuWorkspace may be NULL
            Caffe::GpuWorkspace(workspace_fwd_size_),
            workspace_fwd_size_,
            cudnn::dataType<Dtype>::zero,
            top_desc_,
            top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(Caffe::cudnn_handle(),
              cudnn::dataType<Dtype>::one,
              bias_desc_,
              bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_desc_,
              top_data + top_offset_ * g));
      }
    }
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(Caffe::cudnn_handle(),
              cudnn::dataType<Dtype>::one,
              top_desc_,
              top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_,
              bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              Caffe::cudnn_handle(),
              cudnn::dataType<Dtype>::one,
              bottom_desc_,
              bottom_data + bottom_offset_ * g,
              top_desc_,
              top_diff + top_offset_ * g,
              conv_desc_,
              bwd_filter_algo_,
              // warning: GpuWorkspace may be NULL
              Caffe::GpuWorkspace(workspace_bwd_filter_size_),
              workspace_bwd_filter_size_,
              cudnn::dataType<Dtype>::one,
              filter_desc_,
              weight_diff + this->weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              Caffe::cudnn_handle(),
              cudnn::dataType<Dtype>::one,
              filter_desc_,
              weight + this->weight_offset_ * g,
              top_desc_,
              top_diff + top_offset_ * g,
              conv_desc_,
              bwd_data_algo_,
              // warning: GpuWorkspace may be NULL
              Caffe::GpuWorkspace(workspace_bwd_data_size_),
              workspace_bwd_data_size_,
              cudnn::dataType<Dtype>::zero,
              bottom_desc_,
              bottom_diff + bottom_offset_ * g));
      }
    }
  }
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
