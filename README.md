# Important

to improve the performance when construct and destruct SyncedMemory frequently

a toy GPU / CPU memory pool was used to allocate memory

and memory initialization in SyncedMemory (caffe_gpu_memset and memset) was removed

so if you create a new Blob, the data in it <font color=red size=5>IS NOT INITIALIZED</font>, you should initialize it by yourself if needed

# Dependency

cublas, cudart, curand (if CPU_ONLY is cleared)

cudnn (if CPU_ONLY is cleared and USE_CUDNN is set)

openblas (if USE_EIGEN is cleared)

# About glog_deploy

Glog_deploy is a header-only wrapper provide LOG/DLOG and CHECK/CHECK_** functions provided by google-glog.

To use glog_deploy, simply include caffe/util/glog_deploy.hpp in your project. If glog/logging.h is included before, glog_deploy will not be used.

By default, the log is turned off and only fatal log will be output. Change GlogDeployLogMessage::enable to turn it on.

# Build

git clone path/to/this/project/on/gitlab; git submodule update --init;

Linux: mkdir target; cd target; cmake .. ; make install; make runtest;

Windows: mkdir target; cd target; cmake .. ; Use Visual Studio to build install/runtest project.

Cmake options (default): CPU_ONLY (OFF), USE_CUDNN (ON), USE_EIGEN (ON), DEBUG (OFF), USE_AVX2 (ON), USE_FMA (ON), USE_NEON (ON)

Test passed: all on Linux x86_64 with CUDA 8.0 and CuDNN 5.1, and CPU_ONLY + USE_EIGEN on Windows.

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
