#!/bin/bash
cd src
../submodules/protobuf/build_linux_x86_64/bin/protoc caffe/proto/caffe.proto --cpp_out=.
mkdir -p ../include/caffe/proto
mv caffe/proto/caffe.pb.h ../include/caffe/proto
cd ..
