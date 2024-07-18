# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# compile CUDA with /usr/local/cuda/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_DEFINES = -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -Dsimpleknn_EXPORTS

CUDA_INCLUDES = --options-file CMakeFiles/simpleknn.dir/includes_CUDA.rsp

CUDA_FLAGS =  -DONNX_NAMESPACE=onnx_c2 -gencode arch=compute_86,code=sm_86 -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=set_but_not_used,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl --expt-relaxed-constexpr --expt-extended-lambda -std=c++17 -Xcompiler=-fPIC   -w -D_GLIBCXX_USE_CXX11_ABI=0

CXX_DEFINES = -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -Dsimpleknn_EXPORTS

CXX_INCLUDES = -I/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/thirdparty/octree_map/. -isystem /home/zhujun/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/include -isystem /home/zhujun/anaconda3/envs/gpt/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include

CXX_FLAGS = -std=gnu++17 -fPIC   -w -D_GLIBCXX_USE_CXX11_ABI=0

