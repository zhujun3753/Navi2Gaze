import torch
import os
parent_dir = os.path.dirname(os.path.abspath(__file__))
# print("Load lib")
#* 加载函数
# torch.ops.load_library(parent_dir+"/build/liboctree_map.so")
# distCUDA2 = torch.ops.octree_map.distCUDA2
#* 加载类
torch.classes.load_library(parent_dir+"/build/liboctree_map.so")
OctreeMap = torch.classes.octree_map.OctreeMap()
# print("Load lib end")

OctreeMap.debug_print()