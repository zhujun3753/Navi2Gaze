# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import math
import os
import string
import sys
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import magnum as mn
import numpy as np
from magnum import shaders, text
from magnum.platform.glfw import Application

import habitat_sim
from habitat_sim import ReplayRenderer, ReplayRendererConfiguration, physics
from habitat_sim.logging import LoggingContext, logger
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.utils.settings import default_sim_settings, make_cfg
from typing import Dict, List, Tuple
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import torch
import argparse
# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
# sam
from segment_anything import sam_model_registry
from PIL import Image 
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog
from task_adapter.utils.visualizer import Visualizer
from task_adapter.semantic_sam.tasks.automatic_mask_generator import SemanticSamAutomaticMaskGenerator
from segment_anything import SamAutomaticMaskGenerator
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    mask_to_rle_pytorch,
    rle_to_mask,
    uncrop_masks,
)
from thirdparty.octree_map import *
import imageio
import tqdm
from tqdm import tqdm
# from vlmaps.utils.clip_mapping_utils import load_pose,   save_map,  depth2pc, transform_pc, get_sim_cam_mat,  project_point
from utils.clip_mapping_utils import load_pose,   save_map,  depth2pc, transform_pc, get_sim_cam_mat,  project_point

import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import threading
import time, sharedmem
from PIL import ImageDraw, ImageFont
import base64
import re
from openai import OpenAI
# from vlmaps.habitat_utils import *
from utils.habitat_utils import *

import random

# @markdown if the colab instance doesn't have GPU, untick the following checkbox
has_gpu = True # @param {type: "boolean"}
codec = "h264"
if has_gpu:
    codec = "h264_nvenc"

class MouseMode(Enum):
    LOOK = 0
    GRAB = 1
    MOTION = 2

class MouseGrabber:
    """
    Create a MouseGrabber from RigidConstraintSettings to manipulate objects.
    """

    def __init__(
        self,
        settings: physics.RigidConstraintSettings,
        grip_depth: float,
        sim: habitat_sim.simulator.Simulator,
    ) -> None:
        self.settings = settings
        self.simulator = sim

        # defines distance of the grip point from the camera for pivot updates
        self.grip_depth = grip_depth
        self.constraint_id = sim.create_rigid_constraint(settings)

    def __del__(self):
        self.remove_constraint()

    def remove_constraint(self) -> None:
        """
        Remove a rigid constraint by id.
        """
        self.simulator.remove_rigid_constraint(self.constraint_id)

    def updatePivot(self, pos: mn.Vector3) -> None:
        self.settings.pivot_b = pos
        self.simulator.update_rigid_constraint(self.constraint_id, self.settings)

    def update_frame(self, frame: mn.Matrix3x3) -> None:
        self.settings.frame_b = frame
        self.simulator.update_rigid_constraint(self.constraint_id, self.settings)

    def update_transform(self, transform: mn.Matrix4) -> None:
        self.settings.frame_b = transform.rotation()
        self.settings.pivot_b = transform.translation
        self.simulator.update_rigid_constraint(self.constraint_id, self.settings)

    def rotate_local_frame_by_global_angle_axis(
        self, axis: mn.Vector3, angle: mn.Rad
    ) -> None:
        """rotate the object's local constraint frame with a global angle axis input."""
        object_transform = mn.Matrix4()
        rom = self.simulator.get_rigid_object_manager()
        aom = self.simulator.get_articulated_object_manager()
        if rom.get_library_has_id(self.settings.object_id_a):
            object_transform = rom.get_object_by_id(
                self.settings.object_id_a
            ).transformation
        else:
            # must be an ao
            object_transform = (
                aom.get_object_by_id(self.settings.object_id_a)
                .get_link_scene_node(self.settings.link_id_a)
                .transformation
            )
        local_axis = object_transform.inverted().transform_vector(axis)
        R = mn.Matrix4.rotation(angle, local_axis.normalized())
        self.settings.frame_a = R.rotation().__matmul__(self.settings.frame_a)
        self.simulator.update_rigid_constraint(self.constraint_id, self.settings)

class Timer:
    """
    Timer class used to keep track of time between buffer swaps
    and guide the display frame rate.
    """

    start_time = 0.0
    prev_frame_time = 0.0
    prev_frame_duration = 0.0
    running = False

    @staticmethod
    def start() -> None:
        """
        Starts timer and resets previous frame time to the start time.
        """
        Timer.running = True
        Timer.start_time = time.time()
        Timer.prev_frame_time = Timer.start_time
        Timer.prev_frame_duration = 0.0

    @staticmethod
    def stop() -> None:
        """
        Stops timer and erases any previous time data, resetting the timer.
        """
        Timer.running = False
        Timer.start_time = 0.0
        Timer.prev_frame_time = 0.0
        Timer.prev_frame_duration = 0.0

    @staticmethod
    def next_frame() -> None:
        """
        Records previous frame duration and updates the previous frame timestamp
        to the cur_node time. If the timer is not currently running, perform nothing.
        """
        if not Timer.running:
            return
        Timer.prev_frame_duration = time.time() - Timer.prev_frame_time
        Timer.prev_frame_time = time.time()

def load_depth(depth_filepath):
    with open(depth_filepath, 'rb') as f:
        depth = np.load(f)
    return depth

def get_fast_video_writer(video_file: str, fps: int = 60):
    writer = imageio.get_writer(video_file, fps=fps)
    return writer

def create_video(data_dir: str, output_dir: str, fps: int = 30):
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    rgb_out_path = os.path.join(output_dir, "rgb.mp4")
    depth_out_path = os.path.join(output_dir, "depth.mp4")
    rgb_writer = get_fast_video_writer(rgb_out_path, fps=fps)
    depth_writer = get_fast_video_writer(depth_out_path, fps=fps)
    rgb_list = sorted(os.listdir(rgb_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    depth_list = sorted(os.listdir(depth_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
    depth_list = [os.path.join(depth_dir, x) for x in depth_list]
    pbar = tqdm(total=len(rgb_list), position=0, leave=True)
    for i, (rgb_path, depth_path) in enumerate(zip(rgb_list, depth_list)):
        # if i%5!=0: continue
        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = load_depth(depth_path)
        depth_vis = (depth / 10 * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        rgb_writer.append_data(rgb)
        depth_writer.append_data(depth_color)
        pbar.update(1)
    rgb_writer.close()
    depth_writer.close()

def create_lseg_map_batch(img_save_dir, camera_height, cs=0.05, gs=1000, depth_sample_rate=100):
    mask_version = 1 # 0, 1
    # loading models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading scene {img_save_dir}")
    rgb_dir = os.path.join(img_save_dir, "rgb")
    depth_dir = os.path.join(img_save_dir, "depth")
    pose_dir = os.path.join(img_save_dir, "pose")
    rgb_list = sorted(os.listdir(rgb_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    depth_list = sorted(os.listdir(depth_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    pose_list = sorted(os.listdir(pose_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
    depth_list = [os.path.join(depth_dir, x) for x in depth_list]
    pose_list = [os.path.join(pose_dir, x) for x in pose_list]
    map_save_dir = os.path.join(img_save_dir, "map")
    os.makedirs(map_save_dir, exist_ok=True)
    color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_{mask_version}.npy")
    obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")
    # initialize a grid with zero position at the center
    color_top_down_height = (camera_height + 1) * np.ones((gs, gs), dtype=np.float32)
    color_top_down = np.zeros((gs, gs, 3), dtype=np.uint8)
    obstacles = np.ones((gs, gs), dtype=np.uint8)
    tf_list = []
    pbar = tqdm(total=len(rgb_list))
    # load all images and depths and poses
    for data_sample in zip(rgb_list, depth_list, pose_list):
        rgb_path, depth_path, pose_path = data_sample
        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # read pose
        pos, rot = load_pose(pose_path)  # z backward, y upward, x to the right
        rot_ro_cam = np.eye(3)
        rot_ro_cam[1, 1] = -1
        rot_ro_cam[2, 2] = -1
        rot = rot @ rot_ro_cam
        pos[1] += camera_height
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos.reshape(-1)
        tf_list.append(pose)
        if len(tf_list) == 1:
            init_tf_inv = np.linalg.inv(tf_list[0]) 
        tf = init_tf_inv @ pose
        # read depth
        depth = load_depth(depth_path)
        pc, mask = depth2pc(depth)
        shuffle_mask = np.arange(pc.shape[1]) 
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        pc_global = transform_pc(pc, tf)
        rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])
        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            x = int(gs / 2 + int(p[0] / cs))
            y = int(gs / 2 - int(p[2] / cs))
            if x >= obstacles.shape[0] or y >= obstacles.shape[1] or x < 0 or y < 0 or p_local[1] < -0.5:
                continue
            rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
            rgb_v = rgb[rgb_py, rgb_px, :]
            if p_local[1] < color_top_down_height[y, x]:
                color_top_down[y, x] = rgb_v
                color_top_down_height[y, x] = p_local[1]
            if p_local[1] > camera_height-0.5:
                continue
            obstacles[y, x] = 0
        pbar.update(1)
    if 1:
        obstacles_pil = Image.fromarray(obstacles)
        plt.figure(figsize=(8, 6), dpi=120)
        plt.imshow(obstacles_pil, cmap='gray')
        # plt.show()
        # color_top_down = load_map(color_top_down_save_path)
        color_top_down = color_top_down
        color_top_down_pil = Image.fromarray(color_top_down)
        plt.figure(figsize=(8, 6), dpi=120)
        plt.imshow(color_top_down_pil)
        plt.show()
    save_map(color_top_down_save_path, color_top_down)
    save_map(obstacles_save_path, obstacles)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def num_image(path, num=0, x=10, y=10):
    labeled_path = path[:-4] + "_n"+path[-4:]
    if os.path.exists(labeled_path):
        # print("labeled_path exist!")
        return labeled_path
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    draw.text((x, y), str(num), font=ImageFont.truetype("arial.ttf", 40), fill=(255, 0, 0) )
    # Image.fromarray(np.asarray(image)) #* numpy 到 PIL
    # image.show()
    # import pdb;pdb.set_trace()
    # name_0 = name[:6] + "_n" +name[6:]
    width, height = image.size
    if width==1080 and height==720:
        image = image.resize((864,576))
    image.save(labeled_path)
    # return path + name_0
    return labeled_path

def show_img(img):
    if isinstance(img, np.ndarray):
        image=Image.fromarray(img)
    elif isinstance(img, str) and os.path.exists(img):
        image = Image.open(img)
    else:
        raise NotImplementedError
    image.show()

def extract_numbers(string):
    pattern = r'\d+'  # 匹配连续的数字
    numbers = re.findall(pattern, string)
    return [int(num) for num in numbers]

class BaseGrid: #* 均匀划分网格
    def __init__(self, center, radius, grid_length=0.2, ignore_axis = 1) -> None:
        self.center = center
        self.radius = radius
        self.grid_length = grid_length
        self.ignore_axis = ignore_axis
        self.ignored_axiz_value = center[ignore_axis]*1.0
        self.remaind_axis = [i for i in range(3) if i!=ignore_axis]
        self.center2d = center[self.remaind_axis]
        self.grid_size = int(np.ceil(radius/grid_length))*2
        self.start_pos = self.center2d-self.grid_size/2*grid_length
        self.end_pos = self.center2d+self.grid_size/2*grid_length
        self.grid_data = None
        self.dist2obstacle = radius*100
        self.dist2ground = radius*100
        self.move_dir = None
        self.label=-1

    def change_attr(self, center=None, radius=None, grid_length=None, ignore_axis = None):
        if center is not None: self.center = center
        if radius is not None: self.radius = radius
        if grid_length is not None: self.grid_length = grid_length
        if ignore_axis is not None: self.ignore_axis = ignore_axis
        self.ignored_axiz_value = self.center[self.ignore_axis]*1.0
        self.remaind_axis = [i for i in range(3) if i!=self.ignore_axis]
        self.center2d = self.center[self.remaind_axis]
        self.grid_size = int(np.ceil(self.radius/self.grid_length))*2
        self.start_pos = self.center2d-self.grid_size/2*self.grid_length
        self.end_pos = self.center2d+self.grid_size/2*self.grid_length
        self.grid_data = None

    def split_grid(self):
        self.grid_data = np.zeros((self.grid_size,self.grid_size, 3)) #* grid_center2d weight  网格中心和权重
        half_grid_radius = 0.5*self.grid_length
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid_data[i,j][0] = self.start_pos[0]+i*self.grid_length+half_grid_radius
                self.grid_data[i,j][1] = self.start_pos[1]+j*self.grid_length+half_grid_radius

    #* 获取大网格边界点云
    def get_bounding_box(self, weight_th=-2, pts_n_per_meter=1000, random_color=True):
        grid_start_pos = np.zeros((1, 3))
        grid_start_pos[:, self.ignore_axis] = self.ignored_axiz_value
        grid_start_pos[:, self.remaind_axis] = self.start_pos
        grid_len = 2*self.radius
        pts_n = int(grid_len*pts_n_per_meter)
        ones_ = np.linspace(0, 1, pts_n)*grid_len
        grid_box = np.zeros((pts_n*4, 3))
        grid_box[:, self.ignore_axis] = 0
        grid_box[0:pts_n, self.remaind_axis[1]] = ones_ #* x=0, y=...
        grid_box[pts_n:2*pts_n, self.remaind_axis[0]] = ones_ #* x=..., y=0
        grid_box[2*pts_n:3*pts_n, self.remaind_axis[0]] = grid_len #* x=len, y=...
        grid_box[2*pts_n:3*pts_n, self.remaind_axis[1]] = ones_ #* x=len, y=...
        grid_box[3*pts_n:4*pts_n, self.remaind_axis[0]] = ones_ #* x=..., y=len
        grid_box[3*pts_n:4*pts_n, self.remaind_axis[1]] = grid_len #* x=..., y=len
        all_grid_boxes_pts = grid_box + grid_start_pos
        all_grid_boxes_colors = np.zeros_like(all_grid_boxes_pts)
        # all_grid_boxes_colors[:,:] = np.random.rand(1, 3)
        all_grid_boxes_colors[:,:] = [1,1,1]
        grid_box_pc = o3d.geometry.PointCloud()
        grid_box_pc.points = o3d.utility.Vector3dVector(all_grid_boxes_pts)
        grid_box_pc.colors = o3d.utility.Vector3dVector(all_grid_boxes_colors)
        # o3d.io.write_point_cloud("/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/output/rendered/00009-vLpv2VX547B_basedata/near_datas/test.ply", grid_box_pc)
        # import pdb;pdb.set_trace()
        return grid_box_pc
    
    #* 获取大网格点云
    def get_bounding_box_colors(self, pts_n_per_meter=1000, random_color=True):
        grid_start_pos = np.zeros((1, 3))
        grid_start_pos[:, self.ignore_axis] = self.ignored_axiz_value
        grid_start_pos[:, self.remaind_axis] = self.start_pos
        grid_len = 2*self.radius
        pts_n = int(grid_len*pts_n_per_meter)
        ones_ = np.linspace(0, 1, pts_n)*grid_len
        xv, yv = np.meshgrid(ones_, ones_)
        grid_box = np.stack([xv, yv, np.ones_like(yv)],axis=2).reshape(-1,3)
        grid_box[:, self.ignore_axis] = 0
        grid_box[:, self.remaind_axis[0]] = xv.reshape(-1)
        grid_box[:, self.remaind_axis[1]] = yv.reshape(-1)
        all_grid_boxes_pts = grid_box + grid_start_pos
        all_grid_boxes_colors = np.zeros_like(all_grid_boxes_pts)
        all_grid_boxes_colors[:,:] = np.random.rand(1, 3)
        # all_grid_boxes_colors[:,:] = [1,1,1]
        grid_box_pc = o3d.geometry.PointCloud()
        grid_box_pc.points = o3d.utility.Vector3dVector(all_grid_boxes_pts)
        grid_box_pc.colors = o3d.utility.Vector3dVector(all_grid_boxes_colors)
        # o3d.io.write_point_cloud("/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/output/rendered/00009-vLpv2VX547B_basedata/near_datas/test.ply", grid_box_pc)

        # import pdb;pdb.set_trace()
        # grid_box = np.zeros((pts_n*4, 3))
        # grid_box[:, self.ignore_axis] = self.ignored_axiz_value
        # grid_box[0:pts_n, self.remaind_axis[1]] = ones_ #* x=0, y=...
        # grid_box[pts_n:2*pts_n, self.remaind_axis[0]] = ones_ #* x=..., y=0
        # grid_box[2*pts_n:3*pts_n, self.remaind_axis[0]] = grid_len #* x=len, y=...
        # grid_box[2*pts_n:3*pts_n, self.remaind_axis[1]] = ones_ #* x=len, y=...
        # grid_box[3*pts_n:4*pts_n, self.remaind_axis[0]] = ones_ #* x=..., y=len
        # grid_box[3*pts_n:4*pts_n, self.remaind_axis[1]] = grid_len #* x=..., y=len
        # all_grid_boxes_pts = grid_box + grid_start_pos
        # all_grid_boxes_colors = np.zeros_like(all_grid_boxes_pts)
        # # all_grid_boxes_colors[:,:] = np.random.rand(1, 3)
        # all_grid_boxes_colors[:,:] = [1,1,1]
        # grid_box_pc = o3d.geometry.PointCloud()
        # grid_box_pc.points = o3d.utility.Vector3dVector(all_grid_boxes_pts)
        # grid_box_pc.colors = o3d.utility.Vector3dVector(all_grid_boxes_colors)
        # # o3d.io.write_point_cloud("/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/output/rendered/00009-vLpv2VX547B_basedata/near_datas/test.ply", grid_box_pc)
        # import pdb;pdb.set_trace()
        return grid_box_pc

    def get_small_bounding_box(self, weight_th=-2, pts_n_per_meter=1000, random_color=True):
        if self.grid_data is None:
            return o3d.geometry.PointCloud()
        grid_data_ = self.grid_data.reshape(-1,3)
        grid_data_mask = grid_data_[:,-1]<=weight_th
        grid_data_sel = grid_data_[grid_data_mask, :]
        if len(grid_data_sel)<1:
            return o3d.geometry.PointCloud()
        grid_start_pos2d = grid_data_sel[:, :2] - self.grid_length*0.5
        grid_start_pos = np.zeros((len(grid_start_pos2d), 3))
        grid_start_pos[:, self.ignore_axis] = self.ignored_axiz_value
        grid_start_pos[:, self.remaind_axis] = grid_start_pos2d
        pts_n = int(self.grid_length*pts_n_per_meter)
        ones_ = np.linspace(0, 1, pts_n)*self.grid_length
        grid_box = np.zeros((pts_n*4, 3))
        grid_box[:, self.ignore_axis] = 0
        grid_box[0:pts_n, self.remaind_axis[1]] = ones_ #* x=0, y=...
        grid_box[pts_n:2*pts_n, self.remaind_axis[0]] = ones_ #* x=..., y=0
        grid_box[2*pts_n:3*pts_n, self.remaind_axis[0]] = self.grid_length #* x=len, y=...
        grid_box[2*pts_n:3*pts_n, self.remaind_axis[1]] = ones_ #* x=len, y=...
        grid_box[3*pts_n:4*pts_n, self.remaind_axis[0]] = ones_ #* x=..., y=len
        grid_box[3*pts_n:4*pts_n, self.remaind_axis[1]] = self.grid_length #* x=..., y=len
        all_grid_boxes = []
        all_grid_boxes_colors = []
        cmap = plt.cm.get_cmap('jet')
        weight_colors = cmap(grid_data_sel[:, -1])[:,:3]
        for i in range(len(grid_start_pos)):
            colors_ = np.ones_like(grid_box)
            if random_color:
                colors_[:,:] = np.random.rand(1, 3)
            else:
                colors_[:,:] = weight_colors[i,:3]
            all_grid_boxes.append(grid_start_pos[i:i+1, :]+grid_box)
            all_grid_boxes_colors.append(colors_)
        if len(all_grid_boxes)<1:
            import pdb;pdb.set_trace()
        all_grid_boxes_pts = np.vstack(all_grid_boxes)
        all_grid_boxes_colors = np.vstack(all_grid_boxes_colors)
        grid_box_pc = o3d.geometry.PointCloud()
        grid_box_pc.points = o3d.utility.Vector3dVector(all_grid_boxes_pts)
        grid_box_pc.colors = o3d.utility.Vector3dVector(all_grid_boxes_colors)
        # o3d.io.write_point_cloud("/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/output/rendered/00009-vLpv2VX547B_basedata/near_datas/test.ply", grid_box_pc)
        # import pdb;pdb.set_trace()
        return grid_box_pc

    #* 获取网格中心和权重
    def get_all_center_and_weights(self):
        if self.grid_data is None:
            self.split_grid()
        grid_data_ = self.grid_data.reshape(-1,3)
        results = np.zeros((len(grid_data_), 4))
        results[:, self.remaind_axis] = grid_data_[:,:2]
        results[:, 3] = grid_data_[:,2]
        results[:, self.ignore_axis] = self.ignored_axiz_value
        return results

    def get_obstacle_center_and_weights(self, weight_th=-2):
        if self.grid_data is None:
            return None
        grid_data_ = self.grid_data.reshape(-1,3)
        mask_ = grid_data_[:,2]<=weight_th
        if mask_.sum()<1:
            return None
        results = np.zeros((len(grid_data_), 4))
        results[:, self.remaind_axis] = grid_data_[:,:2]
        results[:, 3] = grid_data_[:,2]
        results[:, self.ignore_axis] = self.ignored_axiz_value
        return results[mask_, :]

    #* 获取单个点或者多个点对应的网格的中心和权重
    def get_center_and_weight(self, pos):
        if self.grid_data is None:
            self.split_grid()
        if len(pos.shape)==1:
            pos2d = pos[self.remaind_axis]
            pos_grid = np.round((pos2d - self.start_pos)/self.grid_length).astype(int)
            if pos_grid.min()<0 or pos_grid.max()>=self.grid_size:
                print("Beyond boundaries")
                return None
            grid_data_ = self.grid_data[pos_grid[0],pos_grid[1]]
            results = np.zeros((1, 4))
            results[:, self.remaind_axis] = grid_data_[:2]
            results[:, 3] = grid_data_[2]
            results[:, self.ignore_axis] = self.ignored_axiz_value
            return results
        pos2d = pos[:, self.remaind_axis]
        pos_grid = np.round((pos2d - self.start_pos)/self.grid_length).astype(int)
        if pos_grid.min()<0 or pos_grid.max()>=self.grid_size:
            print("Beyond boundaries")
            return None
        grid_data_ = self.grid_data[pos_grid[:,0],pos_grid[:,1]]
        results = np.zeros((len(grid_data_), 4))
        results[:, self.remaind_axis] = grid_data_[:,:2]
        results[:, 3] = grid_data_[:,2]
        results[:, self.ignore_axis] = self.ignored_axiz_value
        return results

class SmartCircle: #* 自适应圆圈，圆圈的中心可以调整！
    def __init__(self, center, radius, grid_length=0.2, ignore_axis = 1, min_grid_length=0.01) -> None:
        if center is not None: self.center = center
        if radius is not None: self.radius = radius
        if grid_length is not None: self.grid_length = grid_length
        if ignore_axis is not None: self.ignore_axis = ignore_axis
        if min_grid_length is not None: self.min_grid_length = min_grid_length
        self.min_dist2obstacle = grid_length/6
        self.ignored_axiz_value = self.center[self.ignore_axis]*1.0
        self.remaind_axis = [i for i in range(3) if i!=self.ignore_axis]
        self.center2d = self.center[self.remaind_axis]
        self.grid_size = int(np.ceil(self.radius/self.grid_length))*2
        self.start_pos = self.center2d-self.grid_size/2*self.grid_length
        self.end_pos = self.center2d+self.grid_size/2*self.grid_length
        self.grid_data = {}
        OctreeMap.ground_map_setup(torch.from_numpy(self.center).float(), self.radius, min_grid_length, ignore_axis)
        #* 将目标对象在地面的投影和其他物体的投影分开，沿着目标投影的周围建立圆圈
        #* 1. 首先将其他障碍物的在地面的投影点加入mao
        #* 2. 再将目标对象的投影点加入map
        #* 3. 再将可行地面点的投影点加入map
        #* 4. 确定对象的包络线，可通过简单的从中心点向外扩张的多条射线确定，这些射线两两之间的点不在目标对象投影点区域
        #* 5. 在包络线外围创建圆环，作为机器人可能的目标点
        #* 6. 圆环仅仅存在于已知可行的地面区域，未知区域和障碍区域不能创建
        pass

    def adaptive_move(self, weight_th=-2, scale=0.001):
        grid_box_pc = o3d.geometry.PointCloud()
        keys = tuple(self.grid_data.keys())
        centers = np.asarray(keys)
        obstacle_box_ids = []
        dirs = []
        candidate_keys = []
        new_centers = []
        if 1: # dir 设置
            dirs = [np.zeros(3) for v in range(4)]
            dirs[0][self.remaind_axis[0]] = 1
            dirs[1][self.remaind_axis[0]] = -1
            dirs[2][self.remaind_axis[1]] = 1
            dirs[3][self.remaind_axis[1]] = -1
        #* 先判断每个存在障碍大格子的移动方向 move_dir
        for i in range(len(keys)):
            grid_ = self.grid_data[keys[i]]
            if grid_.grid_data is None:
                continue
            dists = []
            for j in range(4):
                dist2obstacle_along_dir = OctreeMap.check_obstacle_along_dir(torch.from_numpy(centers[i]).float(), torch.from_numpy(dirs[j]).float())
                dists.append(dist2obstacle_along_dir)
            num_invalid_dir = 0
            dist_th = 0.5*self.grid_length+self.min_grid_length
            for j in range(len(dirs)):
                if dists[j]<dist_th:
                    num_invalid_dir += 1
            if num_invalid_dir>2:
                continue
            sel_dir_id = -1
            best_dist = dist_th
            for j in range(len(dirs)):
                if dists[j]<dist_th:
                    continue
                if j<2:
                    new_center = centers[i]+dirs[j]*dist_th
                    dist_1 = OctreeMap.check_obstacle_along_dir(torch.from_numpy(new_center).float(), torch.from_numpy(dirs[2]).float(), True)
                    dist_2 = OctreeMap.check_obstacle_along_dir(torch.from_numpy(new_center).float(), torch.from_numpy(dirs[3]).float(), True)
                    if dist_1>dist_th and dist_2>dist_th and min(dist_1, dist_2)>best_dist:
                        sel_dir_id = j
                        best_dist = min(dist_1, dist_2)
                else:
                    new_center = centers[i]+dirs[j]*dist_th
                    dist_1 = OctreeMap.check_obstacle_along_dir(torch.from_numpy(new_center).float(), torch.from_numpy(dirs[0]).float(), True)
                    dist_2 = OctreeMap.check_obstacle_along_dir(torch.from_numpy(new_center).float(), torch.from_numpy(dirs[1]).float(), True)
                    if dist_1>dist_th and dist_2>dist_th and min(dist_1, dist_2)>best_dist:
                        sel_dir_id = j
                        best_dist = min(dist_1, dist_2)
            for j in range(len(dirs)):
                if j!=sel_dir_id: continue
                length_ = dists[j]
                length_ = dist_th
                dir_=dirs[j]
                #* 确定方向之后就是考虑沿着这个方向前进的距离
                #* 首先沿着两边前进
                k = int(np.ceil(0.5*self.grid_length/self.min_grid_length))
                dist = 3*self.grid_length
                for ki in range(1, k):
                    d_ = 0.5*self.grid_length-ki*self.min_grid_length
                    if d_<=0: break
                    if j<2:
                        new_center1 = centers[i]+dirs[2]*d_
                        new_center2 = centers[i]+dirs[3]*d_
                    else:
                        new_center1 = centers[i]+dirs[0]*d_
                        new_center2 = centers[i]+dirs[1]*d_
                    dist_1 = OctreeMap.check_obstacle_along_dir(torch.from_numpy(new_center1).float(), torch.from_numpy(dir_).float(), False)
                    dist_2 = OctreeMap.check_obstacle_along_dir(torch.from_numpy(new_center2).float(), torch.from_numpy(dir_).float(), False)
                    if dist_1<dist: dist=dist_1
                    if dist_2<dist: dist=dist_2
                dist_0 = OctreeMap.check_obstacle_along_dir(torch.from_numpy(centers[i]).float(), torch.from_numpy(dir_).float(), False)
                if dist_0<dist: dist=dist_0
                if dist/self.grid_length<0.5:
                    del self.grid_data[keys[i]]
                    break
                #* 开始移动网格
                dist_1 = OctreeMap.check_obstacle_along_dir(torch.from_numpy(centers[i]).float(), torch.from_numpy(-dir_).float(), False)
                dist = (dist-dist_1)/2
                new_center = centers[i]+dir_*(dist-self.min_grid_length)
                candidate_keys.append(keys[i])
                new_centers.append(new_center)
                self.grid_data[tuple(new_center)] = BaseGrid(new_center, 0.5*self.grid_length, self.min_grid_length, self.ignore_axis)
                grid_ = self.grid_data[tuple(new_center)]
                if OctreeMap.check_bounding_box_valid(torch.from_numpy(new_center).float(), self.grid_length):
                    grid_.label = 2.0
                    OctreeMap.label_occupied_grid(torch.from_numpy(new_center).float(), self.grid_length, grid_.label)
                new_center_tensor = torch.from_numpy(new_center.reshape(1,-1)).float()
                results = OctreeMap.knn_nearest_search_obstacle(new_center_tensor, 1)
                dist = np.sqrt(results[1].cpu().numpy().squeeze())
                grid_.dist2obstacle = dist
                # print("dist: ",dist/self.grid_length)
                results = OctreeMap.knn_nearest_search_ground(new_center_tensor, 1)
                dist = np.sqrt(results[1].cpu().numpy().squeeze())
                grid_.dist2ground = dist
                del self.grid_data[keys[i]]
                #* 绘制移动方向
                # num_ = int(length_/scale)+1
                # tmp_ = np.linspace(0, length_, num_)
                # pts_ = tmp_.reshape(-1,1)@dir_.reshape(1, -1) + centers[i].reshape(1,-1)
                # colors_ = np.zeros_like(pts_)
                # # colors_[:,:] = np.random.rand(1, 3)
                # # if length_>0.5*self.grid_length:
                # colors_[:,0] = 1.0
                # grid_box_pc_ = o3d.geometry.PointCloud()
                # grid_box_pc_.points = o3d.utility.Vector3dVector(pts_)
                # grid_box_pc_.colors = o3d.utility.Vector3dVector(colors_)
                # grid_box_pc += grid_box_pc_
        # for i in range(len(new_centers)):
        #     s
        return grid_box_pc

    #* 获取大网格边界点云
    def get_bounding_box(self, weight_th=-2, pts_n_per_meter=1000, random_color=True):
        grid_box_pc = o3d.geometry.PointCloud()
        radius_ = self.grid_length/np.sqrt(2) #* 外接圆半径
        min_dist2obstacle_small_grid = self.min_grid_length*2 #* 小网格到障碍物的距离
        # print("min_dist2obstacle_small_grid: ", min_dist2obstacle_small_grid)
        for base_grid in self.grid_data.values():
            # print('base_grid.dist2ground: ', base_grid.dist2ground, " , base_grid.dist2obstacle: ", base_grid.dist2obstacle)
            # grid_box_pc += base_grid.get_bounding_box(weight_th, pts_n_per_meter, random_color)
            if base_grid.dist2ground>=min_dist2obstacle_small_grid: #* 到有效点的距离太远
                continue
            if base_grid.dist2obstacle>radius_:
                grid_box_pc += base_grid.get_bounding_box(weight_th, pts_n_per_meter, random_color)
            elif base_grid.dist2obstacle>self.min_dist2obstacle: #* 这个范围内的四网格需要考虑再划分
                grid_box_pc += base_grid.get_small_bounding_box(weight_th, pts_n_per_meter, random_color)
                grid_box_pc += base_grid.get_bounding_box(weight_th, pts_n_per_meter, random_color)
            else:
                pass
        # o3d.io.write_point_cloud("/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/output/rendered/00009-vLpv2VX547B_basedata/near_datas/test.ply", grid_box_pc)
        # import pdb;pdb.set_trace()
        return grid_box_pc

    #* 获取大网格边界点云,只获取label为2
    def get_bounding_box_with_label(self, weight_th=-2, pts_n_per_meter=1000, random_color=True, label=2.0):
        grid_box_pc = o3d.geometry.PointCloud()
        radius_ = self.grid_length/np.sqrt(2) #* 外接圆半径
        min_dist2obstacle_small_grid = self.min_grid_length*2 #* 小网格到障碍物的距离
        for base_grid in self.grid_data.values():
            if base_grid.label != label: #* 到有效点的距离太远
                continue
            grid_box_pc += base_grid.get_bounding_box(weight_th, pts_n_per_meter, random_color)
        # o3d.io.write_point_cloud("/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/output/rendered/00009-vLpv2VX547B_basedata/near_datas/test.ply", grid_box_pc)
        # import pdb;pdb.set_trace()
        return grid_box_pc

    #* 获取大网格边界点云,只获取label为2
    def get_bounding_box_with_label_colors(self, weight_th=-2, pts_n_per_meter=1000, random_color=True, label=2.0):
        centers = []
        grid_box_pcs = []
        for key in self.grid_data.keys():
        # for base_grid in self.grid_data.values():
            base_grid = self.grid_data[key]
            if base_grid.label != label: #* 到有效点的距离太远
                continue
            grid_box_pc = base_grid.get_bounding_box_colors(pts_n_per_meter, random_color)
            centers.append(key)
            grid_box_pcs.append(grid_box_pc)
        # o3d.io.write_point_cloud("/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/output/rendered/00009-vLpv2VX547B_basedata/near_datas/test.ply", grid_box_pc)
        # import pdb;pdb.set_trace()
        return grid_box_pcs, centers

    def get_ground_map(self):
        #* 空 未知点 蓝色
        #* 0 一般障碍点 黑色
        #* 1 可行地面点 绿色
        #* 2 目标点（同时也是障碍点） 红色
        map_pts_tensor = OctreeMap.get_ground_map()
        map_pts = map_pts_tensor.cpu().numpy()
        map_pts_pc = o3d.geometry.PointCloud()
        map_pts_pc.points = o3d.utility.Vector3dVector(map_pts[:,:3])
        map_pts_pc.colors = o3d.utility.Vector3dVector(map_pts[:,3:6])
        return map_pts_pc

    def update_object_envelope_circles(self):
        #* 空 未知点 蓝色
        #* 0 一般障碍点 黑色
        #* 1 可行地面点 绿色
        #* 2 目标点（同时也是障碍点） 红色
        map_pts_tensor, circle_center_tensor = OctreeMap.update_object_envelope_circles()
        circle_pts = map_pts_tensor.cpu().numpy()
        map_pts = circle_pts.reshape(-1, 6)
        map_pts_pc = o3d.geometry.PointCloud()
        map_pts_pc.points = o3d.utility.Vector3dVector(map_pts[:,:3])
        map_pts_pc.colors = o3d.utility.Vector3dVector(map_pts[:,3:6])
        return map_pts_pc, circle_pts, circle_center_tensor.cpu().numpy()
    
    def get_object_envelope_circles(self, radius):
        #* 空 未知点 蓝色
        #* 0 一般障碍点 黑色
        #* 1 可行地面点 绿色
        #* 2 目标点（同时也是障碍点） 红色
        map_pts_tensor, circle_center_tensor = OctreeMap.get_object_envelope_circles(radius)
        circle_pts = map_pts_tensor.cpu().numpy()
        map_pts = circle_pts.reshape(-1, 6)
        map_pts[:,1] += 0.05 #* 便于显示在上层
        map_pts_pc = o3d.geometry.PointCloud()
        map_pts_pc.points = o3d.utility.Vector3dVector(map_pts[:,:3])
        map_pts_pc.colors = o3d.utility.Vector3dVector(map_pts[:,3:6])
        return map_pts_pc, circle_pts, circle_center_tensor.cpu().numpy()

    def add_pts_to_ground_map(self, near_proj2ground_pts, attr=0.0):
        #* 空 未知点 蓝色
        #* 0 一般障碍点 黑色
        #* 1 可行地面点 绿色
        #* 2 目标点（同时也是障碍点） 红色
        OctreeMap.add_pts_to_ground_map(torch.from_numpy(near_proj2ground_pts).float(), attr)

    def calcu_dist2obstacle(self):
        #* 计算各个网格中心到障碍物的距离
        keys = tuple(self.grid_data.keys())
        centers = np.asarray(keys)
        centers_tensor = torch.from_numpy(centers).float()
        # import pdb;pdb.set_trace()
        results = OctreeMap.knn_nearest_search_obstacle(centers_tensor, 1)
        dist = np.sqrt(results[1].cpu().numpy().squeeze())
        # radius_ = self.grid_length/np.sqrt(2)
        # min_dist2obstacle_small_grid = self.min_grid_length*2 #* 小网格到障碍物的距离
        for i in range(len(keys)):
            grid_ = self.grid_data[keys[i]]
            grid_.dist2obstacle = dist[i]

    def calcu_dist2ground(self):
        keys = tuple(self.grid_data.keys())
        centers = np.asarray(keys)
        centers_tensor = torch.from_numpy(centers).float()
        results = OctreeMap.knn_nearest_search_ground(centers_tensor, 1)
        dist = np.sqrt(results[1].cpu().numpy().squeeze())
        for i in range(len(keys)):
            grid_ = self.grid_data[keys[i]]
            grid_.dist2ground = dist[i]

    def label_grid(self):
        #* 根据距离标记网格
        keys = tuple(self.grid_data.keys())
        centers = np.asarray(keys)
        # import pdb;pdb.set_trace()
        radius_ = self.grid_length/np.sqrt(2)
        min_dist2obstacle_small_grid = self.min_grid_length*2 #* 小网格到障碍物的距离
        for i in range(len(keys)):
            grid_ = self.grid_data[keys[i]]
            if grid_.label == 2.0: continue #* 已经是可行网格的不用标记
            #* 如果到障碍物的距离远，且边界都有效（不存在障碍物且未知的点数量小于8），则标记已经被占据的网格
            if grid_.dist2obstacle>radius_ and OctreeMap.check_bounding_box_valid(torch.from_numpy(centers[i]).float(), self.grid_length):
                grid_.label = 2.0
                OctreeMap.label_occupied_grid(torch.from_numpy(centers[i]).float(), self.grid_length, grid_.label)
            #* 如果网格中心距离地面点比较近且距离障碍物的距离在（1/6*grid_length，\sqrt(2)/2*grid_length）,则考虑大网格的每个小网格到障碍物的距离
            #* 如果每个小网格到障碍物的距离都大于最小网格长度的2倍，则标记为可行网格，否则将不满足的小网格标记为障碍小网格
            if grid_.dist2ground<min_dist2obstacle_small_grid and grid_.dist2obstacle<=radius_ and grid_.dist2obstacle>self.min_dist2obstacle:
                center_and_weights = grid_.get_all_center_and_weights()
                knn_results = OctreeMap.knn_nearest_search_obstacle(torch.from_numpy(center_and_weights[:,:3]).float(), 1)
                dist_ = np.sqrt(knn_results[1].cpu().numpy().squeeze())
                unsafe_mask = (dist_<min_dist2obstacle_small_grid).reshape(grid_.grid_size, grid_.grid_size)
                if unsafe_mask.sum()==0 and OctreeMap.check_bounding_box_valid(torch.from_numpy(centers[i]).float(), self.grid_length):
                    grid_.grid_data = None
                    grid_.label = 2.0
                    OctreeMap.label_occupied_grid(torch.from_numpy(centers[i]).float(), self.grid_length, grid_.label)
                else:
                    grid_.grid_data[unsafe_mask, -1] = -2
                pass

class HabitatSimInteractiveViewer(Application):
    # the maximum number of chars displayable in the app window
    # using the magnum text module. These chars are used to
    # display the CPU/GPU usage data
    MAX_DISPLAY_TEXT_CHARS = 256
    # how much to displace window text relative to the center of the
    # app window (e.g if you want the display text in the top left of
    # the app window, you will displace the text
    # window width * -TEXT_DELTA_FROM_CENTER in the x axis and
    # window height * TEXT_DELTA_FROM_CENTER in the y axis, as the text
    # position defaults to the middle of the app window)
    TEXT_DELTA_FROM_CENTER = 0.49
    # font size of the magnum in-window display text that displays
    # CPU and GPU usage info
    DISPLAY_FONT_SIZE = 16.0

    def __init__(self, sim_settings: Dict[str, Any]) -> None:
        self.sim_settings: Dict[str:Any] = sim_settings
        self.pc_all = o3d.geometry.PointCloud()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
        filename = sim_settings['scene'].split('/')[-1].split('.')[0]
        self.save_path = os.path.join('./output/rendered', filename)
        self.save_vlmap_path = os.path.join('./output/vlmaps_dataset', filename)
        self.basedata_path = os.path.join('./output/rendered', "common", filename) #* gt common vlmaps_dataset
        self.poses = []
        self.save_img_id = 0
        self.image_kuv1 = None
        self.cam_intrinsics = None
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        if not os.path.exists(self.save_vlmap_path): os.makedirs(self.save_vlmap_path)
        images_dir = os.path.join(self.save_path, 'images')
        depths_dir = os.path.join(self.save_path, 'depths')
        plys_dir = os.path.join(self.save_path, 'plys')
        self.save_video = False
        if self.save_video:
            video_rgb_dir = os.path.join(self.basedata_path, 'video', 'rgb')
            os.makedirs(video_rgb_dir, exist_ok=True)
            video_depth_dir = os.path.join(self.basedata_path, 'video', 'depth')
            os.makedirs(video_depth_dir, exist_ok=True)
            rgb_list_ = sorted([v for v in os.listdir(video_rgb_dir) if '.png' in v and  '_' not in v], key=lambda x: int(x.split("_")[-1].split(".")[0]))
            self.video_poses = []
            self.save_video_img_id = 0
            if os.path.exists(os.path.join(self.basedata_path, 'video', 'poses.txt')):
                poses_ = np.loadtxt(os.path.join(self.basedata_path, 'video', 'poses.txt')).reshape(-1,7)
                poses_ = [v for v in poses_]
                if len(poses_)==len(rgb_list_):
                    self.video_poses = poses_
                    self.save_video_img_id = len(poses_)
                    print("Start record image id: ", self.save_img_id)
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        if not os.path.exists(depths_dir):
            os.makedirs(depths_dir)
        if not os.path.exists(plys_dir):
            os.makedirs(plys_dir)
        if os.path.exists(os.path.join(self.save_path, 'poses.txt')):
            # import pdb;pdb.set_trace()
            rgb_list_ = sorted([v for v in os.listdir(images_dir) if '.png' in v and  '_' not in v], key=lambda x: int(x.split("_")[-1].split(".")[0]))
            poses_ = np.loadtxt(os.path.join(self.save_path, 'poses.txt')).reshape(-1,7)
            poses_ = [v for v in poses_]
            if len(poses_)==len(rgb_list_):
                self.poses = poses_
                self.save_img_id = len(poses_)
                print("Start record image id: ", self.save_img_id)
        if self.sim_settings['use_som']:
            self.init_som()
        # import pdb;pdb.set_trace()
        self.enable_batch_renderer: bool = self.sim_settings["enable_batch_renderer"]
        self.num_env: int = (self.sim_settings["num_environments"] if self.enable_batch_renderer else 1)
        # Compute environment camera resolution based on the number of environments to render in the window.
        window_size: mn.Vector2 = (self.sim_settings["window_width"], self.sim_settings["window_height"],)
        configuration = self.Configuration()
        configuration.title = "Habitat Sim Interactive Viewer"
        configuration.size = window_size
        Application.__init__(self, configuration)
        self.fps: float = 60.0
        # Compute environment camera resolution based on the number of environments to render in the window.
        grid_size: mn.Vector2i = ReplayRenderer.environment_grid_size(self.num_env)
        camera_resolution: mn.Vector2 = mn.Vector2(self.framebuffer_size) / mn.Vector2(grid_size)
        self.sim_settings["width"] = camera_resolution[0]
        self.sim_settings["height"] = camera_resolution[1]
        # self.test_map()
        # draw Bullet debug line visualizations (e.g. collision meshes)
        self.debug_bullet_draw = False
        # draw active contact point debug line visualizations
        self.contact_debug_draw = False
        # cache most recently loaded URDF file for quick-reload
        self.cached_urdf = ""
        # set up our movement map
        key = Application.KeyEvent.Key
        self.pressed = {
            key.UP: False,
            key.DOWN: False,
            key.LEFT: False,
            key.RIGHT: False,
            key.A: False,
            key.D: False,
            key.S: False,
            key.W: False,
            key.X: False,
            key.Z: False,
        }
        # set up our movement key bindings map
        key = Application.KeyEvent.Key
        self.key_to_action = {
            key.UP: "look_up",
            key.DOWN: "look_down",
            key.LEFT: "turn_left",
            key.RIGHT: "turn_right",
            key.A: "move_left",
            key.D: "move_right",
            key.S: "move_backward",
            key.W: "move_forward",
            key.X: "move_down",
            key.Z: "move_up",
        }
        # Load a TrueTypeFont plugin and open the font file
        self.display_font = text.FontManager().load_and_instantiate("TrueTypeFont")
        relative_path_to_font = "./utils/fonts/ProggyClean.ttf"
        self.display_font.open_file(os.path.join(os.path.dirname(__file__), relative_path_to_font),13,)
        # Glyphs we need to render everything
        self.glyph_cache = text.GlyphCache(mn.Vector2i(256))
        self.display_font.fill_glyph_cache(
            self.glyph_cache,
            string.ascii_lowercase
            + string.ascii_uppercase
            + string.digits
            + ":-_+,.! %µ",
        )
        # magnum text object that displays CPU/GPU usage data in the app window
        self.window_text = text.Renderer2D(
            self.display_font,
            self.glyph_cache,
            HabitatSimInteractiveViewer.DISPLAY_FONT_SIZE,
            text.Alignment.TOP_LEFT,
        )
        self.window_text.reserve(HabitatSimInteractiveViewer.MAX_DISPLAY_TEXT_CHARS)
        # text object transform in window space is Projection matrix times Translation Matrix
        # put text in top left of window
        self.window_text_transform = mn.Matrix3.projection(self.framebuffer_size) @ mn.Matrix3.translation(mn.Vector2(self.framebuffer_size)
        * mn.Vector2(-HabitatSimInteractiveViewer.TEXT_DELTA_FROM_CENTER,HabitatSimInteractiveViewer.TEXT_DELTA_FROM_CENTER,))
        self.shader = shaders.VectorGL2D()
        # make magnum text background transparent
        mn.gl.Renderer.enable(mn.gl.Renderer.Feature.BLENDING)
        mn.gl.Renderer.set_blend_function(
            mn.gl.Renderer.BlendFunction.ONE,
            mn.gl.Renderer.BlendFunction.ONE_MINUS_SOURCE_ALPHA,
        )
        mn.gl.Renderer.set_blend_equation(mn.gl.Renderer.BlendEquation.ADD, mn.gl.Renderer.BlendEquation.ADD)
        # variables that track app data and CPU/GPU usage
        self.num_frames_to_track = 60
        # Cycle mouse utilities
        self.mouse_interaction = MouseMode.LOOK
        self.mouse_grabber: Optional[MouseGrabber] = None
        self.previous_mouse_point = None
        # toggle physics simulation on/off
        self.simulating = True
        # toggle a single simulation step at the next opportunity if not
        # simulating continuously.
        self.simulate_single_step = False
        # configure our simulator
        self.cfg: Optional[habitat_sim.simulator.Configuration] = None
        self.sim: Optional[habitat_sim.simulator.Simulator] = None
        self.tiled_sims: list[habitat_sim.simulator.Simulator] = None
        self.replay_renderer_cfg: Optional[ReplayRendererConfiguration] = None
        self.replay_renderer: Optional[ReplayRenderer] = None
        self.reconfigure_sim()
        self.pathfinder = self.sim.pathfinder
        # compute NavMesh if not already loaded by the scene.
        if (not self.pathfinder.is_loaded and self.cfg.sim_cfg.scene_id.lower() != "none"):
            self.navmesh_config_and_recompute()
        self.time_since_last_simulation = 0.0
        LoggingContext.reinitialize_from_env()
        logger.setLevel("INFO")
        self.print_help_text()
        self.poses_pts = []
        self.init_pose = False
        self.threads_end = False
        self.last_ext = None
        self.extrinsic_inv = None
        self.enter_press = False
        self.obj_pc_w = None #* 目标对象的点云
        self.ground_height = None #* 地面高度
        # self.top_down_view_show_thread = threading.Thread(target=self.top_down_view_show_func)
        self.top_down_view_update_thread = threading.Thread(target=self.top_down_view_update_func)
        # self.top_down_view_show_thread.start()
        self.top_down_view_update_thread.start()
        # self.get_cur_top_down_view()
        width = int(self.sim_settings["width"])
        height = int(self.sim_settings["height"])
        self.image_in = sharedmem.empty([height, width,3], np.uint8)
        self.image_in[:] = (np.random.random((height, width,3))*255).astype(np.uint8)
        self.cam_pts = sharedmem.empty([400, 5], np.float32) #* [x,y,r,g,b]
        # self.cam_pts[:] = np.random.random((400, 5))
        self.top_down_view_update_flag = sharedmem.empty([1], bool)
        # import pdb;pdb.set_trace()
        self.top_down_view_update_flag[:] = False
        # self.top_down_view_update_flag.sum()==0
        # self.process = multiprocessing.Process(target=self.top_down_view_show_func, args=(self.image_in, self.cam_pts, self.top_down_view_update_flag))
        # self.process.start()
        self.initialize_pose()
        navigation_target = []
        self.navigation_targets = navigation_target
        self.navigation_target_id = 0
        self.obj2cls = None
        self.vlmap_agent_states = []
        if 0: #* 评估
            # self.semexp_data = np.load("output/rendered/common/"+filename+"/SemExp.npy", allow_pickle=True) #* Object-Goal-Navigation  L3MVN SemExp
            # self.l3mvn_data = np.load("output/rendered/common/"+filename+"/L3MVN.npy", allow_pickle=True) #* Object-Goal-Navigation  L3MVN
            # self.vlmaps_data = np.load("output/rendered/common/"+filename+"/vlmaps.npy", allow_pickle=True) #* 暂时借用
            self.navi2gaze_data = np.load("output/rendered/common/"+filename+"/navi2gaze.npy", allow_pickle=True)
            self.navi2gaze0_data = np.load("output/rendered/common/"+filename+"/navi2gaze_DNT.npy", allow_pickle=True)
            self.navi2gaze1_data = np.load("output/rendered/common/"+filename+"/navi2gaze-OGD.npy", allow_pickle=True)
            self.navi2gaze2_data = np.load("output/rendered/common/"+filename+"/navi2gaze_wRTS.npy", allow_pickle=True)
            self.target_data = np.load("output/rendered/common/"+filename+"/target_data.npy", allow_pickle=True) #* 暂时借用
            vis = True
            # target_names_basic = ['plant', 'chair', 'toilet', 'sofa', 'tv_monitor', 'bed']
            target_names_basic = ['sofa', 'tv_monitor', 'bed'] #* 当前所考虑的对象种类
            #* 计算目标对象对应的ids
            target_names = list(set([v["name"] for v in self.target_data]))
            target_name2id = {}
            for name_ in target_names:
                target_name2id[name_] = []
            for i in range(len(self.target_data)):
                target_name2id[self.target_data[i]["name"]].append(i)
            #* 遍历测量数据
            # import pdb;pdb.set_trace()
            # to_eval_datas = {'SemExp':self.semexp_data, 'L3MVN':self.l3mvn_data, 'vlmaps':self.vlmaps_data, 'navi2gaze':self.navi2gaze_data}
            to_eval_datas = {'navi2gaze':self.navi2gaze_data, 'navi2gaze_DNT':self.navi2gaze0_data, 'navi2gaze_OGD':self.navi2gaze1_data, 'navi2gaze_wRTS':self.navi2gaze2_data}
            record_data = {}
            for alg_name in to_eval_datas.keys():
                record_data[alg_name]={"SR":[], "shortest_path":[], "cur_path":[],  "DTG":[], "OTG":[]} #* 失败时候值为-1
                cur_eval_data = to_eval_datas[alg_name]
                goal_name2id = {}
                goal_names = list(set([v["goal_name"] for v in cur_eval_data]))
                for name_ in goal_names:
                    goal_name2id[name_] = []
                for i in range(len(cur_eval_data)):
                    goal_name2id[cur_eval_data[i]["goal_name"]].append(i)
                #* 逐步考虑当前算法所搜索的每一个类别
                for name_ in goal_names:
                    if name_ not in target_names_basic or name_ not in target_names:
                        continue
                    goal_ids = goal_name2id[name_]
                    if len(goal_ids)<1:
                        continue
                    for goal_id in goal_ids:
                        cur_data = cur_eval_data[goal_id]
                        goal_pos = cur_data['agent_state'][:3]*1.0
                        # import pdb;pdb.set_trace()
                        if alg_name in "navi2gaze":
                            #* 因为有旋转，而且采集的是相机的位姿，所以还要沿着y轴减1.5
                            R0 = Rotation.from_quat(np.asarray(cur_data['agent_state'][3:])).as_matrix()
                            y_dir = R0[:3,1]
                            goal_pos -= y_dir*(self.sim_settings["sensor_height"]-0.88*0.0) #* navi2gaze_data
                            # position -= y_dir*(self.sim_settings["sensor_height"]*0.0-0.88*0.0) #* vlmaps_data l3mvn_data semexp_data
                        target_ids = target_name2id[name_]
                        record_data[alg_name]["SR"].append(cur_data['success'])
                        if cur_data['success']!=1:
                            record_data[alg_name]["shortest_path"].append(-1)
                            record_data[alg_name]["cur_path"].append(-1)
                            record_data[alg_name]["DTG"].append(-1)
                            record_data[alg_name]["OTG"].append(-1)
                        else:
                            best_dist = 1000.0
                            best_angle = 0
                            shortest_path = self.path_plan(cur_data["init_agent_state"], cur_data['agent_state']) #* 最短路径
                            for target_id in target_ids:
                                optimal_target_pos = self.target_data[target_id]['optimal_pos'] #* 最优终点
                                target_center = self.target_data[target_id]['center'] # 对象中心
                                dist_tmp = cur_data['agent_state'][:3]-optimal_target_pos
                                dist_tmp[1] = 0
                                dist = np.linalg.norm(dist_tmp) #* 水平距离
                                dir1 = cur_data['agent_state'][:3]-target_center
                                dir2 = optimal_target_pos-target_center
                                dir1[1]=dir2[1]=0
                                angle = np.arccos(dir1.dot(dir2)/np.linalg.norm(dir2)/np.linalg.norm(dir1))/np.pi*180 #* 水平夹角
                                # import pdb;pdb.set_trace()
                                if dist<best_dist:
                                    best_dist = dist
                                    best_angle = angle
                            record_data[alg_name]["shortest_path"].append(shortest_path)
                            record_data[alg_name]["cur_path"].append(cur_data["path_length"])
                            record_data[alg_name]["DTG"].append(best_dist)
                            record_data[alg_name]["OTG"].append(best_angle)
                            print("alg_name: ",alg_name, "goal_name: ", name_, ", best_dist: ", best_dist, ", best_angle: ", best_angle)
                            if vis and alg_name in "navi2gaze" and 1 and shortest_path==0:
                                # agent_state = habitat_sim.AgentState()
                                # agent_state.position = goal_pos
                                # agent_state.rotation = cur_data['agent_state'][3:]*1.0
                                # self.sim.get_agent(0).set_state(agent_state)
                                # self.draw_event()
                                print(record_data[alg_name])
                                import pdb;pdb.set_trace()
            np.save("output/rendered/common/"+filename+"/record_data0.npy", record_data)
            record_data = np.load("output/rendered/common/"+filename+"/record_data0.npy", allow_pickle=True).tolist()
            import pdb;pdb.set_trace()
            pass
        self.l3mvn_data_id = 0
        # self.l3mvn_data = self.navi2gaze_data #* 临时
        self.goal_names_basic = ['plant', 'chair', 'toilet', 'sofa', 'tv_monitor', 'bed']
        self.goal_names_basic_id = 0
        self.eval_data =[]
        self.init_agent_state = None
        self.path_length = 0
        # import pdb;pdb.set_trace()
        self.get_random_navigable_points = []
        self.get_random_navigable_points_id = 0
        if os.path.exists(os.path.join(self.save_path, 'navigable_points.txt')):
            self.get_random_navigable_points = np.loadtxt(os.path.join(self.save_path, 'navigable_points.txt'))
        if 0:
            goal_name2id = {}
            goal_names = list(set([v["goal_name"] for v in self.l3mvn_data]))
            for name_ in goal_names:
                goal_name2id[name_] = []
            for i in range(len(self.l3mvn_data)):
                goal_name2id[self.l3mvn_data[i]["goal_name"]].append(i)
            print([(key_, len(goal_name2id[key_]))for key_ in goal_name2id.keys()])
            # for key_ in goal_name2id.keys():
            #     print(key_, ", ", len(goal_name2id[key_]))
            self.goal_name2id = goal_name2id
            target_names = list(set([v["name"] for v in self.target_data]))
            target_name2id = {}
            for name_ in target_names:
                target_name2id[name_] = []
            for i in range(len(self.target_data)):
                target_name2id[self.target_data[i]["name"]].append(i)
            print([(key_, len(target_name2id[key_]))for key_ in target_name2id.keys()])
            # for key_ in target_name2id.keys():
            #     print(key_, ", ", len(target_name2id[key_]))
            self.target_name2id = target_name2id
        # {'plant', 'chair', 'toilet', 'sofa', 'tv_monitor', 'bed'}
        if 0: #* 结果保存
            alg_names = ['SemExp', 'L3MVN', 'vlmaps', 'navi2gaze']
            algo_datas = {}
            for alg_name in alg_names:
                data_path_ = os.path.join("output/rendered/common", filename, alg_name+".npy")
                if os.path.exists(data_path_):
                    algo_datas[alg_name] = np.load(data_path_, allow_pickle=True).tolist()
            target_data = np.load(os.path.join("output/rendered/common", filename, "target_data.npy"), allow_pickle=True).tolist()
            save_imgs_dir = os.path.join("output/rendered/common", "final_images" ,filename)
            if not os.path.exists(save_imgs_dir):
                os.makedirs(save_imgs_dir)
            #* 计算目标对象对应的ids
            target_names = list(set([v["name"] for v in target_data]))
            target_name2id = {}
            for name_ in target_names:
                target_name2id[name_] = []
            for i in range(len(target_data)):
                target_name2id[target_data[i]["name"]].append(i)
            #* 遍历测量数据
            record_data = {}
            for alg_name in algo_datas.keys():
                cur_eval_data = algo_datas[alg_name]
                goal_name2id = {}
                goal_names = list(set([v["goal_name"] for v in cur_eval_data]))
                for name_ in goal_names:
                    goal_name2id[name_] = []
                for i in range(len(cur_eval_data)):
                    goal_name2id[cur_eval_data[i]["goal_name"]].append(i)
                #* 逐步考虑当前算法所搜索的每一个类别
                for name_ in goal_names:
                    if name_ not in target_names:
                        continue
                    goal_ids = goal_name2id[name_] #* 目标在当前算法中的ids
                    if len(goal_ids)<1:
                        continue
                    #* 选择出最佳id以保存图片
                    num_ = 0
                    for goal_id in goal_ids:
                        cur_data = cur_eval_data[goal_id]
                        if cur_data['success']!=1:
                            continue
                        goal_pos = cur_data['agent_state'][:3]*1.0
                        # import pdb;pdb.set_trace()
                        if alg_name in "navi2gaze":
                            #* 因为有旋转，而且采集的是相机的位姿，所以还要沿着y轴减1.5
                            R0 = Rotation.from_quat(np.asarray(cur_data['agent_state'][3:])).as_matrix()
                            y_dir = R0[:3,1]
                            goal_pos -= y_dir*(self.sim_settings["sensor_height"]) #* navi2gaze_data
                            # position -= y_dir*(self.sim_settings["sensor_height"]*0.0-0.88*0.0) #* vlmaps_data l3mvn_data semexp_data
                        target_ids = target_name2id[name_]
                        best_dist = 1000.0
                        best_angle = 0
                        best_id = goal_ids[0]
                        best_target_id = 0
                        for target_id in target_ids:
                            optimal_target_pos = target_data[target_id]['optimal_pos'] #* 最优终点
                            target_center = target_data[target_id]['center'] # 对象中心
                            dist_tmp = cur_data['agent_state'][:3]-optimal_target_pos
                            dist_tmp[1] = 0
                            dist = np.linalg.norm(dist_tmp) #* 水平距离
                            dir1 = cur_data['agent_state'][:3]-target_center
                            dir2 = optimal_target_pos-target_center
                            dir1[1]=dir2[1]=0
                            angle = np.arccos(dir1.dot(dir2)/np.linalg.norm(dir2)/np.linalg.norm(dir1))/np.pi*180 #* 水平夹角
                            # import pdb;pdb.set_trace()
                            if dist<best_dist:
                                best_dist = dist
                                best_angle = angle
                                best_id = goal_id
                                best_target_id = target_id
                        if best_dist<1.0:
                            target_center = target_data[best_target_id]['center']
                            best_data = cur_eval_data[best_id]
                            goal_pos = best_data['agent_state'][:3]*1.0
                            if alg_name in "navi2gaze":
                                #* 因为有旋转，而且采集的是相机的位姿，所以还要沿着y轴减1.5
                                R0 = Rotation.from_quat(np.asarray(best_data['agent_state'][3:])).as_matrix()
                                y_dir = R0[:3,1]
                                goal_pos -= y_dir*(self.sim_settings["sensor_height"]) #* navi2gaze_data
                            agent_state = habitat_sim.AgentState()
                            agent_state.position = goal_pos
                            agent_state.rotation = best_data['agent_state'][3:]*1.0
                            self.sim.get_agent(0).set_state(agent_state)
                            self.draw_event()
                            self.gaze_target(target_center, 15)
                            obs = self.sim.get_sensor_observations(0)
                            cv2.imwrite(os.path.join(save_imgs_dir, name_+"_"+alg_name+str(best_id).zfill(2)+'.png'), obs['color_sensor'][:,:,[2,1,0]])
                            print("Goal name: ", name_, ", algo name: ", alg_name)
                            num_+=1
                            if num_>5:
                                break
                            # import pdb;pdb.set_trace()
                # import pdb;pdb.set_trace()
            import pdb;pdb.set_trace()
            # prints
            pass
        if 1: #* navi2gaze
            parsed_results = self.parse_object_goal_instruction("sit on the sofa")
            # parsed_results = self.parse_object_goal_instruction("open the fridge")
            # parsed_results = self.parse_object_goal_instruction("go to my bedside table")
            basic_test = eval(parsed_results)
            if not isinstance(basic_test[0], List):
                basic_test = [basic_test]
            navigable_points = np.loadtxt("output/rendered/common/"+filename+"/navigable_points.txt")
            # import pdb;pdb.set_trace()
            self.eval_data =[]
            if os.path.exists(os.path.join(self.basedata_path, "navi2gaze0.npy")):
                self.eval_data = np.load(os.path.join(self.basedata_path, "navi2gaze0.npy"), allow_pickle=True).tolist()
            # import pdb;pdb.set_trace()
            candi_ids = [v for v in range(len(navigable_points))]
            random.shuffle(candi_ids)
            # [49, 25, 39, 46, 44, 20, 22, 42, 15, 43, 40, 4, 24, 5, 8, 17, 41, 36, 37, 19, 45, 6, 38, 26, 2, 0, 12, 16, 11, 9, 14, 27, 28, 1, 31, 32, 29, 7, 18, 47, 34, 30, 21, 35, 3, 10, 48, 13, 33, 23]
            # import pdb;pdb.set_trace()
            for bt in basic_test:
                for i in range(len(candi_ids)):
                    print("i: ",i,"len(candi_ids): ",len(candi_ids))
                    # if i!=13: continue
                    # npi = candi_ids[i]
                    npi = i
                    self.path_length = 0
                    #* 设置初始位姿
                    agent_state = habitat_sim.AgentState()
                    agent_state.position = navigable_points[npi,:3]*1.0
                    agent_state.position[1] -= self.sim_settings["sensor_height"]
                    agent_state.rotation = navigable_points[npi,3:]*1.0
                    # agent_state.position = self.pathfinder.get_random_navigable_point()
                    self.sim.get_agent(0).set_state(agent_state)
                    self.draw_event()
                    self.init_agent_state = self.get_state_np()
                    print("self.init_agent_state: ", self.init_agent_state)
                    # import pdb;pdb.set_trace()
                    success = self.gpt_test(object_name=bt[0], exe=bt[1], image_id = 1, mask_id = 4, use_gpt = True, circle_id=5)
                    #* return 1 成功， 0 GPT失败 -1 导航失败 -2 其他
                    def save_data(goal_name, success):
                        cur_data = {}
                        cur_data['goal_name'] = goal_name
                        cur_data["init_agent_state"] = self.init_agent_state #* 起始状态
                        cur_data["agent_state"] = self.get_state_np() #* 当前状态
                        cur_data["path_length"] = self.path_length
                        cur_data["success"] = success
                        self.eval_data.append(cur_data)
                        return self.eval_data
                    save_data(bt[0], success)
                    print("Goal_name: ", bt[0], "success: ", success, "path_length: ", self.path_length)
                    import pdb;pdb.set_trace()
                    np.save(os.path.join(self.basedata_path, "navi2gaze0.npy"), self.eval_data)
            print("==============navi2gaze end!!======")
        # import pdb;pdb.set_trace()
        if 0: #* 构建gt
            for i in range(20):
                agent_state = habitat_sim.AgentState()
                itn = 0
                while True:
                    agent_state.position = self.pathfinder.get_random_navigable_point()
                    itn += 1
                    if abs(agent_state.position[1]-0.1)<0.2:
                        break
                    if itn>50:
                        exit(1)
                self.sim.get_agent(0).set_state(agent_state)
                self.draw_event()
                self.gpt_test(object_name='sofa', exe="sit on the sofa", image_id = 12, mask_id = 2, use_gpt = False, circle_id=5)
        # obj_center = self.img_and_mask_id_to_pos(4,4)
        # self.navigate_to_target(obj_center)
        # import pdb;pdb.set_trace()
        pass

    def parse_object_goal_instruction(self, language_instr):
        """
        Parse language instruction into a series of landmarks
        Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
        """
        # openai_key = os.environ["OPENAI_KEY"]
        # openai.api_key = openai_key
        question = f"""
        I: go to the kitchen. A: ['kitchen', 'go to the kitchen']
        I: go to the chair. A: ['chair', 'go to the chair']
        I: sit on the sofa. A: ['sofa', 'sit on the sofa']
        I: navigate to the green sofa, then go to the painting. A: [['green sofa', 'navigate to the sofa'], ['painting'], 'go to the painting']
        I: approach the window in front, turn right and go to the television. A: [['window', 'approach the window'], ['television'], 'go to the television']
        I: {language_instr}. A:"""
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
        completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": question,}],
                    model="gpt-4-vision-preview",    # 或者其他模型
                    max_tokens=300,)
        result = completion.choices[0].message.content  # 获取回复内容
        print("result: ", result)
        return result

    def gpt_test(self, object_name='sofa', exe="sit on the sofa", image_id = 1, mask_id = 3, use_gpt = False, circle_id=5, navigatoin_test=False, direct_to_pos=False, no_recon=False, one_step=False):
        #* return 1 成功， 0 GPT失败 -1 导航失败 -2 其他
        print("Object: ", object_name, ", action: ", exe)
        info_record = {}
        info_record['object_name'] = object_name
        info_record['action'] = exe
        img_dir = self.basedata_path
        # print(f"loading scene {img_dir}")
        rgb_dir = os.path.join(img_dir, "images")
        if not os.path.exists(rgb_dir):
            print("There is no file !!!")
            return -2 #* return 1 成功， 0 GPT失败 -1 导航失败 -2 其他
        rgb_list = sorted([v for v in os.listdir(rgb_dir) if '.png' in v and  '_' not in v], key=lambda x: int(x.split("_")[-1].split(".")[0]))
        rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
        img_n = len(rgb_list)
        # import pdb;pdb.set_trace()
        # image_id, mask_id = 7,4
        # mask_id = 3
        # use_gpt = False # * False  True
        # print('============= Select Image =================')
        if navigatoin_test:
            use_gpt = False
            for image_id in range(img_n):
                target_pos = self.database_poses[image_id,:3]*1.0
                target_pos[1] -= self.sim_settings["sensor_height"]
                # result = self.navigate_to_target(tartget_pos, 0.2, time_sleep=0.01)
                if 1:
                    # print("Navigation fails!")
                    # state_np = self.get_state_np()
                    # is_navigable = self.pathfinder.is_navigable(state_np[:3])
                    # if not is_navigable:
                    #     iter_n = 0
                    #     while not is_navigable:
                    #         navigable_pos = self.get_near_navigable_pos(state_np[:3])
                    #         is_navigable = self.pathfinder.is_navigable(navigable_pos)
                    #         iter_n+=1
                    #     import pdb;pdb.set_trace()
                    #     print("cur iter_n: ", iter_n)
                    is_navigable = self.pathfinder.is_navigable(target_pos)
                    if not is_navigable:
                        iter_n = 0
                        while not is_navigable:
                            navigable_pos = self.get_near_navigable_pos(target_pos)
                            is_navigable = self.pathfinder.is_navigable(navigable_pos)
                            iter_n+=1
                            if iter_n>100:
                                break
                        # import pdb;pdb.set_trace()
                        print("goal iter_n: ", iter_n)
                        print("goal is_navigable: ", is_navigable)
                        # import pdb;pdb.set_trace()
                        result = self.navigate_to_target(navigable_pos, 0.2, time_sleep=0.01)
                    else:
                        result = self.navigate_to_target(target_pos, 0.2, time_sleep=0.01)
                    if not result:
                        print("Navigation fails again!")
                        import pdb;pdb.set_trace()
            return 1
        if use_gpt:
            user_input_select_image = f"Among these {img_n} images, each labeled with a red number in the upper left corner, could you identify which one(s) feature a {object_name} and provide the picture number(s)? Then in the picture(s) with the {object_name}, please choose a clearest picture, please directly tell me the number of the picture you selected at last.  Please output only one number to me. If you can not find, return None."
            input_content = [{"type": "text","text": user_input_select_image}]
            for i in range(img_n):
                if i>9: break
                labeled_img_path = num_image(rgb_list[i],i)
                base64_image = encode_image(labeled_img_path)
                # show_img(labeled_img)
                # import pdb;pdb.set_trace()
                input_content.append({"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},})
            reply = "one labeled number 1."
            # import pdb;pdb.set_trace()
            print("Aking GPT")
            completion = self.client.chat.completions.create(messages=[{"role": "user", "content": input_content,}],model="gpt-4-vision-preview",max_tokens=300,)
            print("Get results")
            reply = completion.choices[0].message.content  # 获取回复内容
            info_record['select_image_prompt'] = user_input_select_image
            info_record['select_image_result'] = reply
            print("reply: ", reply)
            extract_numbers_ = extract_numbers(reply)
            if len(extract_numbers_)==0:
                print(f"GPT can not find the {object_name}")
                return 0 #* return 1 成功， 0 GPT失败 -1 导航失败 -2 其他
            image_id = extract_numbers_[0]
        print("The selected image id is ", image_id)
        #* 导航到位置
        target_pos = self.database_poses[image_id,:3]*1.0
        target_pos[1] -= self.sim_settings["sensor_height"]
        result = self.navigate_to_target(target_pos, 0.2, time_sleep=0.001)
        if not result:
            print("Navigation fails! Return")
            return -1 #* return 1 成功， 0 GPT失败 -1 导航失败 -2 其他
        # import pdb;pdb.set_trace()
        #* 计算相机坐标系的z轴在世界坐标系下的坐标
        cam_z_dir_w = ((Rotation.from_quat(np.asarray(self.database_poses[image_id,3:])).as_matrix()@self.rot_ro_cam)@np.array([0,0,1]).reshape(3,1)).reshape(-1)
        # print("cam_z_dir_w:\n ",cam_z_dir_w)
        self.gaze_target(self.get_agent_pose_euler()[:3]+cam_z_dir_w*1, 15)
        seg_mask_path = rgb_list[image_id][:-4]+ "_seg" + rgb_list[image_id][-4:]
        if not os.path.exists(seg_mask_path):
            if not self.sim_settings['use_som']:
                self.sim_settings['use_som'] = True
                self.init_som()
            # img_out, mask_map = self.img_seg(np.asarray(Image.open(rgb_list[image_id]))/255.0)
            img_out, mask_map = self.img_seg(np.asarray(Image.open(rgb_list[image_id]))/255.0, slider=1.2, alpha=0.3)
            cv2.imwrite(os.path.join(self.save_path, 'images', str(image_id).zfill(6)+'_seg.png'), img_out[:,:,[2,1,0]])
            np.save(os.path.join(self.save_path, 'images', str(image_id).zfill(6)+'_mask.npy'), mask_map[:,:,0])
            import pdb;pdb.set_trace()
        if use_gpt:
            base64_image_mask = encode_image(seg_mask_path)
            base64_image_rgb = encode_image(rgb_list[image_id])
            user_input_select_seg_mask = f"Please find the {object_name} in the first image and tell me the corresponding number on it in the second image, the number is best placed on {object_name}.  Please output only one number to me"
            print("Asking GPT")
            completion2 = self.client.chat.completions.create(
                messages=[{"role": "user","content": [{"type": "text","text": user_input_select_seg_mask}, {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image_rgb}"},}, {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image_mask}"},},],}],
                model="gpt-4-vision-preview",    # 或者其他模型
                max_tokens=300,)
            print("Get results")
            reply = completion2.choices[0].message.content  # 获取回复内容
            info_record['select_mask_prompt'] = user_input_select_seg_mask
            info_record['select_mask_result'] = reply
            # reply = " nd image is 4.."
            # import pdb;pdb.set_trace()
            print(reply)
            extract_numbers_ = extract_numbers(reply)
            if len(extract_numbers_)==0:
                print(f"GPT can not find the seg_mask of {object_name}")
                return 0 #* return 1 成功， 0 GPT失败 -1 导航失败 -2 其他
            mask_id = extract_numbers_[0]
        print(f"Select Seg Mask: {mask_id}")  # 输出回复
        obj_center = self.img_and_mask_id_to_pos(image_id, mask_id, no_circle=True)
        def save_target(name, ids, obj_center, circle_center):
            if not isinstance(ids, list):
                print("ids should be list!")
                return
            target_data_path = os.path.join(self.basedata_path, "target_data0.npy")
            target_datas = []
            if os.path.exists(target_data_path):
                target_datas = np.load(target_data_path, allow_pickle=True).tolist()
                # import pdb;pdb.set_trace()
            for id0 in ids:
                target_data = {} #* {'plant', 'chair', 'toilet', 'sofa', 'tv_monitor', 'bed'}
                target_data["name"]=name
                target_data["center"] = obj_center
                target_data["optimal_pos"] = circle_center[id0]
                target_datas.append(target_data)
            print("target_datas: ", target_datas)
            np.save(target_data_path, target_datas)
        if direct_to_pos: #* 直接导航目的地
            result = self.navigate_to_target(obj_center, 0.2, time_sleep=0.001, move_steps_scale=1.0)
            if not result:
                print("Navigation fails! Return")
                return -1 #* return 1 成功， 0 GPT失败 -1 导航失败 -2 其他
            return 1
        #* 结合segmask和点云数据，选择合适的look——down角度   可以向下、左右看看，决定当前可地面区域，再选择合适观测区域
        # import pdb;pdb.set_trace()
        if no_recon:
            self.get_near_ground_area(image_id, mask_id, time_sleep=0.01, look_down_angle=0)
        else:
            self.get_near_ground_area(image_id, mask_id, time_sleep=0.01) #* 通过机器人左右上下看，获得周围的点云： 地面点点云 self.ground_pc  周围点点云 self.near_pc
        self.gaze_target(self.get_agent_pose_euler()[:3]+cam_z_dir_w*1, 30)
        obj_pts = np.array(self.obj_pc_w.points)
        #* 确定目标的向下投影的范围 obj_radius
        if 1:
            obj_pts_ground = obj_pts*1.0
            obj_center_ground = obj_center*1.0
            obj_pts_ground[:,1] = self.ground_height
            obj_center_ground[1] = self.ground_height
            dist_to_center = np.linalg.norm(obj_pts_ground-obj_center_ground, axis=1)
            dist_to_center_sorted = np.sort(dist_to_center)
            obj_radius = dist_to_center_sorted[int(len(dist_to_center_sorted)*0.95)]
        #* 根据目标水平范围确定观测距离
        near_datas_dir = os.path.join(self.basedata_path, "near_datas")
        if 1:
            print("obj_radius: ", obj_radius)
            # import pdb;pdb.set_trace()
            circle_radius = obj_radius+1.0
            near_pc_ds = self.near_pc.voxel_down_sample(voxel_size=0.01) #* 下采样
            near_pts = np.array(near_pc_ds.points)
            obstacle_mask = (near_pts[:,1]>self.ground_height+0.03)*(near_pts[:,1]<self.ground_height+1.5)
            obstacle_pts = near_pts[obstacle_mask]*1.0
            obstacle_pts[:,1]=self.ground_height
            start_t = time.time()
            smart_circles = SmartCircle(obj_center_ground, circle_radius, grid_length=0.4)
            smart_circles.add_pts_to_ground_map(obstacle_pts, 0.0) #* 向网格中添加障碍点
            smart_circles.add_pts_to_ground_map(np.array(self.ground_pc.points), 1.0) #* 向网格中添加地面点
            # print("obj_pts_ground")
            smart_circles.add_pts_to_ground_map(obj_pts_ground, 2.0) #* 向网格中添加目标对象在地面的投影点
            # print("get_ground_map")
            map_pts_pc = smart_circles.get_ground_map()
            map_pts_pc2, circle_pts, circle_center = smart_circles.get_object_envelope_circles(0.2)
            circle_pts[:,:,1] += 0.05
            map_pts_pc += map_pts_pc2
            self.gaze_target(obj_center, 15)
            # self.gaze_target(obj_center, 30)
            rgb_box, grid_boxes_pc, visible_id_and_uv, orig_rgb = self.draw_visible_circles(circle_center, circle_pts, alpha=0.7)
            if len(visible_id_and_uv)<1:
                self.gaze_target(obj_center, 15)
                rgb_box, grid_boxes_pc, visible_id_and_uv, orig_rgb = self.draw_visible_circles(circle_center, circle_pts, alpha=0.7)
            # img_out, mask_map = self.img_seg(orig_rgb)
            # cv2.imwrite(os.path.join(near_datas_dir, 'rgb_box_'+str(image_id).zfill(2)+'_'+str(mask_id).zfill(2)+'_'+str(0).zfill(1)+'_seg.png'), img_out[:,:,[2,1,0]])
            #* visible_id_and_uv 可见网格中心的 id 和对应字符的起始坐标
            cv2.imwrite(os.path.join(near_datas_dir, 'rgb_circle_'+str(image_id).zfill(2)+'_'+str(mask_id).zfill(2)+'_'+str(0).zfill(1)+'.png'), rgb_box[:,:,[2,1,0]])
            # cv2.imwrite(os.path.join(near_datas_dir, 'rgb_circle_'+str(image_id).zfill(2)+'_'+str(mask_id).zfill(2)+'_'+str(0).zfill(1)+'_0.png'), orig_rgb[:,:,[2,1,0]])
            # o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'near_proj2ground_pc_test.ply'), grid_boxes_pc+near_pc_ds)
            # o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'near_proj2ground_pc_test1.ply'), near_pc_ds)
            # # o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'near_proj2ground_pc_test1.ply'), grid_boxes_pc+self.ground_pc+near_pc_ds)
            # o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'near_proj2ground_pc_test2.ply'), map_pts_pc)
            o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'nmap_pts_pc'+str(image_id).zfill(2)+'_'+str(mask_id).zfill(2)+'_'+str(0).zfill(1)+'.ply'), grid_boxes_pc+near_pc_ds)
            # save_target(4)
            # import pdb;pdb.set_trace()
            move_steps_scales = [0.5, 1.0]
            if one_step:
                move_steps_scales = [1.0]
            last_center = obj_center*1.0
            for iter_i in range(1, len(move_steps_scales)+1):
                #* 导航到位置
                if len(visible_id_and_uv)<1:
                    # import pdb;pdb.set_trace()
                    print("No visible circles! Navigating to the target!")
                    move_steps_scale = move_steps_scales[iter_i-1]
                    result = self.navigate_to_target(last_center, 0.2, time_sleep=0.001, move_steps_scale=move_steps_scale)
                    if not result:
                        print("Navigation fails! Return")
                        return -1 #* return 1 成功， 0 GPT失败 -1 导航失败 -2 其他
                else:
                    if len(visible_id_and_uv)==1:
                        circle_id=visible_id_and_uv[0][0]
                    else:
                        circle_id=visible_id_and_uv[0][0]
                        # circle_id= 0
                        if use_gpt:
                            last_circle_image_path = os.path.join(near_datas_dir, 'rgb_circle_'+str(image_id).zfill(2)+'_'+str(mask_id).zfill(2)+'_'+str(iter_i-1).zfill(1)+'.png')
                            base64_image_rgb = encode_image(last_circle_image_path)
                            user_input_ = f"In this image, I have marked certain circles in the image with both color and number, which circle do I need go if I want to {exe}. Please inform me of its corresponding number. Scoring the accessibility of each circle on a scale from 0 to 10, where higher scores indicate better accessibility. Each circle should have a unique score. Please only return circle numbers with the top 6 highest scored values."
                            print("Asking GPT")
                            completion2 = self.client.chat.completions.create(
                                messages=[{"role": "user","content": [{"type": "text","text": user_input_}, {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image_rgb}"}}]}],
                                model="gpt-4-vision-preview",    # 或者其他模型
                                max_tokens=300,)
                            print("Get results")
                            reply = completion2.choices[0].message.content  # 获取回复内容
                            info_record['select_circle_prompt_'+str(iter_i)] = user_input_
                            info_record['select_circle_result_'+str(iter_i)] = reply
                            # reply="Circle 2 - Directl"
                            # import pdb;pdb.set_trace()
                            print(reply)
                            init_sel_numbers = re.findall(r'Circle \d+', reply)
                            if len(init_sel_numbers)>0:
                                extract_numbers_ = extract_numbers(init_sel_numbers[0])
                            else:
                                extract_numbers_ = extract_numbers(reply)
                            if len(extract_numbers_)==0:
                                print(f"GPT can not find the circle of {object_name}")
                                # return False
                            else:
                                circle_id = extract_numbers_[0] #* 因为输出的数字较多，这里可能还需要手动设置一下 circle_id
                        # import pdb;pdb.set_trace()
                    print("circle_id: ", circle_id)
                    move_steps_scale = move_steps_scales[iter_i-1]
                    # import pdb;pdb.set_trace()
                    # move_steps_scale = 1.0
                    # save_target('sofa', [19,20,22,24,26], obj_center, circle_center) #* {'plant', 'chair', 'toilet', 'sofa', 'tv_monitor', 'bed'}
                    result = self.navigate_to_target(np.asarray(circle_center[circle_id]), 0.2, time_sleep=0.001, move_steps_scale=move_steps_scale) #* move_steps_scale=0.5表示只移动0.5倍的步数
                    last_center = circle_center[circle_id]
                    if not result:
                        result = self.navigate_to_target(last_center, 0.2, time_sleep=0.001, move_steps_scale=move_steps_scale)
                        if not result:
                            print("Navigation fails! Return")
                            return -1 #* return 1 成功， 0 GPT失败 -1 导航失败 -2 其他
                self.gaze_target(obj_center, 15)
                if move_steps_scale==1.0:
                    break
                if no_recon:
                    self.get_near_ground_area(image_id, mask_id, time_sleep=0.01, look_down_angle=0)
                else:
                    self.get_near_ground_area(image_id, mask_id, time_sleep=0.01, look_down_angle=15)
                near_pc_ds = self.near_pc.voxel_down_sample(voxel_size=0.01) #* 下采样
                near_pts = np.array(near_pc_ds.points)
                obstacle_mask = (near_pts[:,1]>self.ground_height+0.03)*(near_pts[:,1]<self.ground_height+1.5)
                obstacle_pts = near_pts[obstacle_mask]*1.0
                obstacle_pts[:,1]=self.ground_height
                smart_circles.add_pts_to_ground_map(obstacle_pts, 0.0) #* 向网格中添加障碍点
                smart_circles.add_pts_to_ground_map(np.array(self.ground_pc.points), 1.0) #* 向网格中添加地面点
                map_pts_pc = smart_circles.get_ground_map()
                map_pts_pc2, circle_pts, circle_center = smart_circles.update_object_envelope_circles()
                map_pts_pc += map_pts_pc2
                # print("self.gaze_target(obj_center, 0)")
                self.gaze_target(obj_center, 15)
                rgb_box, grid_boxes_pc, visible_id_and_uv, orig_rgb = self.draw_visible_circles(circle_center, circle_pts, alpha=0.7)
                # img_out, mask_map = self.img_seg(orig_rgb)
                # cv2.imwrite(os.path.join(near_datas_dir, 'rgb_box_'+str(image_id).zfill(2)+'_'+str(mask_id).zfill(2)+'_'+str(iter_i).zfill(1)+'_seg.png'), img_out[:,:,[2,1,0]])
                cv2.imwrite(os.path.join(near_datas_dir, 'rgb_circle_'+str(image_id).zfill(2)+'_'+str(mask_id).zfill(2)+'_'+str(iter_i).zfill(1)+'.png'), rgb_box[:,:,[2,1,0]])
                # cv2.imwrite(os.path.join(near_datas_dir, 'rgb_circle_'+str(image_id).zfill(2)+'_'+str(mask_id).zfill(2)+'_'+str(iter_i).zfill(1)+'.png'), orig_rgb[:,:,[2,1,0]])
                # o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'near_proj2ground_pc_test.ply'), grid_boxes_pc+near_pc_ds)
                # o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'near_proj2ground_pc_test1.ply'), grid_boxes_pc+self.ground_pc+near_pc_ds)
                # o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'near_proj2ground_pc_test2.ply'), map_pts_pc)
                o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'nmap_pts_pc'+str(image_id).zfill(2)+'_'+str(mask_id).zfill(2)+'_'+str(iter_i).zfill(1)+'.ply'), grid_boxes_pc+near_pc_ds)
                # import pdb;pdb.set_trace()
                # save_target('tv_monitor', id0, obj_center, circle_center) #* {'plant', 'chair', 'toilet', 'sofa', 'tv_monitor', 'bed'}
                pass
            print("运行结束")
            obs = self.sim.get_sensor_observations(0)
            cv2.imwrite(os.path.join(near_datas_dir, 'rgb_'+str(image_id).zfill(2)+'_'+str(mask_id).zfill(2)+'_end.png'), obs['color_sensor'][:,:,[2,1,0]])
            if self.save_video:
                import json
                # dict = {'姓名':'小李','性别':'男'}
                jsonstr = json.dumps(info_record, ensure_ascii=False)
                with open(os.path.join(self.basedata_path, "video", "info.json"),'w', encoding='utf-8') as f:#dict转josn
                    f.write(jsonstr)
                # np.save(os.path.join(self.basedata_path, "video", "info.npy"), info_record)
            # o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'near_proj2ground_pc_test.ply'), grid_boxes_pc+near_pc_ds)
            # o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'near_proj2ground_pc_test1.ply'), grid_boxes_pc+self.ground_pc+near_pc_ds)
            # o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'near_proj2ground_pc_test2.ply'), map_pts_pc)
            # o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'nmap_pts_pc'+str(image_id).zfill(6)+'_'+str(mask_id).zfill(3)+'.ply'), grid_boxes_pc+near_pc_ds)
            # import pdb;pdb.set_trace()
            pass
        #* 重新采集数据
        return 1

    def draw_visible_circles(self, centers, circle_pts, alpha=0.1, font_color = (255, 255, 255), font_scale = 1, line_type = 2):
        pose = self.get_agent_pose()
        mn.gl.default_framebuffer.clear(mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH)
        mn.gl.default_framebuffer.bind()
        mn.gl.Renderer.disable(mn.gl.Renderer.Feature.BLENDING)
        obs = self.sim.get_sensor_observations(0)
        rgb_box = obs['color_sensor'][:,:,:3].astype(np.uint8).copy() 
        pose[:3, :3] = pose[:3, :3] @ self.rot_ro_cam
        ext = np.linalg.inv(pose)
        height, width = rgb_box.shape[:2]
        depth_np = obs['depth_sensor']
        img_pts = (self.image_kuv1 * depth_np[:,:,None]).reshape(-1,3)
        img_pc = o3d.geometry.PointCloud()
        img_pc.points = o3d.utility.Vector3dVector(img_pts)
        img_pc.colors = o3d.utility.Vector3dVector(rgb_box.reshape(-1,3).astype(np.float32)/255.0)
        pcd_w = img_pc.transform(pose)
        pcd_w_pts = np.asarray(pcd_w.points)
        pcd_w_colors = np.asarray(pcd_w.colors)
        #* 解决遮挡问题
        rgb_box_valid_mask = np.ones((height, width)).astype(bool)
        if self.ground_height is not None:
            ground_mask = (pcd_w_pts[:,1]<self.ground_height+0.02)*(pcd_w_pts[:,1]>self.ground_height-0.02)
            non_ground_pts_w = pcd_w_pts[~ground_mask,:]*1.0
            pcd_w_colors[~ground_mask,:] = [0,1,0]
            pcd_w.colors = o3d.utility.Vector3dVector(pcd_w_colors)
            non_ground_pts_c_tmp = ext[:3,:3] @ non_ground_pts_w.T + ext[:3,3:]
            non_ground_pts_uvd = self.cam_mat @ non_ground_pts_c_tmp
            non_ground_pts_uv1_ = (non_ground_pts_uvd[:2]/(1e-8+non_ground_pts_uvd[2:3])).astype(int)
            rgb_box_valid_mask[non_ground_pts_uv1_[1, :],non_ground_pts_uv1_[0, :]] = 0
            # o3d.io.write_point_cloud(os.path.join("/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/output/rendered/00009-vLpv2VX547B_basedata/near_datas", 'pts_w.ply'), pcd_w)
            # import pdb;pdb.set_trace()
        #* 将超出边界的网格不显示
        visible_id_and_uv = []
        grid_boxes_pc = o3d.geometry.PointCloud()
        for i in range(len(centers)):
            if centers[i] is not None:
                cicle_center_tmp = ext[:3,:3] @ np.asarray(centers[i]).reshape(-1,1) + ext[:3,3:]
                uvd = self.cam_mat @ cicle_center_tmp
                img_uv1_ = (uvd[:2]/(1e-8+uvd[2:3])).astype(int)
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_position = (int(img_uv1_[0]), int(img_uv1_[1]))
                if text_position[0]<0 or text_position[0]>=width or  text_position[1]<0 or text_position[1]>=height: continue
                if rgb_box_valid_mask[text_position[1],text_position[0]] == 0: continue
                # import pdb;pdb.set_trace()
                (text_width, text_height), baseline = cv2.getTextSize(str(i), font, font_scale, line_type)
                if text_position[0] - text_width // 2<0 or text_position[0] - text_width // 2>=width or  text_position[1] + text_height // 2<0 or text_position[1] + text_height // 2>=height: continue
                if rgb_box_valid_mask[text_position[1] + text_height // 2,text_position[0] - text_width // 2] == 0: continue
                if text_position[0] + text_width // 2<0 or text_position[0] + text_width // 2>=width or  text_position[1] - text_height // 2<0 or text_position[1] - text_height // 2>=height: continue
                if rgb_box_valid_mask[text_position[1] - text_height // 2,text_position[0] + text_width // 2] == 0: continue
                # Calculate the bottom-left corner of the text
                start_x = text_position[0] - text_width // 2
                start_y = text_position[1] + text_height // 2
                visible_id_and_uv.append([i, (start_x, start_y)])
                circle_pc = o3d.geometry.PointCloud()
                circle_pc.points = o3d.utility.Vector3dVector(circle_pts[i][:,:3])
                circle_pc.colors = o3d.utility.Vector3dVector(circle_pts[i][:,3:6])
                grid_boxes_pc+=circle_pc
        grid_boxes_pts = np.asarray(grid_boxes_pc.points)
        grid_boxes_colors = np.asarray(grid_boxes_pc.colors)
        grid_boxes_l_pts = (ext[:3,:3] @ grid_boxes_pts[:,:3].T + ext[:3,3:]).T
        img_uvd = (self.cam_mat @ grid_boxes_l_pts[:,:3].T).T
        valid_mask = img_uvd[:,2]>1e-8
        img_uv1 = (img_uvd[:,:2]/(1e-8+img_uvd[:,2:3])).astype(int)
        valid_mask = valid_mask*(img_uv1[:,1]>=0)*(img_uv1[:,1]<height)*(img_uv1[:,0]>=0)*(img_uv1[:,0]<width)
        img_uv1 = img_uv1[valid_mask,:]
        grid_boxes_l_colors = (grid_boxes_colors[valid_mask,:3]*255).astype(np.uint8)
        mn.gl.default_framebuffer.clear(mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH)
        mn.gl.default_framebuffer.bind()
        mn.gl.Renderer.disable(mn.gl.Renderer.Feature.BLENDING)
        rgb_box[img_uv1[:,1],img_uv1[:,0],:3] = (rgb_box[img_uv1[:,1],img_uv1[:,0],:3]*(1-alpha)+grid_boxes_l_colors*alpha).astype(np.uint8)
        # rgb_box[rgb_box_valid_mask,:3] = [0,0,255]
        # import pdb;pdb.set_trace()
        # cv2.imwrite(os.path.join("/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/output/rendered/00009-vLpv2VX547B_basedata/near_datas/test.png"), rgb_box[:,:,[2,1,0]])
        # cv2.imwrite(os.path.join("/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/output/rendered/00009-vLpv2VX547B_basedata/near_datas/test1.png"), rgb_box_valid_mask)
        #* 在图像中绘制数字
        for i in range(len(visible_id_and_uv)):
            cv2.putText(rgb_box, str(visible_id_and_uv[i][0]), visible_id_and_uv[i][1], font, font_scale, font_color, line_type)
        return rgb_box, grid_boxes_pc, visible_id_and_uv, obs['color_sensor'][:,:,:3].astype(np.uint8)

    def draw_visible_boxes(self, centers, grid_box_pcs, alpha=0.1, font_color = (255, 255, 255), font_scale = 1, line_type = 2):
        pose = self.get_agent_pose()
        mn.gl.default_framebuffer.clear(mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH)
        mn.gl.default_framebuffer.bind()
        mn.gl.Renderer.disable(mn.gl.Renderer.Feature.BLENDING)
        obs = self.sim.get_sensor_observations(0)
        rgb_box = obs['color_sensor'][:,:,:3].astype(np.uint8).copy() 
        pose[:3, :3] = pose[:3, :3] @ self.rot_ro_cam
        ext = np.linalg.inv(pose)
        height, width = rgb_box.shape[:2]
        depth_np = obs['depth_sensor']
        img_pts = (self.image_kuv1 * depth_np[:,:,None]).reshape(-1,3)
        img_pc = o3d.geometry.PointCloud()
        img_pc.points = o3d.utility.Vector3dVector(img_pts)
        img_pc.colors = o3d.utility.Vector3dVector(rgb_box.reshape(-1,3).astype(np.float32)/255.0)
        pcd_w = img_pc.transform(pose)
        pcd_w_pts = np.asarray(pcd_w.points)
        pcd_w_colors = np.asarray(pcd_w.colors)
        #* 解决遮挡问题
        rgb_box_valid_mask = np.ones((height, width)).astype(bool)
        if self.ground_height is not None:
            ground_mask = (pcd_w_pts[:,1]<self.ground_height+0.02)*(pcd_w_pts[:,1]>self.ground_height-0.02)
            non_ground_pts_w = pcd_w_pts[~ground_mask,:]*1.0
            pcd_w_colors[~ground_mask,:] = [0,1,0]
            pcd_w.colors = o3d.utility.Vector3dVector(pcd_w_colors)
            non_ground_pts_c_tmp = ext[:3,:3] @ non_ground_pts_w.T + ext[:3,3:]
            non_ground_pts_uvd = self.cam_mat @ non_ground_pts_c_tmp
            non_ground_pts_uv1_ = (non_ground_pts_uvd[:2]/(1e-8+non_ground_pts_uvd[2:3])).astype(int)
            rgb_box_valid_mask[non_ground_pts_uv1_[1, :],non_ground_pts_uv1_[0, :]] = 0
            # o3d.io.write_point_cloud(os.path.join("/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/output/rendered/00009-vLpv2VX547B_basedata/near_datas", 'pts_w.ply'), pcd_w)
            # import pdb;pdb.set_trace()
        #* 将超出边界的网格不显示
        visible_id_and_uv = []
        grid_boxes_pc = o3d.geometry.PointCloud()
        for i in range(len(centers)):
            if centers[i] is not None:
                cicle_center_tmp = ext[:3,:3] @ np.asarray(centers[i]).reshape(-1,1) + ext[:3,3:]
                uvd = self.cam_mat @ cicle_center_tmp
                img_uv1_ = (uvd[:2]/(1e-8+uvd[2:3])).astype(int)
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_position = (int(img_uv1_[0]), int(img_uv1_[1]))
                if text_position[0]<0 or text_position[0]>=width or  text_position[1]<0 or text_position[1]>=height: continue
                if rgb_box_valid_mask[text_position[1],text_position[0]] == 0: continue
                # import pdb;pdb.set_trace()
                (text_width, text_height), baseline = cv2.getTextSize(str(i), font, font_scale, line_type)
                if text_position[0] - text_width // 2<0 or text_position[0] - text_width // 2>=width or  text_position[1] + text_height // 2<0 or text_position[1] + text_height // 2>=height: continue
                if rgb_box_valid_mask[text_position[1] + text_height // 2,text_position[0] - text_width // 2] == 0: continue
                if text_position[0] + text_width // 2<0 or text_position[0] + text_width // 2>=width or  text_position[1] - text_height // 2<0 or text_position[1] - text_height // 2>=height: continue
                if rgb_box_valid_mask[text_position[1] - text_height // 2,text_position[0] + text_width // 2] == 0: continue
                # Calculate the bottom-left corner of the text
                start_x = text_position[0] - text_width // 2
                start_y = text_position[1] + text_height // 2
                visible_id_and_uv.append([i, (start_x, start_y)])
                grid_boxes_pc+=grid_box_pcs[i]
        grid_boxes_pts = np.asarray(grid_boxes_pc.points)
        grid_boxes_colors = np.asarray(grid_boxes_pc.colors)
        grid_boxes_l_pts = (ext[:3,:3] @ grid_boxes_pts[:,:3].T + ext[:3,3:]).T
        img_uvd = (self.cam_mat @ grid_boxes_l_pts[:,:3].T).T
        valid_mask = img_uvd[:,2]>1e-8
        img_uv1 = (img_uvd[:,:2]/(1e-8+img_uvd[:,2:3])).astype(int)
        valid_mask = valid_mask*(img_uv1[:,1]>=0)*(img_uv1[:,1]<height)*(img_uv1[:,0]>=0)*(img_uv1[:,0]<width)
        img_uv1 = img_uv1[valid_mask,:]
        grid_boxes_l_colors = (grid_boxes_colors[valid_mask,:3]*255).astype(np.uint8)
        mn.gl.default_framebuffer.clear(mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH)
        mn.gl.default_framebuffer.bind()
        mn.gl.Renderer.disable(mn.gl.Renderer.Feature.BLENDING)
        rgb_box[img_uv1[:,1],img_uv1[:,0],:3] = (rgb_box[img_uv1[:,1],img_uv1[:,0],:3]*(1-alpha)+grid_boxes_l_colors*alpha).astype(np.uint8)
        # rgb_box[rgb_box_valid_mask,:3] = [0,0,255]
        # import pdb;pdb.set_trace()
        # cv2.imwrite(os.path.join("/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/output/rendered/00009-vLpv2VX547B_basedata/near_datas/test.png"), rgb_box[:,:,[2,1,0]])
        # cv2.imwrite(os.path.join("/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/output/rendered/00009-vLpv2VX547B_basedata/near_datas/test1.png"), rgb_box_valid_mask)
        #* 在图像中绘制数字
        for i in range(len(visible_id_and_uv)):
            cv2.putText(rgb_box, str(visible_id_and_uv[i][0]), visible_id_and_uv[i][1], font, font_scale, font_color, line_type)
        return rgb_box, grid_boxes_pc, visible_id_and_uv, obs['color_sensor'][:,:,:3].astype(np.uint8)

    def init_som(self):
        self.metadata = MetadataCatalog.get('coco_2017_train_panoptic')
        data_path = '/media/zhujun/0DFD06D20DFD06D2/SLAM/SoM/'
        semsam_cfg = data_path+"/configs/semantic_sam_only_sa-1b_swinL.yaml"
        seem_cfg = data_path+"/configs/seem_focall_unicl_lang_v1.yaml"
        semsam_ckpt = data_path+"/ckpt/swinl_only_sam_many2many.pth"
        sam_ckpt = data_path+"/ckpt/sam_vit_h_4b8939.pth"
        seem_ckpt = data_path+"/ckpt/seem_focall_v1.pt"
        print("Loading model ....")
        opt_semsam = load_opt_from_config_file(semsam_cfg)
        opt_seem = load_opt_from_config_file(seem_cfg)
        opt_seem = init_distributed_seem(opt_seem)
        self.model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
        self.model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
        self.model_seem = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                self.model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
        print("Loading model done")

    @torch.no_grad()
    def img_seg(self, image, slider=1.2, mode='Automatic', alpha=0.3, label_mode='Number', anno_mode=['Mask', 'Mark']):
        # import pdb;pdb.set_trace()
        # image = {'image': <PIL.Image.Image image mode=RGB size=1080x720 at 0x7F5A24175DC0>, 'mask': <PIL.Image.Image image mode=RGB size=1080x720 at 0x7F5A2417BA30>}
        # slider = 2
        # mode = 'Automatic'
        # alpha = 0.1
        # label_mode = 'Number'
        # anno_mode = ['Mask', 'Mark']
        # args = ()
        # kwargs = {}
        if slider <= 1.5:
            model_name = 'seem'
        elif slider >= 2.5:
            model_name = 'sam'
        else:
            if mode == 'Automatic':
                model_name = 'semantic-sam'
                if slider < 1.64:                
                    level = [1]
                elif slider < 1.78:
                    level = [2]
                elif slider < 1.92:
                    level = [3]
                elif slider < 2.06:
                    level = [4]
                elif slider < 2.2:
                    level = [5]
                elif slider < 2.34:
                    level = [6]
                else:
                    level = [6, 1, 2, 3, 4, 5]
            else:
                model_name = 'sam'
        label_mode = 'a' if label_mode == 'Alphabet' else '1'
        text_size=640
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # t = []
            # t.append(transforms.Resize(int(text_size), interpolation=Image.BICUBIC))
            # transform1 = transforms.Compose(t)
            # image_ori = transform1(image['image'])
            image_ori = image
            image_ori = np.asarray(image_ori)
            if model_name == 'semantic-sam':
                model = self.model_semsam
                images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
                mask_generator = SemanticSamAutomaticMaskGenerator(model,points_per_side=32,
                        pred_iou_thresh=0.88,
                        stability_score_thresh=0.92,
                        min_mask_region_area=10,
                        level=level,
                    )
                outputs = mask_generator.generate(images)
            elif model_name == 'sam':
                model = self.model_sam
                mask_generator = SamAutomaticMaskGenerator(model)
                outputs = mask_generator.generate(image_ori)
            elif model_name == 'seem':
                model = self.model_seem
                images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
                orig_size = images.shape[-2:]
                orig_h, orig_w = orig_size
                crop_box = [0,0,orig_w,orig_h]
                data = {"image": images, "height": orig_h, "width": orig_w}
                batch_inputs = [data]
                model.model.metadata = self.metadata
                outputs = model.model.evaluate(batch_inputs)
                pano_mask = outputs[0]['panoptic_seg'][0]
                pano_info = outputs[0]['panoptic_seg'][1]
                masks = []
                for seg_info in pano_info:
                    masks += [pano_mask == seg_info['id']]
                masks = torch.stack(masks, dim=0)
                iou_preds = torch.ones(masks.shape[0], dtype=torch.float32)
                points = torch.zeros((masks.shape[0], 2), dtype=torch.float32)
                mask_data = MaskData(
                    masks=masks,
                    iou_preds=iou_preds,
                    points=points,
                )
                mask_data["stability_score"] = torch.ones(masks.shape[0], dtype=torch.float32)
                del masks
                mask_data["boxes"] = batched_mask_to_box(mask_data["masks"])
                mask_data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(mask_data["boxes"]))])
                # Compress to RLE
                mask_data["masks"] = uncrop_masks(mask_data["masks"], crop_box, orig_h, orig_w)
                mask_data["rles"] = mask_to_rle_pytorch(mask_data["masks"])
                del mask_data["masks"]
                mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
                # Write mask records
                outputs = []
                for idx in range(len(mask_data["segmentations"])):
                    ann = {
                        "segmentation": mask_data["segmentations"][idx],
                        "area": area_from_rle(mask_data["rles"][idx]),
                        "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                        "predicted_iou": mask_data["iou_preds"][idx].item(),
                        "point_coords": [mask_data["points"][idx].tolist()],
                        "stability_score": mask_data["stability_score"][idx].item(),
                        "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
                    }
                    outputs.append(ann)
            visual = Visualizer(image_ori, metadata=self.metadata)
            sorted_anns = sorted(outputs, key=(lambda x: x['area']), reverse=True)
            label = 1
            mask_map = np.zeros(image_ori.shape, dtype=np.uint8)    
            for i, ann in enumerate(sorted_anns):
                mask = ann['segmentation']
                demo = visual.draw_binary_mask_with_number(mask, edge_color=[1,1,1], text=str(label), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
                mask_map[mask == 1] = label
                label += 1
            im = demo.get_image()
            output = im
            mask = mask_map
            return output, mask_map

    def draw_contact_debug(self):
        """
        This method is called to render a debug line overlay displaying active contact points and normals.
        Yellow lines show the contact distance along the normal and red lines show the contact normal at a fixed length.
        """
        yellow = mn.Color4.yellow()
        red = mn.Color4.red()
        cps = self.sim.get_physics_contact_points()
        self.sim.get_debug_line_render().set_line_width(1.5)
        camera_position = self.render_camera.render_camera.node.absolute_translation
        # only showing active contacts
        active_contacts = (x for x in cps if x.is_active)
        for cp in active_contacts:
            # red shows the contact distance
            self.sim.get_debug_line_render().draw_transformed_line(
                cp.position_on_b_in_ws,
                cp.position_on_b_in_ws
                + cp.contact_normal_on_b_in_ws * -cp.contact_distance,
                red,
            )
            # yellow shows the contact normal at a fixed length for visualization
            self.sim.get_debug_line_render().draw_transformed_line(
                cp.position_on_b_in_ws,
                # + cp.contact_normal_on_b_in_ws * cp.contact_distance,
                cp.position_on_b_in_ws + cp.contact_normal_on_b_in_ws * 0.1,
                yellow,
            )
            self.sim.get_debug_line_render().draw_circle(
                translation=cp.position_on_b_in_ws,
                radius=0.005,
                color=yellow,
                normal=camera_position - cp.position_on_b_in_ws,
            )

    def debug_draw(self):
        """
        Additional draw commands to be called during draw_event.
        """
        if self.debug_bullet_draw:
            render_cam = self.render_camera.render_camera
            proj_mat = render_cam.projection_matrix.__matmul__(render_cam.camera_matrix)
            self.sim.physics_debug_draw(proj_mat)
        if self.contact_debug_draw:
            self.draw_contact_debug()

    def draw_event(
        self,
        simulation_call: Optional[Callable] = None,
        global_call: Optional[Callable] = None,
        active_agent_id_and_sensor_name: Tuple[int, str] = (0, "color_sensor"),
    ) -> None:
        """
        Calls continuously to re-render frames and swap the two frame buffers
        at a fixed rate.
        """
        agent_acts_per_sec = self.fps
        mn.gl.default_framebuffer.clear(mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH)
        # Agent actions should occur at a fixed rate per second
        self.time_since_last_simulation += Timer.prev_frame_duration
        num_agent_actions: int = self.time_since_last_simulation * agent_acts_per_sec
        self.move_and_look(int(num_agent_actions))
        # Occasionally a frame will pass quicker than 1/60 seconds
        if self.time_since_last_simulation >= 1.0 / self.fps:
            if self.simulating or self.simulate_single_step:
                self.sim.step_world(1.0 / self.fps)
                self.simulate_single_step = False
                if simulation_call is not None:
                    simulation_call()
            if global_call is not None:
                global_call()
            # reset time_since_last_simulation, accounting for potential overflow
            self.time_since_last_simulation = math.fmod(self.time_since_last_simulation, 1.0 / self.fps)
        keys = active_agent_id_and_sensor_name
        if self.enable_batch_renderer:
            self.render_batch()
        else:
            self.sim._Simulator__sensors[keys[0]][keys[1]].draw_observation()
            agent = self.sim.get_agent(keys[0])
            self.render_camera = agent.scene_node.node_sensor_suite.get(keys[1])
            self.debug_draw()
            self.render_camera.render_target.blit_rgba_to_default()
        # draw CPU/GPU usage data and other info to the app window
        mn.gl.default_framebuffer.bind()
        self.draw_text(self.render_camera.specification())
        self.swap_buffers()
        Timer.next_frame()
        self.redraw()

    def default_agent_config(self) -> habitat_sim.agent.AgentConfiguration:
        """
        Set up our own agent and agent controls
        """
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec
        MOVE, LOOK = 0.07, 1.5
        # all of our possible actions' names
        action_list = [
            "move_left",
            "turn_left",
            "move_right",
            "turn_right",
            "move_backward",
            "look_up",
            "move_forward",
            "look_down",
            "move_down",
            "move_up",
        ]
        action_space: Dict[str, habitat_sim.agent.ActionSpec] = {}
        # build our action space map
        for action in action_list:
            actuation_spec_amt = MOVE if "move" in action else LOOK
            action_spec = make_action_spec(action, make_actuation_spec(actuation_spec_amt))
            action_space[action] = action_spec
        sensor_spec: List[habitat_sim.sensor.SensorSpec] = self.cfg.agents[self.agent_id].sensor_specifications
        agent_config = habitat_sim.agent.AgentConfiguration(
            height=1.5,
            radius=0.1,
            sensor_specifications=sensor_spec,
            action_space=action_space,
            body_type="cylinder",
        )
        return agent_config

    def reconfigure_sim(self) -> None:
        """
        Utilizes the cur_node `self.sim_settings` to configure and set up a new
        `habitat_sim.Simulator`, and then either starts a simulation instance, or replaces
        the cur_node simulator instance, reloading the most recently loaded scene
        """
        # configure our sim_settings but then set the agent to our default
        # import pdb;pdb.set_trace()
        self.cfg = make_cfg(self.sim_settings) #* 设置见这个函数
        # import pdb;pdb.set_trace()
        # self.cfg.agents[0].sensor_specifications[0].hfov
        self.agent_id: int = self.sim_settings["default_agent"]
        self.cfg.agents[self.agent_id] = self.default_agent_config()
        if self.enable_batch_renderer:
            self.cfg.enable_batch_renderer = True
            self.cfg.sim_cfg.create_renderer = False
            self.cfg.sim_cfg.enable_gfx_replay_save = True
        if self.sim_settings["use_default_lighting"]:
            logger.info("Setting default lighting override for scene.")
            self.cfg.sim_cfg.override_scene_light_defaults = True
            self.cfg.sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
        if self.sim is None:
            self.tiled_sims = []
            for _i in range(self.num_env):
                # import pdb;pdb.set_trace()
                self.tiled_sims.append(habitat_sim.Simulator(self.cfg))
            self.sim = self.tiled_sims[0]
        else:  # edge case
            for i in range(self.num_env):
                if (self.tiled_sims[i].config.sim_cfg.scene_id == self.cfg.sim_cfg.scene_id):
                    # we need to force a reset, so change the internal config scene name
                    self.tiled_sims[i].config.sim_cfg.scene_id = "NONE"
                self.tiled_sims[i].reconfigure(self.cfg)
        # post reconfigure
        self.default_agent = self.sim.get_agent(self.agent_id)
        self.render_camera = self.default_agent.scene_node.node_sensor_suite.get("color_sensor")
        # set sim_settings scene name as actual loaded scene
        self.sim_settings["scene"] = self.sim.curr_scene_name
        # Initialize replay renderer
        if self.enable_batch_renderer and self.replay_renderer is None:
            self.replay_renderer_cfg = ReplayRendererConfiguration()
            self.replay_renderer_cfg.num_environments = self.num_env
            self.replay_renderer_cfg.standalone = (False)  # Context is owned by the GLFW window
            self.replay_renderer_cfg.sensor_specifications = self.cfg.agents[self.agent_id].sensor_specifications
            self.replay_renderer_cfg.gpu_device_id = self.cfg.sim_cfg.gpu_device_id
            self.replay_renderer_cfg.force_separate_semantic_scene_graph = False
            self.replay_renderer_cfg.leave_context_with_background_renderer = False
            self.replay_renderer = ReplayRenderer.create_batch_replay_renderer(self.replay_renderer_cfg)
            # Pre-load composite files
            if sim_settings["composite_files"] is not None:
                for composite_file in sim_settings["composite_files"]:
                    self.replay_renderer.preload_file(composite_file)
        Timer.start()
        self.step = -1

    def render_batch(self):
        """
        This method updates the replay manager with the cur_node state of environments and renders them.
        """
        for i in range(self.num_env):
            # Apply keyframe
            keyframe = self.tiled_sims[i].gfx_replay_manager.extract_keyframe()
            self.replay_renderer.set_environment_keyframe(i, keyframe)
            # Copy sensor transforms
            sensor_suite = self.tiled_sims[i]._sensors
            for sensor_uuid, sensor in sensor_suite.items():
                transform = sensor._sensor_object.node.absolute_transformation()
                self.replay_renderer.set_sensor_transform(i, sensor_uuid, transform)
            # Render
            self.replay_renderer.render(mn.gl.default_framebuffer)

    def move_and_look(self, repetitions: int) -> None:
        """
        This method is called continuously with `self.draw_event` to monitor
        any changes in the movement keys map `Dict[KeyEvent.key, Bool]`.
        When a key in the map is set to `True` the corresponding action is taken.
        """
        # avoids unnecessary updates to grabber's object position
        if repetitions == 0:
            return
        key = Application.KeyEvent.Key
        agent = self.sim.agents[self.agent_id]
        press: Dict[key.key, bool] = self.pressed
        act: Dict[key.key, str] = self.key_to_action
        action_queue: List[str] = [act[k] for k, v in press.items() if v]
        # if(len(action_queue)>0):
        #     print("len(action_queue): ",len(action_queue), action_queue)
        for _ in range(int(repetitions)):
            [agent.act(x) for x in action_queue]
        # update the grabber transform when our agent is moved
        if self.mouse_grabber is not None:
            # update location of grabbed object
            self.update_grab_position(self.previous_mouse_point)

    def invert_gravity(self) -> None:
        """
        Sets the gravity vector to the negative of it's previous value. This is
        a good method for testing simulation functionality.
        """
        gravity: mn.Vector3 = self.sim.get_gravity() * -1
        self.sim.set_gravity(gravity)

    def get_agent_pose(self):
        pose = np.eye(4)
        agent_state = self.sim.get_agent(0).state.sensor_states['depth_sensor']
        orig_position = agent_state.position.copy()
        rotation = agent_state.rotation.copy()
        pose[:3,3] = np.asarray(orig_position)
        pose[:3,:3] = Rotation.from_quat(np.asarray([rotation.x, rotation.y, rotation.z, rotation.w])).as_matrix()
        return pose

    def get_agent_pose_euler(self):
        pose_euler = np.ones((6,))
        agent_state = self.sim.get_agent(0).state.sensor_states['depth_sensor']
        orig_position = agent_state.position.copy()
        rotation = agent_state.rotation.copy()
        pose_euler[:3] = np.asarray(orig_position)
        pose_euler[3:6] = Rotation.from_quat(np.asarray([rotation.x, rotation.y, rotation.z, rotation.w])).as_euler('xyz', degrees=True)
        return pose_euler

    def set_state_np(self, pose, l3mvn=0.0):
        agent_state = self.sim.get_agent(0).state.sensor_states['depth_sensor']
        # import pdb;pdb.set_trace()
        agent_state.position = pose[:3]
        if l3mvn>0:
            agent_state.position[1] -= self.sim_settings["sensor_height"]-0.88*l3mvn
        agent_state.rotation.x = pose[3]
        agent_state.rotation.y = pose[4]
        agent_state.rotation.z = pose[5]
        agent_state.rotation.w = pose[6]
        self.sim.get_agent(0).set_state(agent_state)

    def initialize_pose(self):
        # self.init_pose = True
        # 获取场景的根节点
        scene_graph = self.sim.get_active_scene_graph()
        root_node = scene_graph.get_root_node()
        # # 计算根节点的包围盒，这代表了整个场景的范围
        scene_bounds = root_node.cumulative_bb
        # # 打印场景范围（包围盒的最小和最大顶点）
        # print("Scene bounds:", scene_bounds.min, scene_bounds.max)
        center = 0.5*(scene_bounds.min+ scene_bounds.max)
        center = np.asarray(center)
        center[1] = 0.2
        # navi = self.pathfinder.get_random_navigable_point_near(center,0.50)
        for i in range(10):
            navi = self.pathfinder.get_random_navigable_point()
            # navi[1] = 0.2
            collision = not self.pathfinder.is_navigable(navi)
            if abs(navi[1]-0.1)<0.2 and not collision:
                break
        # 根据碰撞检测结果判断是否能前进
        if not collision:
            print("设置随机导航起始点.", navi)
            agent_state = self.sim.get_agent(0).state.sensor_states['depth_sensor']
            agent_state.position = navi
            self.sim.get_agent(0).set_state(agent_state)
        else:
            print("前方有障碍物，代理不能前进.")
        #* 加载数据
        print(f"loading scene {self.basedata_path}")
        rgb_dir = os.path.join(self.basedata_path, "images")
        depth_dir = os.path.join(self.basedata_path, "depths")
        if not os.path.exists(rgb_dir):
            print("There is no basedata!")
            return
        rgb_list = sorted([v for v in os.listdir(rgb_dir) if '.png' in v and  '_' not in v], key=lambda x: int(x.split("_")[-1].split(".")[0]))
        self.rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
        seg_list = sorted([v for v in os.listdir(rgb_dir) if 'seg' in v], key=lambda x: int(x.split("_")[0]))
        if len(seg_list)!=len(rgb_list):
            if not self.sim_settings['use_som']:
                self.sim_settings['use_som'] = True
                self.init_som()
            for image_id in range(len(rgb_list)):
                img_out, mask_map = self.img_seg(Image.open(self.rgb_list[image_id]), slider=1.2, alpha=0.3) #* slider=1.2, alpha=0.3
                # img_out, mask_map = self.img_seg(Image.open(self.rgb_list[image_id]))
                cv2.imwrite(os.path.join(rgb_dir, str(image_id).zfill(6)+'_seg.png'), img_out[:,:,[2,1,0]])
                np.save(os.path.join(rgb_dir, str(image_id).zfill(6)+'_mask.npy'), mask_map[:,:,0])
                # if image_id==4: import pdb;pdb.set_trace()
        seg_list = sorted([v for v in os.listdir(rgb_dir) if 'seg' in v], key=lambda x: int(x.split("_")[0]))
        mask_list = sorted([v for v in os.listdir(rgb_dir) if 'mask.npy' in v], key=lambda x: int(x.split("_")[0]))
        depth_list = sorted(os.listdir(depth_dir), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        self.seg_list = [os.path.join(rgb_dir, x) for x in seg_list]
        self.mask_list = [os.path.join(rgb_dir, x) for x in mask_list]
        self.depth_list = [os.path.join(depth_dir, x) for x in depth_list]
        self.database_poses = np.loadtxt(os.path.join(self.basedata_path, "poses.txt"))
        # import pdb;pdb.set_trace()
        navi = self.pathfinder.get_random_navigable_point_near(self.database_poses[0,:3],2.0)
        # navi = self.pathfinder.get_random_navigable_point()
        collision = not self.pathfinder.is_navigable(navi)
        # 根据碰撞检测结果判断是否能前进
        if not collision:
            # print("设置随机导航起始点.")
            agent_state = self.sim.get_agent(0).state.sensor_states['depth_sensor']
            agent_state.position = navi
            self.sim.get_agent(0).set_state(agent_state)
        else:
            print("前方有障碍物，代理不能前进.")
        # initialize a grid with zero position at the center
        # bgr = cv2.imread(self.rgb_list[0])
        depth = np.load(self.depth_list[0])
        h, w = depth.shape
        self.cam_mat = np.loadtxt(os.path.join(self.basedata_path, "intrinsic.txt"))
        self.cam_intrinsics = self.cam_mat.copy()
        cam_mat_inv = np.linalg.inv(self.cam_mat)
        xx, yy = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))  # pytorch's meshgrid has indexing='ij'
        image_uv1 = np.stack([xx, yy, np.ones_like(xx)],axis=2)
        # import pdb;pdb.set_trace()
        self.image_kuv1 = (cam_mat_inv @ image_uv1.reshape(-1,3).T).T.reshape(h,w,3)
        self.rot_ro_cam = np.eye(3)
        self.rot_ro_cam[1, 1] = -1
        self.rot_ro_cam[2, 2] = -1

    def save_video_data(self, ):
        if not self.save_video: return
        video_rgb_dir = os.path.join(self.basedata_path, 'video', 'rgb')
        video_depth_dir = os.path.join(self.basedata_path, 'video', 'depth')
        mn.gl.default_framebuffer.clear(mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH)
        mn.gl.default_framebuffer.bind()
        mn.gl.Renderer.disable(mn.gl.Renderer.Feature.BLENDING)
        obs = self.sim.get_sensor_observations(0)
        agent_state = self.sim.get_agent(0).state.sensor_states['depth_sensor']
        pose = np.zeros(7)
        pose[:3] = agent_state.position
        pose[3:] = np.asarray([agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z, agent_state.rotation.w])
        self.video_poses.append(pose)
        # self.video_poses = []
        # self.save_video_img_id = 0
        if not os.path.exists(os.path.join(self.basedata_path, 'video', 'intrinsic.txt')):
            width = int(self.sim_settings["width"])
            height = int(self.sim_settings["height"])
            hfov = math.radians(self.cfg.agents[0].sensor_specifications[0].hfov)
            cam_intrinsics = np.eye(3)
            cam_intrinsics[0,0] = width / (2.0 * math.tan(hfov / 2.0))
            # cam_intrinsics[1,1] = cam_intrinsics[0,0] * (height / width)
            cam_intrinsics[1,1] = cam_intrinsics[0,0]
            cam_intrinsics[0,2] = width/2
            cam_intrinsics[1,2] = height/2
            np.savetxt(os.path.join(self.basedata_path, 'video', 'intrinsic.txt'), cam_intrinsics)
        np.savetxt(os.path.join(self.basedata_path,  'video', 'poses.txt'), np.asarray(self.video_poses))
        cv2.imwrite(os.path.join(video_rgb_dir, str(self.save_video_img_id).zfill(6)+'.png'), obs['color_sensor'][:,:,[2,1,0]])
        np.save(os.path.join(video_depth_dir, str(self.save_video_img_id).zfill(6)+'.npy'), obs['depth_sensor'])
        self.save_video_img_id += 1

    #* 采集当前相机位置附近的图像信息，获得地面区域点云
    def get_near_ground_area(self, image_id, mask_id, look_down_angle=60, look_right_left_angle=30, delta_look_rl_angle=30, time_sleep = 0.1):
        near_datas_dir = os.path.join(self.basedata_path, "near_datas")
        if not os.path.exists(near_datas_dir):
            os.makedirs(near_datas_dir)
        agent = self.sim.get_agent(0)
        next_look_right_left_angle = 0
        self.near_data_id_ = 0
        self.near_pc = o3d.geometry.PointCloud()
        self.ground_pc = o3d.geometry.PointCloud()
        def save_data():
            mn.gl.default_framebuffer.clear(mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH)
            mn.gl.default_framebuffer.bind()
            mn.gl.Renderer.disable(mn.gl.Renderer.Feature.BLENDING)
            obs = self.sim.get_sensor_observations(0)
            depth_np = obs['depth_sensor']
            rgb = (obs['color_sensor'][:,:,:3])/255.0
            img_pts = (self.image_kuv1 * depth_np[:,:,None]).reshape(-1,3)
            img_pc = o3d.geometry.PointCloud()
            img_pc.points = o3d.utility.Vector3dVector(img_pts)
            img_pc.colors = o3d.utility.Vector3dVector(rgb.reshape(-1,3))
            pose = self.get_agent_pose()
            pose[:3,:3] = pose[:3,:3] @ self.rot_ro_cam
            pcd_w = img_pc.transform(pose)
            pcd_w_pts = np.asarray(pcd_w.points)
            pcd_w_colors = np.asarray(pcd_w.colors)
            if self.ground_height is None:
                # ground_mask = (pcd_w_pts[:,1]<pcd_w_pts[:,1].min()+0.1)
                ground_mask = (pcd_w_pts[:,1]<0.15)*(pcd_w_pts[:,1]>-0.05)
                ground_pts_init = pcd_w_pts[ground_mask,:]
                ground_clr_init = pcd_w_colors[ground_mask,:]
                ground_init_pc = o3d.geometry.PointCloud()
                # import pdb;pdb.set_trace()
                ground_init_pc.points = o3d.utility.Vector3dVector(ground_pts_init)
                # ground_init_pc.colors = o3d.utility.Vector3dVector(ground_clr_init)
                # o3d.io.write_point_cloud(os.path.join(near_datas_dir, str(self.near_data_id_).zfill(6)+'_test.ply'), ground_init_pc)
                plane_model, inliers = ground_init_pc.segment_plane(distance_threshold=0.01, ransac_n=10, num_iterations=1000)
                if len(inliers)>0.7*len(ground_pts_init):
                    inlier_cloud = ground_init_pc.select_by_index(inliers)
                    # o3d.io.write_point_cloud(os.path.join(near_datas_dir, str(self.near_data_id_).zfill(6)+'_g.ply'), inlier_cloud)
                    inlier_cloud_pts = np.asarray(inlier_cloud.points)
                    self.ground_height = inlier_cloud_pts[:,1].mean()
                    # plt.figure()
                    # plt.plot(inlier_cloud_pts[:,1])
                    # plt.show()
                    # import pdb;pdb.set_trace()
                    # plt.figure()
                    # plt.imshow(obs['depth_sensor'])
                    # plt.show()
                    pass
            if self.ground_height is not None:
                ground_mask = (pcd_w_pts[:,1]<self.ground_height+0.05)*(pcd_w_pts[:,1]>self.ground_height-0.05)
                pcd_w.points = o3d.utility.Vector3dVector(pcd_w_pts)
                pcd_w.colors = o3d.utility.Vector3dVector(pcd_w_colors*1.0)
                # pcd_w_colors[ground_mask,:] = [0,1,0]
                ground_pc = o3d.geometry.PointCloud()
                ground_pc.points = o3d.utility.Vector3dVector(pcd_w_pts[ground_mask,:])
                # ground_pc.colors = o3d.utility.Vector3dVector(pcd_w_colors[ground_mask,:])
                ground_pc.paint_uniform_color([0,1,0])
                self.ground_pc += ground_pc
                # o3d.io.write_point_cloud(os.path.join(near_datas_dir, str(self.near_data_id_).zfill(6)+'.ply'), pcd_w)
            # np.savetxt(os.path.join(near_datas_dir, str(self.near_data_id_).zfill(6)+'.txt'), self.get_agent_pose())
            # cv2.imwrite(os.path.join(near_datas_dir, str(self.near_data_id_).zfill(6)+'.png'), obs['color_sensor'][:,:,[2,1,0]])
            # np.save(os.path.join(near_datas_dir, str(self.near_data_id_).zfill(6)+'.npy'), obs['depth_sensor'])
            # o3d.io.write_point_cloud(os.path.join(near_datas_dir, str(self.near_data_id_).zfill(6)+'.ply'), pcd_w)
            self.near_pc += pcd_w
            # if self.sim_settings['use_som']:
            #     img_out, mask_map = self.img_seg(obs['color_sensor'][:,:,:3])
            #     cv2.imwrite(os.path.join(near_datas_dir, str(self.near_data_id_).zfill(6)+'_seg.png'), img_out[:,:,[2,1,0]])
            #     np.save(os.path.join(near_datas_dir, str(self.near_data_id_).zfill(6)+'_mask.npy'), mask_map[:,:,0])
            self.near_data_id_ +=1
        if look_down_angle>5:
            pose_ = self.get_agent_pose()
            z_dir_init = pose_[:3,2]*1.0
            max_turn_n = 20
            max_turn_0=0
            while 1:
                pose_ = self.get_agent_pose()
                z_dir_cur = pose_[:3,2]*1.0
                cos_ = z_dir_init.dot(z_dir_cur)
                if cos_>1.0: cos_=1.0
                angle_ = np.arccos(cos_)/np.pi*180
                # print("look_down angle_: ", angle_, " cos_: ",cos_)
                #* 向左向右看
                if angle_>=next_look_right_left_angle and angle_<50:
                    next_look_right_left_angle += delta_look_rl_angle
                    pose_ = self.get_agent_pose()
                    z_dir_init_ = pose_[:3,2]*1.0
                    #* 向右看
                    max_turn_i=0
                    while 1:
                        pose_ = self.get_agent_pose()
                        z_dir_cur = pose_[:3,2]*1.0
                        cos_ = z_dir_init_.dot(z_dir_cur)
                        if cos_>1.0: cos_=1.0
                        angle_1 = np.arccos(cos_)/np.pi*180
                        if angle_1>look_right_left_angle or max_turn_i>max_turn_n:
                            save_data()
                            break
                        next_action = 'turn_right'
                        # print("向右看 turn_right angle_1: ", angle_1)
                        agent.act(next_action)
                        self.draw_event()
                        if self.save_video: self.save_video_data()
                        max_turn_i += 1
                        time.sleep(time_sleep)
                    #* 向左看
                    agent.act('turn_left')
                    self.draw_event()
                    if self.save_video: self.save_video_data()
                    agent.act('turn_left')
                    self.draw_event()
                    if self.save_video: self.save_video_data()
                    max_turn_i=0
                    while 1:
                        pose_ = self.get_agent_pose()
                        z_dir_cur = pose_[:3,2]*1.0
                        cos_ = z_dir_init_.dot(z_dir_cur)
                        if cos_>1.0: cos_=1.0
                        angle_1 = np.arccos(cos_)/np.pi*180
                        if angle_1>look_right_left_angle or max_turn_i>2*max_turn_n:
                            save_data()
                            break
                        next_action = 'turn_left'
                        # print("向左看 turn_left angle_1: ", angle_1)
                        agent.act(next_action)
                        max_turn_i += 1
                        self.draw_event()
                        if self.save_video: self.save_video_data()
                        time.sleep(time_sleep)
                    #* 回到中间
                    max_turn_i=0
                    while 1:
                        pose_ = self.get_agent_pose()
                        z_dir_cur = pose_[:3,2]*1.0
                        cos_ = z_dir_init_.dot(z_dir_cur)
                        if cos_>1.0: cos_=1.0
                        angle_1 = np.arccos(cos_)/np.pi*180
                        if angle_1<5 or max_turn_i>max_turn_n:
                            save_data()
                            break
                        next_action = 'turn_right'
                        # print("回到中间 turn_right angle_1: ", angle_1)
                        agent.act(next_action)
                        self.draw_event()
                        if self.save_video: self.save_video_data()
                        max_turn_i += 1
                        time.sleep(time_sleep)
                if angle_>look_down_angle or max_turn_0>3*max_turn_n:
                    save_data()
                    break
                next_action = 'look_down'
                agent.act(next_action)
                max_turn_0+=1
                self.draw_event()
                if self.save_video: self.save_video_data()
                time.sleep(time_sleep)
            last_angle = angle_
            first_flag = True
            max_turn_0 = 0
            while 1:
                pose_ = self.get_agent_pose()
                z_dir_cur = pose_[:3,2]*1.0
                cos_ = z_dir_init.dot(z_dir_cur)
                angle_ = np.arccos(cos_)/np.pi*180
                # print("look_up angle_: ", angle_)
                if first_flag:
                    last_angle=angle_
                    first_flag = False
                if angle_<5 or angle_>last_angle or max_turn_0>3*max_turn_n:
                    break
                next_action = 'look_up'
                agent.act(next_action)
                self.draw_event()
                if self.save_video: self.save_video_data()
                max_turn_0+=1
                last_angle = angle_
                time.sleep(time_sleep)
        else:
            save_data()
        # o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'near_pc_'+str(image_id).zfill(6)+'_'+str(mask_id).zfill(3)+'.ply'), self.obj_pc_w+self.near_pc)
        # o3d.io.write_point_cloud(os.path.join(near_datas_dir, 'ground_pc_'+str(image_id).zfill(6)+'_'+str(mask_id).zfill(3)+'.ply'), self.obj_pc_w+self.ground_pc)
        pass

    def img_and_mask_id_to_pos(self, rgb_i=4, mask_id=4, xy_scale=0.01, radius=2.0, no_circle=False, angle_scale=45, alpha=0.2, depth_sample_rate=100, marginal_ = 5, max_depth_gap = 0.5):
        if rgb_i>=len(self.rgb_list):
            print("too large image id!")
            return None
        rgb = cv2.imread(self.rgb_list[rgb_i])[:,:,[2,1,0]]
        img_out = cv2.imread(self.seg_list[rgb_i])[:,:,[2,1,0]]
        mask_map = np.load(self.mask_list[rgb_i])
        depth = np.load(self.depth_list[rgb_i])
        seg_i = mask_id if isinstance(mask_id, int) else mask_id[0]
        valid_seg = mask_map==seg_i
        #* mask_map 滤波
        h, w = depth.shape
        valid_seg[:marginal_,:]=0
        valid_seg[-marginal_:,:]=0
        valid_seg[:, :marginal_]=0
        valid_seg[:, -marginal_:]=0
        for i in range(marginal_, h-marginal_):
            for j in range(marginal_, w-marginal_):
                if valid_seg[i,j]:
                    if abs(depth[i,j]-depth[i-marginal_,j-marginal_])>max_depth_gap or abs(depth[i,j]-depth[i-marginal_,j+marginal_])>max_depth_gap:
                        valid_seg[i,j] = 0
                        continue
                    if abs(depth[i,j]-depth[i+marginal_,j-marginal_])>max_depth_gap or abs(depth[i,j]-depth[i+marginal_,j+marginal_])>max_depth_gap:
                        valid_seg[i,j] = 0
                        continue
        # plt.show()
        pc = self.image_kuv1 * depth[:,:,None]
        mask_ = (depth>0.1)
        orig_pc = o3d.geometry.PointCloud()
        orig_pc.points = o3d.utility.Vector3dVector((pc[mask_]).reshape(-1,3))
        orig_pc.colors = o3d.utility.Vector3dVector((rgb[mask_]).reshape(-1,3)/255.0)
        # orig_pc.paint_uniform_color([0,1,0])
        mask_ = (depth>0.1)*(valid_seg)
        obj_pt = (pc[mask_]).reshape(-1,3)
        obj_rgb_f = (rgb[mask_]).reshape(-1,3)/255.0
        pos = self.database_poses[rgb_i,0:3]
        quat = self.database_poses[rgb_i,3:]
        r = Rotation.from_quat(quat)
        rot = r.as_matrix()
        rot = rot @ self.rot_ro_cam
        # pos[1] += camera_height
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos.reshape(-1)
        obj_pc = o3d.geometry.PointCloud()
        obj_pc.points = o3d.utility.Vector3dVector(obj_pt)
        obj_pc.colors = o3d.utility.Vector3dVector(obj_rgb_f)
        obj_pc.paint_uniform_color([1,0,0])
        obj_pc_w = obj_pc.transform(pose)
        self.obj_pc_w = obj_pc_w
        orig_pc_w = orig_pc.transform(pose)
        pcd_w_pts = np.asarray(obj_pc_w.points)
        orig_pc_w_pts = np.asarray(orig_pc_w.points)
        obj_center = pcd_w_pts.sum(0)/(len(pcd_w_pts))
        if no_circle:
            return obj_center
        save_path = os.path.join(self.basedata_path, 'images', str(rgb_i).zfill(6)+'_'+str(mask_id).zfill(3)+'_circle.png')
        radius_int = int(radius/xy_scale)
        xz_grid = np.ones((2*radius_int+1, 2*radius_int+1))
        inside_pts_init = []
        invalid_pos_ = []
        floor_h = orig_pc_w_pts[:,1].min()
        for i in range(len(orig_pc_w_pts)):
            if orig_pc_w_pts[i][1]>floor_h+1.5:
                continue
            dirs = orig_pc_w_pts[i]-obj_center
            dist = np.sqrt(dirs[0]**2+dirs[2]**2)
            if dist>radius:
                continue
            xz_grid_pos = (dirs[[0,2]]/xy_scale).astype(int)
            if orig_pc_w_pts[i][1]>floor_h+xy_scale*2:
                xz_grid[radius_int+xz_grid_pos[0], radius_int+xz_grid_pos[1]] = 0
                invalid_pos_.append([radius_int+xz_grid_pos[0], radius_int+xz_grid_pos[1]])
            else:
                inside_pts_init.append(orig_pc_w_pts[i])
        angle_r = int(180/angle_scale)
        angle_scale = angle_scale/180*np.pi
        inside_pts, inside_pts_colors = [], []
        colors = np.random.uniform(0,1,(2*angle_r, 3))
        num_map = {}
        for i in range(2*angle_r):
            num_map[i]=[]
        min_theta_int, max_theta_int = 0,0
        for i in range(len(inside_pts_init)):
            dirs = inside_pts_init[i]-obj_center
            xz_grid_pos = (dirs[[0,2]]/xy_scale).astype(int)
            if xz_grid[radius_int+xz_grid_pos[0], radius_int+xz_grid_pos[1]] > 0:
                dirs_uni = dirs[[0,2]]/np.linalg.norm(dirs[[0,2]])
                theta_ = np.arccos(dirs_uni[0])
                theta_ *= -1.0 if dirs_uni[1]<0 else 1.0
                theta_int = int(np.floor(theta_/angle_scale))+angle_r
                # min_theta_int = theta_int if theta_int<min_theta_int else min_theta_int
                # max_theta_int = theta_int if theta_int>max_theta_int else max_theta_int
                inside_pts.append(inside_pts_init[i])
                inside_pts_colors.append(colors[theta_int])
                num_map[theta_int].append(len(inside_pts)-1)
        # print('min_theta_int, max_theta_int: ',min_theta_int, max_theta_int)
        inside_pts = np.asarray(inside_pts)
        inside_pts_colors = np.asarray(inside_pts_colors)
        valid_mask = np.ones((len(inside_pts_colors),), dtype=bool)
        cicle_center = {}
        for i in range(2*angle_r):
            if len(num_map[i])<1e3:
                valid_mask[num_map[i]]=0
                cicle_center[i] = None
            else:
                cicle_center[i] = inside_pts[num_map[i]].sum(0)/len(inside_pts[num_map[i]])
            # print("i: ", i, ", num: ", len(num_map[i]), ", center: ", cicle_center[i])
        if os.path.exists(save_path):
            return obj_center, cicle_center
        valid_mask = np.asarray(valid_mask).astype(bool)
        inside_pts_pc = o3d.geometry.PointCloud()
        # o3d.io.write_point_cloud(os.path.join(self.basedata_path, 'images', str(rgb_i).zfill(6)+'_'+str(mask_id).zfill(3)+'_circle.ply'), obj_pc_w+orig_pc_w)
        # import pdb;pdb.set_trace()
        inside_pts_pc.points = o3d.utility.Vector3dVector(inside_pts[valid_mask])
        inside_pts_pc.colors = o3d.utility.Vector3dVector(inside_pts_colors[valid_mask])
        axis_pts = self.get_axis_pts(use_cam=False)
        axis_pc = o3d.geometry.PointCloud()
        axis_pc.points = o3d.utility.Vector3dVector(axis_pts[:,:3])
        axis_pc.colors = o3d.utility.Vector3dVector(axis_pts[:,3:6])
        o3d.io.write_point_cloud(os.path.join(self.basedata_path, 'plys', str(rgb_i).zfill(6)+'.ply'),inside_pts_pc+obj_pc_w+orig_pc_w+axis_pc)
        ext = np.linalg.inv(pose)
        inside_l_pts = (ext[:3,:3] @ inside_pts[:,:3].T + ext[:3,3:]).T
        img_uvd = (self.cam_mat @ inside_l_pts[:,:3].T).T
        valid_mask = valid_mask*(img_uvd[:,2]>1e-8)
        img_uv1 = (img_uvd[:,:2]/(1e-8+img_uvd[:,2:3])).astype(int)
        valid_mask = valid_mask*(img_uv1[:,1]>=0)*(img_uv1[:,1]<h)*(img_uv1[:,0]>=0)*(img_uv1[:,0]<w)
        img_uv1 = img_uv1[valid_mask,:]
        inside_l_colors = (inside_pts_colors[valid_mask,:3]*255).astype(np.uint8)
        rgb_circle = rgb.copy()
        rgb_circle[img_uv1[:,1],img_uv1[:,0],:3] = (rgb_circle[img_uv1[:,1],img_uv1[:,0],:3]*(1-alpha)+inside_l_colors*alpha).astype(np.uint8)
        for i in range(2*angle_r):
            if cicle_center[i] is not None:
                cicle_center_tmp = ext[:3,:3] @ cicle_center[i].reshape(-1,1) + ext[:3,3:]
                uvd = self.cam_mat @ cicle_center_tmp
                img_uv1 = (uvd[:2]/(1e-8+uvd[2:3])).astype(int)
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_position = (int(img_uv1[0]), int(img_uv1[1]))
                font_scale = 1
                font_color = (255, 0, 0)
                line_type = 2
                cv2.putText(rgb_circle, str(i), text_position, font, font_scale, font_color, line_type)
        # plt.figure()
        # plt.imshow(img_out)
        # plt.figure()
        # plt.imshow(mask_map)
        # plt.figure()
        # plt.imshow(rgb_circle)
        # plt.show()
        cv2.imwrite(save_path, rgb_circle[:,:,[2,1,0]])
        # o3d.visualization.draw_geometries([pcd_w])
        # import pdb;pdb.set_trace()
        # plt.show()
        return obj_center, cicle_center

    def gaze_target(self, obj_center, look_down_angle=0, sleep=0):
        print("========= Start Gaze ==========")
        agent = self.sim.get_agent(0)
        turn_left = True
        first_flag = True
        #* 偏航调整
        while True:
            if sleep>0: time.sleep(sleep)
            next_action = None
            pose_ = self.get_agent_pose()
            dir_ = self.rot_ro_cam.T@(pose_[:3,:3].T@(np.asarray(obj_center)-pose_[:3,3]))
            dir_[1] = 0
            dir_ /= np.linalg.norm(dir_)
            # z_dir = pose_[:3,2]*1.0
            z_dir = np.array([0,0,1.0]) #* 相机坐标系下的前方
            z_dir[1] = 0.0
            z_dir /= np.linalg.norm(z_dir)
            cos_ = z_dir.dot(dir_)
            angle_ = np.arccos(cos_)/np.pi*180
            cross_ = np.cross(z_dir, dir_)[1] #* 相机坐标系：x向右，y向下，z向前
            # print('偏航调整 angle_: ', angle_, " cross_: ",cross_)
            # print("z_dir: ",z_dir, " dir_: ", dir_)
            if angle_<5:
                break
            if first_flag:
                if cross_>0:
                    turn_left = False
                first_flag = False
                # import pdb;pdb.set_trace()
            if cross_>0:
                if turn_left:
                    break
                next_action = 'turn_right'
            if cross_<0:
                if not turn_left:
                    break
                next_action = 'turn_left'
            # import pdb;pdb.set_trace()
            # print(next_action)
            # break
            agent.act(next_action)
            self.draw_event()
            if self.save_video: self.save_video_data()
        #* 俯仰调整
        look_down = True
        first_flag = True
        while True:
            # time.sleep(0.01)
            if sleep>0: time.sleep(sleep)
            next_action = None
            pose_ = self.get_agent_pose()
            # dir_ = np.asarray(obj_center)-pose_[:3,3]
            dir_ = self.rot_ro_cam.T@(pose_[:3,:3].T@(np.asarray(obj_center)-pose_[:3,3]))
            dir_[0] = 0
            dir_ /= np.linalg.norm(dir_)
            # z_dir = pose_[:3,2]*1.0
            z_dir = np.array([0,0,1.0]) #* 相机坐标系下的前方
            z_dir[0] = 0.0
            z_dir /= np.linalg.norm(z_dir)
            cos_ = z_dir.dot(dir_)
            angle_ = np.arccos(cos_)/np.pi*180
            cross_ = np.cross(z_dir, dir_)[0] #* 相机坐标系：x向右，y向下，z向前
            # print('俯仰调整 angle_: ', angle_, " cross_: ",cross_)
            # print("z_dir: ",z_dir, " dir_: ", dir_)
            if angle_<5:
                break
            if first_flag:
                if cross_>0:
                    look_down = False
                first_flag = False
                # import pdb;pdb.set_trace()
            if cross_>0:
                if look_down:
                    break
                next_action = 'look_up'
            if cross_<0:
                if not look_down:
                    break
                next_action = 'look_down'
            # import pdb;pdb.set_trace()
            if next_action is None:
                break
            # break
            # print(next_action)
            agent.act(next_action)
            self.draw_event()
            if self.save_video: self.save_video_data()
        print("========= End Gaze ==========")
        if look_down_angle>5:
            pose_ = self.get_agent_pose()
            z_dir_init = pose_[:3,2]*1.0
            while 1:
                pose_ = self.get_agent_pose()
                z_dir_cur = pose_[:3,2]*1.0
                cos_ = z_dir_init.dot(z_dir_cur)
                angle_ = np.arccos(cos_)/np.pi*180
                if angle_>look_down_angle:
                    break
                next_action = 'look_down'
                agent.act(next_action)
                self.draw_event()
                if self.save_video: self.save_video_data()

    def get_near_navigable_pos(self, goal_pos, start_dist=0.01):
        for i in range(100):
            new_goal = self.pathfinder.get_random_navigable_point_near(goal_pos, start_dist*(i+1))
            if not np.isnan(new_goal).any():
                return new_goal
        return None

    def navigate_to_target(self, obj_center, dist_near=2.0, iter_max=10, rand_goal = False, time_sleep=0.1, move_steps_scale=-1):
        print("========= Start navigation ==========")
        agent_state = self.sim.get_agent(0).state.sensor_states['depth_sensor']
        agent = self.sim.get_agent(0)
        follower = habitat_sim.GreedyGeodesicFollower(self.pathfinder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right",)
        follower.reset()
        path = habitat_sim.ShortestPath()
        if rand_goal:
            for i in range(iter_max):
                goal_pos = self.pathfinder.get_random_navigable_point_near(obj_center, dist_near)
                path.requested_start = agent_state.position
                path.requested_end = goal_pos
                if self.pathfinder.find_path(path) and path.geodesic_distance > dist_near:
                    break
                if i%3==0:
                    dist_near*=2
                if i==iter_max-1:
                    state = habitat_sim.AgentState()
                    while True:
                        state.position = self.pathfinder.get_random_navigable_point()
                        goal_pos = self.pathfinder.get_random_navigable_point_near(obj_center, dist_near)
                        path = habitat_sim.ShortestPath()
                        path.requested_start = state.position
                        path.requested_end = goal_pos
                        if self.pathfinder.find_path(path) and path.geodesic_distance > 2.0:
                            break
                    print("There is no path in current position, reset the position !!")
        else:
            goal_pos = obj_center
            path.requested_start = agent_state.position
            path.requested_end = goal_pos
        num_acts = 0
        try:
            action_list = follower.find_path(goal_pos)
        except habitat_sim.errors.GreedyFollowerError:
            action_list = [None]
            # print('habitat_sim.errors.GreedyFollowerError action_list:', action_list)
            #* ['build_navmesh_vertex_indices', 'build_navmesh_vertices', 'closest_obstacle_surface_point', 'distance_to_closest_obstacle', 'find_path', 'get_bounds', 'get_island', 'get_random_navigable_point', 'get_random_navigable_point_near', 'get_topdown_island_view', 'get_topdown_view', 'is_loaded', 'is_navigable', 'island_area', 'island_radius', 'load_nav_mesh', 'nav_mesh_settings', 'navigable_area', 'num_islands', 'save_nav_mesh', 'seed', 'snap_point', 'try_step', 'try_step_no_sliding']
            # self.pathfinder.save_nav_mesh('./asdf.navmesh')
            self.pathfinder.is_navigable(goal_pos)
            # find_path = True
            for i in range(10):
                find_path = True
                new_goal = self.get_near_navigable_pos(goal_pos)
                if new_goal is not None:
                    # print("Try with new goal: ",new_goal)
                    try:
                        action_list = follower.find_path(new_goal)
                    except habitat_sim.errors.GreedyFollowerError:
                        find_path = False
                if find_path: break
            if not find_path and len(action_list)<2:
                print('habitat_sim.errors.GreedyFollowerError action_list:', action_list)
                return False
        if move_steps_scale>0 and move_steps_scale<1.0:
            n_steps = len(action_list)
            # print("n_steps: ", n_steps)
            move_steps_n = int(n_steps*move_steps_scale)
            action_list = action_list[:move_steps_n]
            action_list.append(None)
        last_state = self.get_state_np()
        while True:
            time.sleep(time_sleep)
            next_action = action_list[0]
            action_list = action_list[1:]
            if next_action is None: #* 初步导航结束，指向目标物体
                break
                # import pdb;pdb.set_trace()
            agent.act(next_action)
            self.draw_event()
            if self.save_video: self.save_video_data()
            if self.save_video:
                pass
            cur_state = self.get_state_np()
            self.path_length += np.linalg.norm(cur_state[:3]-last_state[:3])
            last_state = cur_state
            # print("random navigation test----next_action: ", next_action)
            # print('get_agent_pose_euler: ', self.get_agent_pose_euler())
            # print("dist to goal: ", np.linalg.norm(self.get_agent_pose_euler()[:3]-goal_pos))
            num_acts += 1
            if num_acts > 1e4:
                break
        print("========= End navigation ==========")
        return True

    def get_cur_top_down_view(self, height_=15):
        self.enter_press = True
        agent_state = self.sim.get_agent(0).state.sensor_states['depth_sensor']
        orig_position = agent_state.position*1.0
        orig_rotation = agent_state.rotation*1.0
        # print('get_agent_pose_euler0: ', self.get_agent_pose_euler(), ", orig_position: ", orig_position)
        # import pdb;pdb.set_trace()
        if isinstance(agent_state.position ,np.ndarray):
            agent_state.position[1] = height_
        else:
            agent_state.position.y = height_
        agent_state.rotation = habitat_sim.utils.quat_from_angle_axis(np.pi / 2, np.array([-1, 0, 0]))
        self.sim.get_agent(0).set_state(agent_state)
        obs = self.sim.get_sensor_observations(0)
        self.top_down_view = obs["color_sensor"]
        if len(self.poses_pts)>0:
            width = int(self.sim_settings["width"])
            height = int(self.sim_settings["height"])
            pose = np.zeros(7)
            pose[:3] = agent_state.position
            # pose[1] *=-1
            # pose[3:] = np.asarray([agent_state.rotation.x, -1*agent_state.rotation.y, agent_state.rotation.z, agent_state.rotation.w])
            pose[3:] = np.asarray([agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z, agent_state.rotation.w])
            extrinsic = np.eye(4)
            extrinsic[:3,3] = pose[:3]
            extrinsic[:3,:3] = Rotation.from_quat(pose[3:]).as_matrix()
            rot_ro_cam = np.eye(3)
            rot_ro_cam[1, 1] = -1
            rot_ro_cam[2, 2] = -1
            extrinsic[:3,:3] = extrinsic[:3,:3] @ rot_ro_cam
            extrinsic_inv = np.linalg.inv(extrinsic)
            self.extrinsic_inv = extrinsic_inv.copy()
            poses_pts_ = np.vstack(self.poses_pts)
            poses_pts_[:,:3] = (extrinsic_inv[:3,:3] @ poses_pts_[:,:3].T + extrinsic_inv[:3,3:]).T
            # import pdb;pdb.set_trace()
            img_uvd = (self.cam_intrinsics @ poses_pts_[:,:3].T).T
            valid_mask = img_uvd[:,2]>1e-8
            img_uv1 = (img_uvd[:,:2]/(1e-8+img_uvd[:,2:3])).astype(int)
            valid_mask = valid_mask*(img_uv1[:,1]>=0)*(img_uv1[:,1]<height)*(img_uv1[:,0]>=0)*(img_uv1[:,0]<width)
            img_uv1 = img_uv1[valid_mask,:]
            colors = (poses_pts_[valid_mask,3:]*255.0).astype(np.uint8)
            self.top_down_view[img_uv1[:,1],img_uv1[:,0],:3] = colors
            self.image_in[:] = self.top_down_view[:, :, :3].copy()
            self.top_down_view_update_flag[:] = True
            # cv2.imwrite('', self.top_down_view)
            # import pdb;pdb.set_trace()
        agent_state.position = orig_position
        agent_state.rotation = orig_rotation
        # print('get_agent_pose_euler1: ', self.get_agent_pose_euler(), ", orig_position: ", orig_position)
        self.sim.get_agent(0).set_state(agent_state)
        # print(self.sim.get_agent(0).state.sensor_states['depth_sensor'])
        # obs = self.sim.get_sensor_observations(0)
        # print('get_agent_pose_euler2: ', self.get_agent_pose_euler(), ", orig_position: ", orig_position)
        self.draw_event()
        # print('get_agent_pose_euler3: ', self.get_agent_pose_euler())
        # self.sim.step("turn_left")
        # self.sim.step("turn_right")
        self.enter_press = False

    def top_down_view_update_func(self, dist_ = 0.01):
        agent_state = self.sim.get_agent(0).state.sensor_states['depth_sensor']
        last_position = np.asarray(agent_state.position.copy())
        rot_ro_cam = np.eye(3)
        rot_ro_cam[1, 1] = -1
        rot_ro_cam[2, 2] = -1
        width = int(self.sim_settings["width"])
        height = int(self.sim_settings["height"])
        while not self.threads_end:
            agent_state = self.sim.get_agent(0).state.sensor_states['depth_sensor']
            cur_position = np.asarray(agent_state.position.copy())
            if self.extrinsic_inv is not None and not self.enter_press:
                # self.get_cur_top_down_view()
                # print("self.get_cur_top_down_view()")
                pose = np.zeros(7)
                pose[:3] = agent_state.position
                pose[3:] = np.asarray([agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z, agent_state.rotation.w])
                extrinsic = np.eye(4)
                extrinsic[:3,3] = pose[:3]
                extrinsic[:3,:3] = Rotation.from_quat(pose[3:]).as_matrix()
                extrinsic[:3,:3] = extrinsic[:3,:3] @ rot_ro_cam
                axis_pts = self.get_axis_pts()
                axis_pts[:,:3] = (extrinsic[:3,:3] @ axis_pts[:,:3].T + extrinsic[:3,3:]).T
                extrinsic_inv = self.extrinsic_inv.copy()
                # poses_pts_ = np.vstack(self.poses_pts)
                axis_pts[:,:3] = (extrinsic_inv[:3,:3] @ axis_pts[:,:3].T + extrinsic_inv[:3,3:]).T
                img_uvd = (self.cam_intrinsics @ axis_pts[:,:3].T).T
                valid_mask = img_uvd[:,2]>1e-8
                img_uv1 = (img_uvd[:,:2]/(1e-8+img_uvd[:,2:3])).astype(int)
                valid_mask = valid_mask*(img_uv1[:,1]>=0)*(img_uv1[:,1]<height)*(img_uv1[:,0]>=0)*(img_uv1[:,0]<width)
                img_uv1 = img_uv1[valid_mask,:]
                colors = axis_pts[valid_mask,3:]
                cam_pts_ = np.hstack([img_uv1, colors])
                # print(cam_pts_.shape)
                # self.top_down_view[img_uv1[:,1],img_uv1[:,0],:3] = colors
                # self.image_in[:] = self.top_down_view[:, :, :3].copy()
                if len(cam_pts_)<400:
                    print("Out !!")
                    continue
                    # self.get_cur_top_down_view()
                self.cam_pts[:] = cam_pts_
                last_position = cur_position.copy()
            time.sleep(0.1)

    def top_down_view_show_func(self,img,cam_pts,top_down_view_update_flag):
        self.img_name = 'top_down_view'
        # 初始化图像
        fig, ax = plt.subplots()
        # width = int(self.sim_settings["width"])
        # height = int(self.sim_settings["height"])
        # self.top_down_view = np.random.random((10, 10,3))  # 一个初始的空白图像
        # self.get_cur_top_down_view()
        # im = ax.imshow(self.top_down_view)
        im = ax.imshow(img)
        ax.set_title(self.img_name)
        scatter = ax.scatter([], [], s=1)
        # 更新图像的函数
        def update(frame):
            # 这里随机生成新的图像数据模拟实时更新
            # im.set_data(self.top_down_view)
            if top_down_view_update_flag:
                scatter.set_offsets(np.empty((0, 2)))
                scatter.set_color(np.empty((0, 3)))
                im.set_data(img)
                top_down_view_update_flag[:] = False
                # print("top_down_view_update_flag")
            if cam_pts.nonzero()[0].shape[0]>0:
                scatter.set_offsets(np.c_[cam_pts[:,0], cam_pts[:,1]])  # 更新点的位置
                scatter.set_color(cam_pts[:,2:5]) 
        # 创建动画
        self.ani = FuncAnimation(fig, update, frames=np.arange(100), interval=200)
        plt.show()
        print("top_down_view_show_func end")
        # 在需要的时候停止动画
        # self.ani.event_source.stop()
        pass

    def get_axis_pts(self, num=50, length=0.5, use_cam=True, cam_radius=0.5, cam_len=1.0):
        if use_cam:
            cam_pts = []
            ones = np.linspace(0, 1.0, num)
            far_plane_corners = np.array([
                    [cam_radius, cam_radius, cam_len],  # 远平面右上角
                    [-cam_radius, cam_radius, cam_len],  # 远平面左上角
                    [-cam_radius, -cam_radius, cam_len],  # 远平面左下角
                    [cam_radius, -cam_radius, cam_len]  # 远平面右下角
                ])
            #* 绘制原点到远点的线
            for i in range(4):
                cam_pts.append(far_plane_corners[i:i+1,:]*ones.reshape(-1,1))
            cam_pts = np.vstack(cam_pts)
            len1 = len(cam_pts)
            cam_pts = [cam_pts]
            for i in range(4):
                cam_pts.append(far_plane_corners[i,:].reshape(1,-1)+(far_plane_corners[(i+1)%4,:].reshape(1,-1) - far_plane_corners[i,:].reshape(1,-1))*ones.reshape(-1,1))
            cam_pts_np = np.vstack(cam_pts)
            cam_pts_cl = np.zeros_like(cam_pts_np)
            cam_pts_cl[len1:,0] = 1.0
            cam_pts_cl[:len1,2] = 1.0
            return np.hstack([cam_pts_np, cam_pts_cl])
        ones = np.linspace(0,length, num)
        x_pts = np.zeros((num,3))
        y_pts = np.zeros((num,3))
        z_pts = np.zeros((num,3))
        x_colors = np.zeros_like(x_pts)
        y_colors = np.zeros_like(y_pts)
        z_colors = np.zeros_like(z_pts)
        x_pts[:,0] = ones*1.0
        y_pts[:,1] = ones*1.0
        z_pts[:,2] = ones*1.0
        x_colors[:,0] = 1.0
        y_colors[:,1] = 1.0
        z_colors[:,2] = 1.0
        axis_pts = np.vstack([x_pts, y_pts, z_pts])
        axis_clrs = np.vstack([x_colors, y_colors, z_colors])
        return np.hstack([axis_pts, axis_clrs])

    def get_state_np(self):
        # state_ = self.sim.get_agent(0).get_state()
        state_ = self.sim.get_agent(0).state.sensor_states['depth_sensor']
        pose = np.zeros(7)
        pose[:3] = state_.position
        # pose[1] *=-1
        # pose[3:] = np.asarray([agent_state.rotation.x, -1*agent_state.rotation.y, agent_state.rotation.z, agent_state.rotation.w])
        pose[3:] = np.asarray([state_.rotation.x, state_.rotation.y, state_.rotation.z, state_.rotation.w])
        return  pose

    def path_plan(self, start_pose, goal_pose):
        agent = self.sim.get_agent(0)
        follower = habitat_sim.GreedyGeodesicFollower(self.pathfinder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right",)
        follower.reset()
        path = habitat_sim.ShortestPath()
        path.requested_start = start_pose[:3]*1.0
        path.requested_end = goal_pose[:3]*1.0
        if not self.pathfinder.find_path(path):
            return -1
        return path.geodesic_distance

    def key_press_event(self, event: Application.KeyEvent) -> None:
        """
        Handles `Application.KeyEvent` on a key press by performing the corresponding functions.
        If the key pressed is part of the movement keys map `Dict[KeyEvent.key, Bool]`, then the
        key will be set to False for the next `self.move_and_look()` to update the cur_node actions.
        """
        key = event.key
        pressed = Application.KeyEvent.Key
        mod = Application.InputEvent.Modifier
        shift_pressed = bool(event.modifiers & mod.SHIFT)
        alt_pressed = bool(event.modifiers & mod.ALT)
        # warning: ctrl doesn't always pass through with other key-presses
        if key == pressed.ESC:
            event.accepted = True
            self.threads_end = True
            print("=======Save Point Cloud===========")
            if len(np.asarray(self.pc_all.points))>100:
                o3d.io.write_point_cloud(os.path.join(self.save_path, 'pc_all.ply'), self.pc_all)
            print("=======ESC===========")
            # self.ani.event_source.stop()
            self.process.join()
            # self.top_down_view_show_thread.join()
            self.top_down_view_update_thread.join()
            self.exit_event(Application.ExitEvent)
            return
        elif key == pressed.H:
            self.print_help_text()
        elif key == pressed.TAB:
            # NOTE: (+ALT) - reconfigure without cycling scenes
            if not alt_pressed:
                # cycle the active scene from the set available in MetadataMediator
                inc = -1 if shift_pressed else 1
                scene_ids = self.sim.metadata_mediator.get_scene_handles()
                cur_scene_index = 0
                if self.sim_settings["scene"] not in scene_ids:
                    matching_scenes = [
                        (ix, x)
                        for ix, x in enumerate(scene_ids)
                        if self.sim_settings["scene"] in x
                    ]
                    if not matching_scenes:
                        logger.warning(
                            f"The cur_node scene, '{self.sim_settings['scene']}', is not in the list, starting cycle at index 0."
                        )
                    else:
                        cur_scene_index = matching_scenes[0][0]
                else:
                    cur_scene_index = scene_ids.index(self.sim_settings["scene"])

                next_scene_index = min(
                    max(cur_scene_index + inc, 0), len(scene_ids) - 1
                )
                self.sim_settings["scene"] = scene_ids[next_scene_index]
            self.reconfigure_sim()
            logger.info(
                f"Reconfigured simulator for scene: {self.sim_settings['scene']}"
            )
        elif key == pressed.SPACE:
            if not self.sim.config.sim_cfg.enable_physics:
                logger.warn("Warning: physics was not enabled during setup")
            else:
                self.simulating = not self.simulating
                logger.info(f"Command: physics simulating set to {self.simulating}")
        elif key == pressed.PERIOD:
            if self.simulating:
                logger.warn("Warning: physics simulation already running")
            else:
                self.simulate_single_step = True
                logger.info("Command: physics step taken")
        elif key == pressed.COMMA:
            self.debug_bullet_draw = not self.debug_bullet_draw
            logger.info(f"Command: toggle Bullet debug draw: {self.debug_bullet_draw}")
        elif key == pressed.C:
            if shift_pressed:
                self.contact_debug_draw = not self.contact_debug_draw
                logger.info(
                    f"Command: toggle contact debug draw: {self.contact_debug_draw}"
                )
            else:
                # perform a discrete collision detection pass and enable contact debug drawing to visualize the results
                logger.info(
                    "Command: perform discrete collision detection and visualize active contacts."
                )
                self.sim.perform_discrete_collision_detection()
                self.contact_debug_draw = True
        elif key == pressed.T:
            # load URDF
            fixed_base = alt_pressed
            urdf_file_path = ""
            if shift_pressed and self.cached_urdf:
                urdf_file_path = self.cached_urdf
            else:
                urdf_file_path = input("Load URDF: provide a URDF filepath:").strip()

            if not urdf_file_path:
                logger.warn("Load URDF: no input provided. Aborting.")
            elif not urdf_file_path.endswith((".URDF", ".urdf")):
                logger.warn("Load URDF: input is not a URDF. Aborting.")
            elif os.path.exists(urdf_file_path):
                self.cached_urdf = urdf_file_path
                aom = self.sim.get_articulated_object_manager()
                ao = aom.add_articulated_object_from_urdf(
                    urdf_file_path,
                    fixed_base,
                    1.0,
                    1.0,
                    True,
                    maintain_link_order=False,
                    intertia_from_urdf=False,
                )
                ao.translation = (
                    self.default_agent.scene_node.transformation.transform_point(
                        [0.0, 1.0, -1.5]
                    )
                )
                # check removal and auto-creation
                joint_motor_settings = habitat_sim.physics.JointMotorSettings(
                    position_target=0.0,
                    position_gain=1.0,
                    velocity_target=0.0,
                    velocity_gain=1.0,
                    max_impulse=1000.0,
                )
                existing_motor_ids = ao.existing_joint_motor_ids
                for motor_id in existing_motor_ids:
                    ao.remove_joint_motor(motor_id)
                ao.create_all_motors(joint_motor_settings)
            else:
                logger.warn("Load URDF: input file not found. Aborting.")
        elif key == pressed.M:
            self.cycle_mouse_mode()
            logger.info(f"Command: mouse mode set to {self.mouse_interaction}")
        elif key == pressed.V:
            self.invert_gravity()
            logger.info("Command: gravity inverted")
        elif key == pressed.N:
            # (default) - toggle navmesh visualization
            # NOTE: (+ALT) - re-sample the agent position on the NavMesh
            # NOTE: (+SHIFT) - re-compute the NavMesh
            if alt_pressed:
                logger.info("Command: resample agent state from navmesh")
                if self.pathfinder.is_loaded:
                    new_agent_state = habitat_sim.AgentState()
                    new_agent_state.position = (self.pathfinder.get_random_navigable_point())
                    new_agent_state.rotation = quat_from_angle_axis(self.sim.random.uniform_float(0, 2.0 * np.pi),np.array([0, 1, 0]),)
                    self.default_agent.set_state(new_agent_state)
                else:
                    logger.warning("NavMesh is not initialized. Cannot sample new agent state.")
            elif shift_pressed:
                logger.info("Command: recompute navmesh")
                self.navmesh_config_and_recompute()
            else:
                if self.pathfinder.is_loaded:
                    self.sim.navmesh_visualization = not self.sim.navmesh_visualization
                    logger.info("Command: toggle navmesh")
                else:
                    logger.warn("Warning: recompute navmesh first")
        elif key == pressed.R: #* 随机导航点查看
            if self.get_random_navigable_points_id<len(self.get_random_navigable_points):
                pose_ = self.get_random_navigable_points[self.get_random_navigable_points_id]
                agent_state = habitat_sim.AgentState()
                agent_state.position = pose_[:3]*1.0
                agent_state.rotation = pose_[3:]*1.0
                agent_state.position[1] -= self.sim_settings["sensor_height"]
                #* 因为有旋转，而且采集的是相机的位姿，所以还要沿着y轴减1.5
                # R0 = Rotation.from_quat(np.asarray(data_l3mvn['agent_state'][3:])).as_matrix()
                # y_dir = R0[:3,1]
                # agent_state.position -= y_dir*(self.sim_settings["sensor_height"]-0.88*0.0)
                self.sim.get_agent(0).set_state(agent_state)
                self.draw_event()
                self.get_random_navigable_points_id += 1
            pass
        elif key == pressed.B: #* 随机游走并生成可导航点
            agent = self.sim.get_agent(0)
            # TURN_DEGREE = 1
            # agent.agent_config.action_space["turn_left"].actuation.amount = TURN_DEGREE
            # agent.agent_config.action_space["turn_right"].actuation.amount = TURN_DEGREE
            follower = habitat_sim.GreedyGeodesicFollower(self.pathfinder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right",)
            # result = self.sim.step('move_forward')
            # print('get_agent_pose_euler: ', self.get_agent_pose_euler())
            for _ in range(1):
                follower.reset()
                state = habitat_sim.AgentState()
                while True:
                    state.position = self.pathfinder.get_random_navigable_point()
                    goal_pos = self.pathfinder.get_random_navigable_point()
                    path = habitat_sim.ShortestPath()
                    path.requested_start = state.position
                    path.requested_end = goal_pos
                    if self.pathfinder.find_path(path) and path.geodesic_distance > 2.0 and abs(goal_pos[1]-0.1)<0.2  and abs(state.position[1]-0.1)<0.2:
                        break
                agent.state = state
                failed = False
                gt_geo = path.geodesic_distance
                agent_distance = 0.0
                last_xyz = state.position
                num_acts = 0
                try:
                    action_list = follower.find_path(goal_pos)
                except habitat_sim.errors.GreedyFollowerError:
                    action_list = [None]
                    print('action_list:', action_list)
                    continue
                while True:
                    # If there is action noise, we need to plan a single action, actually take it, and repeat
                    next_action = action_list[0]
                    action_list = action_list[1:]
                    if next_action is None:
                        break
                    agent.act(next_action)
                    # self.sim.step(next_action)
                    self.draw_event()
                    print("random navigation test----next_action: ", next_action)
                    print('get_agent_pose_euler: ', self.get_agent_pose_euler())
                    agent_distance += float(np.linalg.norm(last_xyz - agent.state.position))
                    last_xyz = agent.state.position
                    num_acts += 1
                    if num_acts%5==0 and len(self.get_random_navigable_points)<50:
                        self.get_random_navigable_points.append(self.get_state_np())
                    if num_acts > 1e4:
                        break
                end_state = agent.state
                path.requested_start = end_state.position
                self.pathfinder.find_path(path)
                failed = path.geodesic_distance > follower.forward_spec.amount
                spl = float(not failed) * gt_geo / max(gt_geo, agent_distance)
                print("failed: ",failed,', spl: ', spl)
                # if len(self.get_random_navigable_points)<50:
                    # self.get_random_navigable_points.append(self.get_state_np())
                print("len(self.get_random_navigable_points): ", len(self.get_random_navigable_points))
                np.savetxt(os.path.join(self.save_path, 'navigable_points.txt'), np.asarray(self.get_random_navigable_points))
        elif key == pressed.G:
            # {'plant', 'chair', 'toilet', 'sofa', 'tv_monitor', 'bed'}
            # goal_name_ = 'tv_monitor'
            if self.goal_names_basic_id < len(self.goal_names_basic):
                goal_name_ = self.goal_names_basic[self.goal_names_basic_id]
                if goal_name_ in self.goal_name2id.keys() and goal_name_ in self.target_name2id.keys() and self.l3mvn_data_id<len(self.goal_name2id[goal_name_]):
                    data_l3mvn = self.l3mvn_data[self.goal_name2id[goal_name_][self.l3mvn_data_id]]
                    # import pdb;pdb.set_trace()
                    agent_state = habitat_sim.AgentState()
                    agent_state.position = data_l3mvn['agent_state'][:3]*1.0
                    agent_state.rotation = data_l3mvn['agent_state'][3:]*1.0
                    # agent_state.position[1] -= self.sim_settings["sensor_height"] -0.88
                    #* 因为有旋转，而且采集的是相机的位姿，所以还要沿着y轴减1.5
                    R0 = Rotation.from_quat(np.asarray(data_l3mvn['agent_state'][3:])).as_matrix()
                    y_dir = R0[:3,1]
                    # agent_state.position -= y_dir*(self.sim_settings["sensor_height"]-0.88*0.0) #* navi2gaze_data
                    agent_state.position -= y_dir*(self.sim_settings["sensor_height"]*0.0-0.88*0.0) #* vlmaps_data l3mvn_data semexp_data
                    self.sim.get_agent(0).set_state(agent_state)
                    self.draw_event()
                    # self.set_state_np(data_l3mvn['agent_state'], l3mvn=0.0)
                    print(self.l3mvn_data_id, data_l3mvn)
                    target_ids = self.target_name2id[goal_name_]
                    if data_l3mvn['success']==1:
                        goal_pos = data_l3mvn['agent_state'][:3]
                        best_dist = 1000.0
                        best_angle = 0
                        for target_id in target_ids:
                            optimal_target_pos = self.target_data[target_id]['optimal_pos']
                            target_center = self.target_data[target_id]['center']
                            dist_tmp = goal_pos-optimal_target_pos
                            dist_tmp[1] = 0
                            dist = np.linalg.norm(dist_tmp)
                            dir1 = goal_pos-target_center
                            dir2 = optimal_target_pos-target_center
                            dir1[1]=dir2[1]=0
                            angle = np.arccos(dir1.dot(dir2)/np.linalg.norm(dir2)/np.linalg.norm(dir1))/np.pi*180
                            geodesic_distance = self.path_plan(data_l3mvn["init_agent_state"], data_l3mvn['agent_state'])
                            # import pdb;pdb.set_trace()
                            if target_id == 0:
                                best_dist = dist
                                best_angle = angle
                            else:
                                if dist<best_dist:
                                    best_dist = dist
                                    best_angle = angle
                        print("name: ", goal_name_, ", best_dist: ", best_dist, ", best_angle: ", best_angle)
                    # import pdb;pdb.set_trace()
                    self.l3mvn_data_id += 1
                else:
                    self.goal_names_basic_id += 1
                    self.l3mvn_data_id = 0
            else:
                print("End of l3mvn_data")
            # target, exe = self.navigation_targets[self.navigation_target_id]
            # self.gpt_test(target, exe)
            # self.navigation_target_id += 1
            # self.gpt_test(object_name='sofa', exe="sit on the sofa", image_id = 4, mask_id = 8, use_gpt = False, circle_id=5)
            pass
        elif key == pressed.ENTER:
            # logger.info("Command: resample agent state from navmesh")
            self.enter_press = True
            #* 不加这几句渲染出来的深度会有问题！！！
            mn.gl.default_framebuffer.clear(mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH)
            mn.gl.default_framebuffer.bind()
            mn.gl.Renderer.disable(mn.gl.Renderer.Feature.BLENDING)
            obs = self.sim.get_sensor_observations(0)
            agent_state = self.sim.get_agent(0).state.sensor_states['depth_sensor'] #* 实际只有0.1的高度，这里的位姿是传感器的高度！！！！
            pose = np.zeros(7)
            pose[:3] = agent_state.position
            # pose[1] *=-1
            # pose[3:] = np.asarray([agent_state.rotation.x, -1*agent_state.rotation.y, agent_state.rotation.z, agent_state.rotation.w])
            pose[3:] = np.asarray([agent_state.rotation.x, agent_state.rotation.y, agent_state.rotation.z, agent_state.rotation.w])
            self.poses.append(pose)
            if 1:
                if not os.path.exists(os.path.join(self.save_path, 'intrinsic.txt')):
                    width = int(self.sim_settings["width"])
                    height = int(self.sim_settings["height"])
                    hfov = math.radians(self.cfg.agents[0].sensor_specifications[0].hfov)
                    cam_intrinsics = np.eye(3)
                    cam_intrinsics[0,0] = width / (2.0 * math.tan(hfov / 2.0))
                    # cam_intrinsics[1,1] = cam_intrinsics[0,0] * (height / width)
                    cam_intrinsics[1,1] = cam_intrinsics[0,0]
                    cam_intrinsics[0,2] = width/2
                    cam_intrinsics[1,2] = height/2
                    np.savetxt(os.path.join(self.save_path, 'intrinsic.txt'), cam_intrinsics)
                np.savetxt(os.path.join(self.save_path, 'poses.txt'), np.asarray(self.poses))
                cv2.imwrite(os.path.join(self.save_path, 'images', str(self.save_img_id).zfill(6)+'.png'), obs['color_sensor'][:,:,[2,1,0]])
                np.save(os.path.join(self.save_path, 'depths', str(self.save_img_id).zfill(6)+'.npy'), obs['depth_sensor'])
                if self.sim_settings['use_som']:
                    # pts = torch.from_numpy(np.asarray(pcd_w.points)).float()
                    # param = torch.tensor([1,2,3,4]).float()
                    # OctreeMap.add_pts_with_attr_cpu(pts, param)
                    # data = OctreeMap.get_data()
                    # data2 = OctreeMap.nearest_search(pts, param)
                    img_out, mask_map = self.img_seg(obs['color_sensor'][:,:,:3], slider=1.2, alpha=0.3)
                    # plt.figure()
                    # plt.imshow(img_out)
                    # plt.figure()
                    # plt.imshow(mask_map[:,:,0])
                    # plt.show()
                    cv2.imwrite(os.path.join(self.save_path, 'images', str(self.save_img_id).zfill(6)+'_seg.png'), img_out[:,:,[2,1,0]])
                    np.save(os.path.join(self.save_path, 'images', str(self.save_img_id).zfill(6)+'_mask.npy'), mask_map[:,:,0])
                    # import pdb;pdb.set_trace()
                #* save data for vlmaps
                if 0:
                    if self.obj2cls is None:
                        self.obj2cls = {int(obj.id.split("_")[-1]): (obj.category.index(), obj.category.name()) for obj in self.sim.semantic_scene.objects}
                    save_obs(self.save_vlmap_path, self.sim_settings, obs, self.save_img_id, self.obj2cls)
                    self.vlmap_agent_states.append(agent_state)
                    save_states(self.save_vlmap_path, self.vlmap_agent_states)
            # self.save_path = os.path.join('../output/rendered', filename)
            # if not os.path.exists(self.save_path):
                # os.makedirs(self.save_path)
            if 1:
                if self.image_kuv1 is None:
                    width = int(self.sim_settings["width"])
                    height = int(self.sim_settings["height"])
                    hfov = math.radians(self.cfg.agents[0].sensor_specifications[0].hfov)
                    cam_intrinsics = np.eye(3)
                    cam_intrinsics[0,0] = width / (2.0 * math.tan(hfov / 2.0))
                    # cam_intrinsics[1,1] = cam_intrinsics[0,0] * (height / width)
                    cam_intrinsics[1,1] = cam_intrinsics[0,0]
                    cam_intrinsics[0,2] = width/2
                    cam_intrinsics[1,2] = height/2
                    self.cam_intrinsics = cam_intrinsics
                    xx, yy = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))  # pytorch's meshgrid has indexing='ij'
                    self.image_kuv1 = np.stack([xx, yy, np.ones_like(xx)],axis=2)
                    self.image_kuv1 = (np.linalg.inv(self.cam_intrinsics) @ self.image_kuv1.reshape(-1,3).T).T.reshape(height,width,3)
                depth_np = obs['depth_sensor']
                rgb = (obs['color_sensor'][:,:,:3])/255.0
                img_pts = (self.image_kuv1 * depth_np[:,:,None]).reshape(-1,3)
                img_pc = o3d.geometry.PointCloud()
                img_pc.points = o3d.utility.Vector3dVector(img_pts)
                img_pc.colors = o3d.utility.Vector3dVector(rgb.reshape(-1,3))
                img_pc = img_pc.voxel_down_sample(voxel_size=0.03)
                extrinsic = np.eye(4)
                extrinsic[:3,3] = pose[:3]
                extrinsic[:3,:3] = Rotation.from_quat(pose[3:]).as_matrix()
                rot_ro_cam = np.eye(3)
                rot_ro_cam[1, 1] = -1
                rot_ro_cam[2, 2] = -1
                extrinsic[:3,:3] = extrinsic[:3,:3] @ rot_ro_cam #* x 左 y 上 z前
                if 0: #* 显示相机视锥与路径
                    axis_pts = self.get_axis_pts()
                    # img_pc = o3d.geometry.PointCloud()
                    # img_pc.points = o3d.utility.Vector3dVector(axis_pts[:,:3])
                    # img_pc.colors = o3d.utility.Vector3dVector(axis_pts[:,3:])
                    # o3d.io.write_point_cloud(os.path.join(self.save_path, 'plys', 'test.ply'),img_pc)
                    axis_pts[:,:3] = (extrinsic[:3,:3] @ axis_pts[:,:3].T + extrinsic[:3,3:]).T
                    if self.last_ext is not None:
                        dir_ = extrinsic[:3,3]-self.last_ext[:3,3]
                        dist_ = np.linalg.norm(dir_)
                        if dist_>0.2:
                            start_pos = self.last_ext[:3,3]
                            end_pos = extrinsic[:3,3]
                            # self.pathfinder.distance_to_closest_obstacle(start_pos)
                            # 创建路径请求
                            path = habitat_sim.ShortestPath()
                            path.requested_start = start_pos
                            path.requested_end = end_pos
                            # 使用find_path方法寻找路径
                            found_path = self.pathfinder.find_path(path)
                            if found_path:
                                print("found_path")
                                ones = np.linspace(0, 1.0, 50)
                                pts_ = np.asarray(path.points)
                                pts_path = []
                                for pi_ in range(len(pts_)-1):
                                    pts_path.append(pts_[pi_,:3].reshape(1,-1)+(pts_[(pi_+1),:3].reshape(1,-1) - pts_[pi_,:3].reshape(1,-1))*ones.reshape(-1,1))
                                pts_ = np.vstack(pts_path)
                                #* 碰撞机制暂时不明确，暂不考虑
                                if 0:
                                    navmesh_vertices = np.array(self.pathfinder.build_navmesh_vertices())
                                    navmesh_indices = np.array(self.pathfinder.build_navmesh_vertex_indices())
                                    # navmesh_pc = o3d.geometry.PointCloud()
                                    # navmesh_pc.points = o3d.utility.Vector3dVector(navmesh_vertices)
                                    # navmesh_pc.paint_uniform_color([1,0,0])
                                    # o3d.io.write_point_cloud(os.path.join(self.save_path, 'plys', str(self.save_img_id).zfill(6)+'_navmesh.ply'),navmesh_pc)
                                    # # navmesh_pc.colors = o3d.utility.Vector3dVector(rgb.reshape(-1,3))
                                    # import pdb;pdb.set_trace()
                                    # 将三维顶点投影到二维平面，这里我们忽略了y轴（高度）
                                    vertices_2d = navmesh_vertices[:, [0, 2]]
                                    # 创建一个图形来绘制导航网格
                                    fig, ax = plt.subplots()
                                    for tri in navmesh_indices.reshape(-1, 3):
                                        # 绘制每个三角形
                                        triangle = np.vstack((vertices_2d[tri], vertices_2d[tri][0]))  # 闭合三角形
                                        ax.plot(triangle[:, 0], triangle[:, 1], 'k-')
                                        polygon = patches.Polygon(vertices_2d[tri], closed=True, color='blue', edgecolor='black')
                                        ax.add_patch(polygon)
                                    ax.set_aspect('equal')
                                    for pi in range(len(pts_)-1):
                                        p_ = pts_[pi]
                                        if not self.pathfinder.is_navigable(p_):
                                            print("Collision!!")
                                            ax.plot(pts_[pi:pi+2,0], pts_[pi:pi+2,2],'r-')
                                        else:
                                            print("Safe")
                                            ax.plot(pts_[pi:pi+2,0], pts_[pi:pi+2,2],'g-')
                                    print(pts_.shape)
                                    plt.show()
                            else:
                                print("not found_path")
                                len_ = int(dist_/0.01)
                                ones_ = np.linspace(0, 1, len_)
                                pts_ = (self.last_ext[:3,3:4] + dir_.reshape(-1,1)*ones_.reshape(1,-1)).T
                            clrs = np.zeros_like(pts_)
                            clrs[:,1] = 1.0
                            self.poses_pts.append(np.hstack([pts_,clrs]))
                            # agent_state.position = self.pathfinder.get_random_navigable_point()
                            # goal_pos = self.pathfinder.get_random_navigable_point()
                            # path = habitat_sim.ShortestPath()
                            # path.requested_start = agent_state.position
                            # path.requested_end = goal_pos
                            # follower = habitat_sim.GreedyGeodesicFollower(self.pathfinder, self.sim.get_agent(0), forward_key="move_forward", left_key="turn_left", right_key="turn_right",)
                            # action_list = follower.find_path(goal_pos)
                            # self.sim.get_agent(0).act(action_list[0])
                            # import pdb;pdb.set_trace()
                    self.poses_pts.append(axis_pts)
                    self.last_ext = extrinsic.copy()
                # o3d.visualization.draw_geometries([img_pc])
                # pcd_w = img_pc.transform(np.linalg.inv(extrinsic))
                # o3d.io.write_point_cloud(os.path.join(self.save_path, 'plys', str(self.save_img_id).zfill(6)+'.ply'),img_pc)
                pcd_w = img_pc.transform(extrinsic)
                # pcd_w = img_pc
                self.pc_all += pcd_w
                # o3d.visualization.draw_geometries([self.pc_all])
                o3d.io.write_point_cloud(os.path.join(self.save_path, 'pc_all.ply'), self.pc_all)
                # import pdb;pdb.set_trace()
            print(agent_state)
            self.save_img_id += 1
            # self.get_cur_top_down_view()
            self.enter_press = False
            # print('position: ',agent_state.position,', rotation: ',agent_state.rotation)
            # print(obs.keys())
            # if len(obs.keys())>0:
            #     for key in obs.keys():
            #         print(obs[key].shape)
            # import pdb;pdb.set_trace()
            # new_agent_state = habitat_sim.AgentState()
            # new_agent_state.position = (self.pathfinder.get_random_navigable_point())
            # new_agent_state.rotation = quat_from_angle_axis(self.sim.random.uniform_float(0, 2.0 * np.pi),np.array([0, 1, 0]),)
        # update map of moving/looking keys which are currently pressed
        print('get_agent_pose_euler: ', self.get_agent_pose_euler())
        if key in self.pressed:
            self.pressed[key] = True
        event.accepted = True
        self.redraw()

    def key_release_event(self, event: Application.KeyEvent) -> None:
        """
        Handles `Application.KeyEvent` on a key release. When a key is released, if it
        is part of the movement keys map `Dict[KeyEvent.key, Bool]`, then the key will
        be set to False for the next `self.move_and_look()` to update the cur_node actions.
        """
        key = event.key
        # update map of moving/looking keys which are currently pressed
        if key in self.pressed:
            self.pressed[key] = False
        event.accepted = True
        self.redraw()

    def mouse_move_event(self, event: Application.MouseMoveEvent) -> None:
        """
        Handles `Application.MouseMoveEvent`. When in LOOK mode, enables the left
        mouse button to steer the agent's facing direction. When in GRAB mode,
        continues to update the grabber's object position with our agents position.
        """
        button = Application.MouseMoveEvent.Buttons
        # if interactive mode -> LOOK MODE
        if event.buttons == button.LEFT and self.mouse_interaction == MouseMode.LOOK:
            agent = self.sim.agents[self.agent_id]
            delta = self.get_mouse_position(event.relative_position) / 2
            action = habitat_sim.agent.ObjectControls()
            act_spec = habitat_sim.agent.ActuationSpec

            # left/right on agent scene node
            action(agent.scene_node, "turn_right", act_spec(delta.x))

            # up/down on cameras' scene nodes
            action = habitat_sim.agent.ObjectControls()
            sensors = list(self.default_agent.scene_node.subtree_sensors.values())
            [action(s.object, "look_down", act_spec(delta.y), False) for s in sensors]

        # if interactive mode is TRUE -> GRAB MODE
        elif self.mouse_interaction == MouseMode.GRAB and self.mouse_grabber:
            # update location of grabbed object
            self.update_grab_position(self.get_mouse_position(event.position))

        self.previous_mouse_point = self.get_mouse_position(event.position)
        self.redraw()
        event.accepted = True

    def mouse_press_event(self, event: Application.MouseEvent) -> None:
        """
        Handles `Application.MouseEvent`. When in GRAB mode, click on
        objects to drag their position. (right-click for fixed constraints)
        """
        button = Application.MouseEvent.Button
        physics_enabled = self.sim.get_physics_simulation_library()

        # if interactive mode is True -> GRAB MODE
        if self.mouse_interaction == MouseMode.GRAB and physics_enabled:
            render_camera = self.render_camera.render_camera
            ray = render_camera.unproject(self.get_mouse_position(event.position))
            raycast_results = self.sim.cast_ray(ray=ray)

            if raycast_results.has_hits():
                hit_object, ao_link = -1, -1
                hit_info = raycast_results.hits[0]

                if hit_info.object_id >= 0:
                    # we hit an non-staged collision object
                    ro_mngr = self.sim.get_rigid_object_manager()
                    ao_mngr = self.sim.get_articulated_object_manager()
                    ao = ao_mngr.get_object_by_id(hit_info.object_id)
                    ro = ro_mngr.get_object_by_id(hit_info.object_id)

                    if ro:
                        # if grabbed an object
                        hit_object = hit_info.object_id
                        object_pivot = ro.transformation.inverted().transform_point(
                            hit_info.point
                        )
                        object_frame = ro.rotation.inverted()
                    elif ao:
                        # if grabbed the base link
                        hit_object = hit_info.object_id
                        object_pivot = ao.transformation.inverted().transform_point(
                            hit_info.point
                        )
                        object_frame = ao.rotation.inverted()
                    else:
                        for ao_handle in ao_mngr.get_objects_by_handle_substring():
                            ao = ao_mngr.get_object_by_handle(ao_handle)
                            link_to_obj_ids = ao.link_object_ids

                            if hit_info.object_id in link_to_obj_ids:
                                # if we got a link
                                ao_link = link_to_obj_ids[hit_info.object_id]
                                object_pivot = (
                                    ao.get_link_scene_node(ao_link)
                                    .transformation.inverted()
                                    .transform_point(hit_info.point)
                                )
                                object_frame = ao.get_link_scene_node(
                                    ao_link
                                ).rotation.inverted()
                                hit_object = ao.object_id
                                break
                    # done checking for AO

                    if hit_object >= 0:
                        node = self.default_agent.scene_node
                        constraint_settings = physics.RigidConstraintSettings()

                        constraint_settings.object_id_a = hit_object
                        constraint_settings.link_id_a = ao_link
                        constraint_settings.pivot_a = object_pivot
                        constraint_settings.frame_a = (
                            object_frame.to_matrix() @ node.rotation.to_matrix()
                        )
                        constraint_settings.frame_b = node.rotation.to_matrix()
                        constraint_settings.pivot_b = hit_info.point

                        # by default use a point 2 point constraint
                        if event.button == button.RIGHT:
                            constraint_settings.constraint_type = (
                                physics.RigidConstraintType.Fixed
                            )

                        grip_depth = (
                            hit_info.point - render_camera.node.absolute_translation
                        ).length()

                        self.mouse_grabber = MouseGrabber(
                            constraint_settings,
                            grip_depth,
                            self.sim,
                        )
                    else:
                        logger.warn("Oops, couldn't find the hit object. That's odd.")
                # end if didn't hit the scene
            # end has raycast hit
        # end has physics enabled

        self.previous_mouse_point = self.get_mouse_position(event.position)
        self.redraw()
        event.accepted = True

    def mouse_scroll_event(self, event: Application.MouseScrollEvent) -> None:
        """
        Handles `Application.MouseScrollEvent`. When in LOOK mode, enables camera
        zooming (fine-grained zoom using shift) When in GRAB mode, adjusts the depth
        of the grabber's object. (larger depth change rate using shift)
        """
        if 1:
            return
        print('mouse_scroll_event')
        scroll_mod_val = (
            event.offset.y
            if abs(event.offset.y) > abs(event.offset.x)
            else event.offset.x
        )
        if not scroll_mod_val:
            return

        # use shift to scale action response
        shift_pressed = bool(event.modifiers & Application.InputEvent.Modifier.SHIFT)
        alt_pressed = bool(event.modifiers & Application.InputEvent.Modifier.ALT)
        ctrl_pressed = bool(event.modifiers & Application.InputEvent.Modifier.CTRL)

        # if interactive mode is False -> LOOK MODE
        if self.mouse_interaction == MouseMode.LOOK:
            # use shift for fine-grained zooming
            mod_val = 1.01 if shift_pressed else 1.1
            mod = mod_val if scroll_mod_val > 0 else 1.0 / mod_val
            cam = self.render_camera
            cam.zoom(mod)
            self.redraw()

        elif self.mouse_interaction == MouseMode.GRAB and self.mouse_grabber:
            # adjust the depth
            mod_val = 0.1 if shift_pressed else 0.01
            scroll_delta = scroll_mod_val * mod_val
            if alt_pressed or ctrl_pressed:
                # rotate the object's local constraint frame
                agent_t = self.default_agent.scene_node.transformation_matrix()
                # ALT - yaw
                rotation_axis = agent_t.transform_vector(mn.Vector3(0, 1, 0))
                if alt_pressed and ctrl_pressed:
                    # ALT+CTRL - roll
                    rotation_axis = agent_t.transform_vector(mn.Vector3(0, 0, -1))
                elif ctrl_pressed:
                    # CTRL - pitch
                    rotation_axis = agent_t.transform_vector(mn.Vector3(1, 0, 0))
                self.mouse_grabber.rotate_local_frame_by_global_angle_axis(
                    rotation_axis, mn.Rad(scroll_delta)
                )
            else:
                # update location of grabbed object
                self.mouse_grabber.grip_depth += scroll_delta
                self.update_grab_position(self.get_mouse_position(event.position))
        self.redraw()
        event.accepted = True

    def mouse_release_event(self, event: Application.MouseEvent) -> None:
        """
        Release any existing constraints.
        """
        del self.mouse_grabber
        self.mouse_grabber = None
        event.accepted = True

    def update_grab_position(self, point: mn.Vector2i) -> None:
        """
        Accepts a point derived from a mouse click event and updates the
        transform of the mouse grabber.
        """
        # check mouse grabber
        if not self.mouse_grabber:
            return

        render_camera = self.render_camera.render_camera
        ray = render_camera.unproject(point)

        rotation: mn.Matrix3x3 = self.default_agent.scene_node.rotation.to_matrix()
        translation: mn.Vector3 = (
            render_camera.node.absolute_translation
            + ray.direction * self.mouse_grabber.grip_depth
        )
        self.mouse_grabber.update_transform(mn.Matrix4.from_(rotation, translation))

    def get_mouse_position(self, mouse_event_position: mn.Vector2i) -> mn.Vector2i:
        """
        This function will get a screen-space mouse position appropriately
        scaled based on framebuffer size and window size.  Generally these would be
        the same value, but on certain HiDPI displays (Retina displays) they may be
        different.
        """
        scaling = mn.Vector2i(self.framebuffer_size) / mn.Vector2i(self.window_size)
        return mouse_event_position * scaling

    def cycle_mouse_mode(self) -> None:
        """
        This method defines how to cycle through the mouse mode.
        """
        if self.mouse_interaction == MouseMode.LOOK:
            self.mouse_interaction = MouseMode.GRAB
        elif self.mouse_interaction == MouseMode.GRAB:
            self.mouse_interaction = MouseMode.LOOK

    def navmesh_config_and_recompute(self) -> None:
        """
        This method is setup to be overridden in for setting config accessibility
        in inherited classes.
        """
        self.navmesh_settings = habitat_sim.NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_height = self.cfg.agents[self.agent_id].height
        self.navmesh_settings.agent_radius = self.cfg.agents[self.agent_id].radius
        self.navmesh_settings.include_static_objects = True
        self.sim.recompute_navmesh(
            self.pathfinder,
            self.navmesh_settings,
        )

    def exit_event(self, event: Application.ExitEvent):
        """
        Overrides exit_event to properly close the Simulator before exiting the
        application.
        """
        for i in range(self.num_env):
            self.tiled_sims[i].close(destroy=True)
            event.accepted = True
        exit(0)

    def draw_text(self, sensor_spec):
        self.shader.bind_vector_texture(self.glyph_cache.texture)
        self.shader.transformation_projection_matrix = self.window_text_transform
        self.shader.color = [1.0, 1.0, 1.0]

        sensor_type_string = str(sensor_spec.sensor_type.name)
        sensor_subtype_string = str(sensor_spec.sensor_subtype.name)
        if self.mouse_interaction == MouseMode.LOOK:
            mouse_mode_string = "LOOK"
        elif self.mouse_interaction == MouseMode.GRAB:
            mouse_mode_string = "GRAB"
        self.window_text.render(
            f"""
{self.fps} FPS
Sensor Type: {sensor_type_string}
Sensor Subtype: {sensor_subtype_string}
Mouse Interaction Mode: {mouse_mode_string}
            """
        )
        self.shader.draw(self.window_text.mesh)

    def print_help_text(self) -> None:
        """
        Print the Key Command help text.
        """
        logger.info(
            """
=====================================================
Welcome to the Habitat-sim Python Viewer application!
=====================================================
Mouse Functions ('m' to toggle mode):
----------------
In LOOK mode (default):
    LEFT:
        Click and drag to rotate the agent and look up/down.
    WHEEL:
        Modify orthographic camera zoom/perspective camera FOV (+SHIFT for fine grained control)

In GRAB mode (with 'enable-physics'):
    LEFT:
        Click and drag to pickup and move an object with a point-to-point constraint (e.g. ball joint).
    RIGHT:
        Click and drag to pickup and move an object with a fixed frame constraint.
    WHEEL (with picked object):
        default - Pull gripped object closer or push it away.
        (+ALT) rotate object fixed constraint frame (yaw)
        (+CTRL) rotate object fixed constraint frame (pitch)
        (+ALT+CTRL) rotate object fixed constraint frame (roll)
        (+SHIFT) amplify scroll magnitude


Key Commands:
-------------
    esc:        Exit the application.
    'h':        Display this help message.
    'm':        Cycle mouse interaction modes.

    Agent Controls:
    'wasd':     Move the agent's body forward/backward and left/right.
    'zx':       Move the agent's body up/down.
    arrow keys: Turn the agent's body left/right and camera look up/down.

    Utilities:
    'r':        Reset the simulator with the most recently loaded scene.
    'n':        Show/hide NavMesh wireframe.
                (+SHIFT) Recompute NavMesh with default settings.
                (+ALT) Re-sample the agent(camera)'s position and orientation from the NavMesh.
    ',':        Render a Bullet collision shape debug wireframe overlay (white=active, green=sleeping, blue=wants sleeping, red=can't sleep).
    'c':        Run a discrete collision detection pass and render a debug wireframe overlay showing active contact points and normals (yellow=fixed length normals, red=collision distances).
                (+SHIFT) Toggle the contact point debug render overlay on/off.

    Object Interactions:
    SPACE:      Toggle physics simulation on/off.
    '.':        Take a single simulation step if not simulating continuously.
    'v':        (physics) Invert gravity.
    't':        Load URDF from filepath
                (+SHIFT) quick re-load the previously specified URDF
                (+ALT) load the URDF with fixed base
=====================================================
"""
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # optional arguments
    parser.add_argument("--scene",type=str,help='scene/stage file to load (default: "./data/test_assets/scenes/simple_room.glb")',    )
    parser.add_argument("--dataset",default="data/versioned_data/mp3d_example_scene_1.1/mp3d.scene_dataset_config.json",type=str,metavar="DATASET",help='dataset configuration file to use (default: "./data/objects/ycb/ycb.scene_dataset_config.json")',)
    parser.add_argument("--disable-physics",action="store_true",help="disable physics simulation (default: False)",)
    parser.add_argument("--use-default-lighting",action="store_true",help="Override configured lighting to use default lighting for the stage.",)
    parser.add_argument("--hbao",action="store_true",help="Enable horizon-based ambient occlusion, which provides soft shadows in corners and crevices.",    )
    parser.add_argument("--enable-batch-renderer",action="store_true",help="Enable batch rendering mode. The number of concurrent environments is specified with the num-environments parameter.",)
    parser.add_argument("--num-environments",default=1,type=int,help="Number of concurrent environments to batch render. Note that only the first environment simulates physics and can be controlled.",)
    parser.add_argument("--composite-files",type=str,nargs="*",help="Composite files that the batch renderer will use in-place of simulation assets to improve memory usage and performance. If none is specified, the original scene files will be loaded from disk.",)
    parser.add_argument("--width",default=1080,type=int,help="Horizontal resolution of the window.",)
    parser.add_argument("--height",default=720,type=int,help="Vertical resolution of the window.",)
    args = parser.parse_args()
    if args.num_environments < 1:
        parser.error("num-environments must be a positive non-zero integer.")
    if args.width < 1:
        parser.error("width must be a positive non-zero integer.")
    if args.height < 1:
        parser.error("height must be a positive non-zero integer.")
    # Setting up sim_settings
    sim_settings: Dict[str, Any] = default_sim_settings
    sim_settings["scene"] = args.scene
    sim_settings["scene_dataset_config_file"] = args.dataset
    # sim_settings["enable_physics"] = not args.disable_physics
    sim_settings["enable_physics"] = False
    sim_settings["use_default_lighting"] = args.use_default_lighting
    sim_settings["enable_batch_renderer"] = args.enable_batch_renderer
    sim_settings["num_environments"] = args.num_environments
    sim_settings["composite_files"] = args.composite_files
    sim_settings["window_width"] = args.width
    sim_settings["window_height"] = args.height
    sim_settings["default_agent_navmesh"] = False
    sim_settings["enable_hbao"] = args.hbao
    sim_settings["color_sensor"] = True
    sim_settings["depth_sensor"] = True
    sim_settings["semantic_sensor"] = True
    sim_settings["move_forward"] = 0.1
    sim_settings["turn_left"] = 5
    sim_settings["turn_right"] = 5
    sim_settings["sensor_height"] = 1.5
    sim_settings["use_som"] = True
    sim_settings["img_save_dir"] = '/media/zhujun/0DFD06D20DFD06D2/SLAM/vlmaps/output/test'

    # start the application
    HabitatSimInteractiveViewer(sim_settings).exec()

