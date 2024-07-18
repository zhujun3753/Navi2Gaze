// Copyright (c) 2023 Jun Zhu, Tsinghua University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <tuple>

#include <torch/custom_class.h>
#include <torch/script.h>
#include "Octree.hpp"
#include <Eigen/Eigen>
#include <cstdlib>
#include <time.h>


#ifndef OCTREEMAP_H_INCLUDED
#define OCTREEMAP_H_INCLUDED

#define HASH_P 116101
#define MAX_N 10000000000
class VOXEL_LOC
{
public:
	int64_t x, y, z;

	VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
		: x(vx), y(vy), z(vz) {}

	bool operator==(const VOXEL_LOC &other) const
	{
		return (x == other.x && y == other.y && z == other.z);
	}

    bool operator<(const VOXEL_LOC &other) const
	{
		return (x < other.x ||  x == other.x && y < other.y || x == other.x && y == other.y && z < other.z);
	}

    friend std::ostream& operator << (std::ostream& os, const VOXEL_LOC& p)
    {
        os  << "("<< p.x <<", "<<p.y<<", "<<p.z<<")"<<std::endl;
        return os;
    }
};

// Hash value
namespace std
{
	template <>
	struct hash<VOXEL_LOC>
	{
		int64_t operator()(const VOXEL_LOC &s) const
		{
			using std::hash;
			using std::size_t;
			return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
		}
	};
} // namespace std

template <typename data_type = float, typename T = void *>
struct Hash_map_3d
{
    //* 映射嵌套,niubi
    using hash_3d_T = std::unordered_map<data_type, std::unordered_map<data_type, std::unordered_map<data_type, T>>>;
    hash_3d_T m_map_3d_hash_map;
    void insert(const data_type &x, const data_type &y, const data_type &z, const T &target)
    {
        m_map_3d_hash_map[x][y][z] = target;
    }

    void erase(const data_type &x, const data_type &y, const data_type &z)
    {
        if(m_map_3d_hash_map.find(x) == m_map_3d_hash_map.end()  )
            return;
        else if(m_map_3d_hash_map[x].find(y) ==  m_map_3d_hash_map[x].end() )
            return;
        else if( m_map_3d_hash_map[x][y].find(z) == m_map_3d_hash_map[x][y].end() )
            return;
        m_map_3d_hash_map[x][y].erase(z);
    }
    
    int if_exist(const data_type &x, const data_type &y, const data_type &z)
    {
        if(m_map_3d_hash_map.find(x) == m_map_3d_hash_map.end()  )
            return 0;
        else if(m_map_3d_hash_map[x].find(y) ==  m_map_3d_hash_map[x].end() )
            return 0;
        else if( m_map_3d_hash_map[x][y].find(z) == m_map_3d_hash_map[x][y].end() )
            return 0;
        return 1;
    }

    int get(const data_type &x, const data_type &y, const data_type &z, T * target)
    {
        if(if_exist(x,y,z))
        {
            *target = m_map_3d_hash_map[x][y][z];
            return 1;
        }
        return 0;
    }

    void clear()
    {
        m_map_3d_hash_map.clear();
    }

    int total_size()
    {
        int count =0 ;
        for(auto it : m_map_3d_hash_map)
            for(auto it_it: it.second)
                for( auto it_it_it: it_it.second )
                    count++;
        return count;
    }

	int size()
    {
        return total_size();
    }
};

class RGB_pts
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    float m_pos[ 3 ] = { 0 }; //* 位置
    float m_rgb[ 3 ] = { 0 }; //* 颜色
	int attr = -1; // -1未知 0 障碍点 1 无障碍点
	int id = -1; // 

    RGB_pts() { clear(); };

    ~RGB_pts(){};

	void clear()
    {
        m_rgb[ 0 ] = 0;
        m_rgb[ 1 ] = 0;
        m_rgb[ 2 ] = 0;
    };

    void set_pos( const Eigen::Vector3f &pos ) 
    {
        m_pos[0] = pos(0); 
        m_pos[1] = pos(1); 
        m_pos[2] = pos(2); 
    };

	void set_rgb( const Eigen::Vector3f &pos ) 
    {
        m_rgb[0] = pos(0); 
        m_rgb[1] = pos(1); 
        m_rgb[2] = pos(2); 
    };

    Eigen::Vector3f get_pos(){ return Eigen::Vector3f(m_pos[0], m_pos[1], m_pos[2]); };

    Eigen::Vector3f get_rgb(){ return Eigen::Vector3f(m_rgb[0], m_rgb[1], m_rgb[2]); };


};
using RGB_pt_ptr = std::shared_ptr< RGB_pts >;

struct OctreeMap : torch::CustomClassHolder
{
public:
	thuni::Octree octree_feature;
	thuni::Octree octree_ground;
	thuni::Octree octree_obstacle;
	thuni::Octree octree_object;
	float voxel_size = 0.1;
	std::vector< RGB_pt_ptr > m_ground_pts_vec;
	std::vector< RGB_pt_ptr > m_obstacle_pts_vec;
	Hash_map_3d< long, RGB_pt_ptr > m_hashmap_3d_pts;
	Eigen::Vector3f m_center=Eigen::Vector3f::Zero(), m_start_pos=Eigen::Vector3f::Zero();
	double m_radius=1.0, m_grid_size=0.01;
	int64_t m_ignore_axis=-1, m_width=0, m_max_id=-1, m_dim=4;
	std::vector<int64_t> m_remaind_axis;
	std::vector<std::vector<float>> m_data;
	std::vector<Eigen::Vector3f> circle_centers;
	float circle_radius;

	OctreeMap()
	{
		octree_feature.set_min_extent(voxel_size/2);
		octree_feature.set_bucket_size(1);
		srand(0);
		// octree_feature.set_down_size();
	}

	~OctreeMap()
	{
		octree_feature.clear();
	}

	void clear()
	{
		octree_feature.clear();
		octree_ground.clear();
		octree_obstacle.clear();
		octree_object.clear();
		voxel_size = 0.1;
		m_ground_pts_vec.clear();
		m_obstacle_pts_vec.clear();
		m_hashmap_3d_pts.clear();
		m_center=Eigen::Vector3f::Zero(), m_start_pos=Eigen::Vector3f::Zero();
		m_radius=1.0, m_grid_size=0.01;
		m_ignore_axis=-1, m_width=0, m_max_id=-1, m_dim=4;
		m_remaind_axis.clear();
		m_data.clear();
	}

	int64_t pos2id(const Eigen::Vector3f & pt )
	{
		if(m_ignore_axis<0)
		{
			std::cout<<"Need Init!"<<std::endl;
			return -1;
		}
		Eigen::Vector3f rel_pos = pt - m_start_pos;
		int64_t axis1 = std::round(rel_pos[m_remaind_axis[0]]/m_grid_size);
		int64_t axis2 = std::round(rel_pos[m_remaind_axis[1]]/m_grid_size);
		if(axis1<0 || axis1>=m_width || axis2<0 || axis2>=m_width) return -1;
		return axis1*m_width+axis2;
	}

	Eigen::Vector3f id2pos(const  int64_t & id)
	{
		if(m_ignore_axis<0)
		{
			std::cout<<"Need Init!"<<std::endl;
			return Eigen::Vector3f::Zero();
		}
		int64_t axis2 = id%m_width;
		int64_t axis1 = (id-axis2)/m_width;
		Eigen::Vector3f pt = m_start_pos;
		pt[m_remaind_axis[0]] += axis1*m_grid_size;
		pt[m_remaind_axis[1]] += axis2*m_grid_size;
		return pt;
	}

	void ground_map_setup(torch::Tensor center, double radius=1.0, double grid_size=0.01, int64_t ignore_axis = 1)
	{
		if(center.dim()!=1)
		{
			std::cout<<"The dim of pts_attr should be 1!!\n";
			return;
		}
		if(center.size(0)<3)
		{
			std::cout<<"center Must be a 3D pts!\n";
			return;
		}
		auto center_access = center.accessor<float,1>();
		m_center = Eigen::Vector3f(center_access[0], center_access[1], center_access[2]);
		m_start_pos = m_center - Eigen::Vector3f(radius,radius,radius);
		m_start_pos[ignore_axis] = m_center[ignore_axis];
		m_radius = radius;
		m_grid_size = grid_size;
		m_ignore_axis = ignore_axis;
		for(int64_t i=0; i<3; i++)
		{
			if(i!=ignore_axis) m_remaind_axis.push_back(i);
		}
		m_width = int64_t(ceil(radius*2/grid_size))+1;
		m_max_id = m_width*m_width;
		m_data.resize(m_max_id);
		std::cout<<"m_width: "<<m_width<<", m_max_id: "<<m_max_id<<std::endl;
	}

	void label_occupied_grid(torch::Tensor center, double box_len, double label)
	{
		if(center.dim()!=1)
		{
			std::cout<<"The dim of pts_attr should be 1!!\n";
			return;
		}
		if(center.size(0)<3)
		{
			std::cout<<"center Must be a 3D pts!\n";
			return;
		}
		auto center_access = center.accessor<float,1>();
		Eigen::Vector3f center_eigen = Eigen::Vector3f(center_access[0], center_access[1], center_access[2]);
		Eigen::Vector3f start_pos = center_eigen - Eigen::Vector3f(box_len,box_len,box_len)/2;
		Eigen::Vector3f end_pos = center_eigen + Eigen::Vector3f(box_len,box_len,box_len)/2;
		int64_t start_id = pos2id(start_pos);
		int64_t end_id = pos2id(end_pos);
		if(start_id<0 || end_id<0) return;
		int64_t start_u = start_id%m_width, start_v=(start_id-start_u)/m_width;
		int64_t end_u = end_id%m_width, end_v=(end_id-end_u)/m_width;
		float label_f = label;
		for(int64_t i=start_v; i<=end_v; i++)
		{
			for(int64_t j=start_u; j<= end_u; j++)
			{
				int64_t id_ = i*m_width+j;
				if(m_data[id_].size()==0) continue;
				m_data[id_][3] =label_f;
			}
		}
	}

	bool check_bounding_box_valid(torch::Tensor center, double box_len)
	{
		if(center.dim()!=1)
		{
			std::cout<<"The dim of pts_attr should be 1!!\n";
			return false;
		}
		if(center.size(0)<3)
		{
			std::cout<<"center Must be a 3D pts!\n";
			return false;
		}
		auto center_access = center.accessor<float,1>();
		Eigen::Vector3f center_eigen = Eigen::Vector3f(center_access[0], center_access[1], center_access[2]);
		Eigen::Vector3f start_pos = center_eigen - Eigen::Vector3f(box_len,box_len,box_len)/2;
		Eigen::Vector3f end_pos = center_eigen + Eigen::Vector3f(box_len,box_len,box_len)/2;
		int64_t start_id = pos2id(start_pos);
		int64_t end_id = pos2id(end_pos);
		if(start_id<0 || end_id<0) return false;
		int64_t start_u = start_id%m_width, start_v=(start_id-start_u)/m_width;
		int64_t end_u = end_id%m_width, end_v=(end_id-end_u)/m_width;
		bool valid = true;
		int64_t empty_grid_n = 0;
		int64_t empty_grid_n_max = 8;
		for(int64_t i=start_v; i<=end_v; i++)
		{
			int64_t j=start_u;
			int64_t id_ = i*m_width+j;
			if(m_data[id_].size()==0)
			{
				empty_grid_n++;
				if(empty_grid_n>empty_grid_n_max)
				{
					valid = false;
					break;
				}
			}
			else if(m_data[id_][3] == 0.0)
			{
				valid = false;
				break;
			}
			j=end_u;
			id_ = i*m_width+j;
			if(m_data[id_].size()==0)
			{
				empty_grid_n++;
				if(empty_grid_n>empty_grid_n_max)
				{
					valid = false;
					break;
				}
			}
			else if(m_data[id_][3] == 0.0)
			{
				valid = false;
				break;
			}
		}
		for(int64_t j=start_u; j<=end_u && valid; j++)
		{
			int64_t i=start_v;
			int64_t id_ = i*m_width+j;
			if(m_data[id_].size()==0)
			{
				empty_grid_n++;
				if(empty_grid_n>empty_grid_n_max)
				{
					valid = false;
					break;
				}
				break;
			}
			else if(m_data[id_][3] == 0.0)
			{
				valid = false;
				break;
			}
			i=end_v;
			id_ = i*m_width+j;
			if(m_data[id_].size()==0)
			{
				empty_grid_n++;
				if(empty_grid_n>empty_grid_n_max)
				{
					valid = false;
					break;
				}
				break;
			}
			else if(m_data[id_][3] == 0.0)
			{
				valid = false;
				break;
			}
		}
		return valid;
	}

	// 沿着某一个方向检索障碍物
	double check_obstacle_along_dir(torch::Tensor center, torch::Tensor dir, bool only_obstacle = false)
	{
		if(dir.dim()!=1 || center.dim()!=1)
		{
			std::cout<<"The dim should be 1!!\n";
			return -1;
		}
		if(dir.size(0)<3 || center.size(0)<3)
		{
			std::cout<<"Must be a 3D pt!\n";
			return -1;
		}
		auto dir_access = dir.accessor<float,1>();
		auto center_access = center.accessor<float,1>();
		Eigen::Vector3f dir_eigen = Eigen::Vector3f(dir_access[0], dir_access[1], dir_access[2]);
		Eigen::Vector3f center_eigen = Eigen::Vector3f(center_access[0], center_access[1], center_access[2]);
		dir_eigen /= dir_eigen.norm();
		float scale_ = 0.01;
		Eigen::Vector3f scaled_dir = scale_*dir_eigen;
		float start_step = 1.0;
		float dist2obstacle_along_dir = 0.0;
		while(1)
		{
			dist2obstacle_along_dir = scale_*start_step;
			Eigen::Vector3f pt = center_eigen+dir_eigen*dist2obstacle_along_dir;
			int64_t id_ = pos2id(pt);
			start_step += 1;
			if(id_<0) break;
			// std::cout<<"id_: "<<id_<<std::endl;
			if(m_data[id_].size()==0 && !only_obstacle) break;
			if(m_data[id_].size()==0 && only_obstacle) continue;
			if(m_data[id_][3] == 0.0) break;
			if(m_data[id_][3] == 2.0 && !only_obstacle) break;
		}
		return double(dist2obstacle_along_dir);
	}

	void add_pts_to_ground_map(torch::Tensor pts_attr, double attr=1.0)
	{
		if(pts_attr.dim()!=2)
		{
			std::cout<<"The dim of pts_attr should be 2!!\n";
			return;
		}
		if(pts_attr.size(1)<3)
		{
			std::cout<<"Must be 3D pts!\n";
			return;
		}
		int num = pts_attr.size(0);
		int attr_n = pts_attr.size(1);
		// std::cout<<"num: "<<num<<", attr_n: "<<attr_n<<std::endl;;
		auto pts_attr_acces = pts_attr.accessor<float,2>();
		std::vector<Eigen::Vector3f> obstacle_pts, ground_pts, remove_ground_pts, object_pts;
		bool obstacle = attr!=1.0;
		// #* 空 未知点 蓝色
        // #* 0 一般障碍点 黑色
        // #* 1 可行地面点 绿色
        // #* 2 目标点（同时也是障碍点） 红色
		for (int i=0; i<num; i++)
		{
			Eigen::Vector3f pt = Eigen::Vector3f(pts_attr_acces[i][0], pts_attr_acces[i][1], pts_attr_acces[i][2]);
			int64_t id_ = pos2id(pt);
			if(id_<0) continue;
			// std::cout<<"id_: "<<id_<<std::endl;;
			if(m_data[id_].size()==0)
			{
				if(!obstacle)
				{
					bool have_obstacle = false; // 周围8个点不存在障碍点
					int64_t sur_ids[8] = {id_-1, id_+1, id_-m_width-1, id_-m_width,  id_-m_width+1,  id_+m_width-1, id_+m_width, id_+m_width+1};
					for(int j=0; j<8; j++)
					{
						if(sur_ids[j]<0 || sur_ids[j]>=m_max_id) continue;
						if(m_data[sur_ids[j]].size()==0) continue;
						if(m_data[sur_ids[j]][3]==0)
						{
							have_obstacle = true;
							break;
						}
					}
					if(have_obstacle) continue;
				}
				m_data[id_] = {pt[0], pt[1], pt[2], attr};
				if(obstacle) obstacle_pts.push_back(pt);
				else ground_pts.push_back(pt);
				if(attr==2.0) object_pts.push_back(pt);
			}
			else
			{
				// std::cout<<"obstacle\n";
				if(obstacle)
				{
					m_data[id_][3] = attr;
					obstacle_pts.push_back(pt);
					remove_ground_pts.push_back(Eigen::Vector3f(m_data[id_][0], m_data[id_][1], m_data[id_][2]));
					if(attr==2.0) object_pts.push_back(pt);
				}
			}
		}
		// std::cout<<"remove_pts_vec\n";
		std::vector<float> tmp;
		if(obstacle)
		{
			std::cout<<"obstacle_pts.size(): "<<obstacle_pts.size()<<std::endl;
			octree_obstacle.update_with_attr(obstacle_pts, tmp);
			if(attr==2.0) octree_object.update_with_attr(object_pts, tmp);
		}
		else
		{
			std::cout<<"ground_pts.size(): "<<ground_pts.size()<<std::endl;
			octree_ground.update_with_attr(ground_pts, tmp);
			// std::cout<<"remove_pts_vec\n";
			octree_ground.remove_pts_vec(remove_ground_pts, 1e-4);
			std::cout<<"octree_ground.size(): "<<octree_ground.size()<<std::endl;
		}
		// std::cout<<"m_hashmap_3d_pts.size(): "<<m_hashmap_3d_pts.size()<<std::endl;
	}

	std::tuple<torch::Tensor, torch::Tensor> update_object_envelope_circles()
	{
		if(octree_object.size()<1)
		{
			std::cout<<"Empty octree!\n";
			return std::tuple<torch::Tensor, torch::Tensor>();
		}
		// std::cout<<"step!\n";
		// 对于已经满足要求的圆圈，保持不变，仅仅改变未知区域的圆形
		float radius = circle_radius;
		// std::cout<<"candidate_pts!\n";
		// 初步筛选需要更改的圆
		std::vector<Eigen::Vector3f> candidate_pts = circle_centers;
		std::vector<Eigen::Vector3f> candidate_pts_inti;
		std::vector<float> candidate_pt_dist;
		std::vector<Eigen::Vector3f> candidate_pt_near_pts;
		thuni::Octree octree_circle_centers;
		std::vector<Eigen::Vector3f> candidate_pts_move;
		for(int i=0; i<candidate_pts.size(); i++)
		{
			std::vector<float> query;
			for(int j=0; j<3; j++)
				query.push_back(candidate_pts[i][j]);
			{
				std::vector<std::vector<float>> resultIndices;
				std::vector<float> distances;
				octree_obstacle.knnNeighbors_eigen(query, 1, resultIndices, distances);
				if(distances.size()>0)
				{
					float dist = sqrt(distances[0]);
					if(dist>radius) // 已经满足要求的圆形就不再改变了
					{
						std::vector<float> tmp;
						std::vector<Eigen::Vector3f> candidate_pts_move_;
						candidate_pts_move_.push_back(candidate_pts[i]);
						octree_circle_centers.update_with_attr(candidate_pts_move_, tmp);
						candidate_pts_move.push_back(candidate_pts[i]);
						continue;
					}
				}
			}
			// 计算圆形中心到目标对象点的距离和最近点
			{
				std::vector<std::vector<float>> resultIndices;
				std::vector<float> distances;
				octree_object.knnNeighbors_eigen(query, 1, resultIndices, distances);
				if(distances.size()>0)
				{
					float dist = sqrt(distances[0]);
					candidate_pts_inti.push_back(candidate_pts[i]);
					candidate_pt_dist.push_back(dist);
					candidate_pt_near_pts.push_back(Eigen::Vector3f(resultIndices[0][0],resultIndices[0][1],resultIndices[0][2]));
				}
			}
		}
		std::vector<float> tmp;
		// 圆形筛选
		// std::cout<<"candidate_pts_move!\n";
		for(int i=0; i<candidate_pts_inti.size(); i++)
		{
			Eigen::Vector3f pt = candidate_pts_inti[i];
			Eigen::Vector3f move_dir = pt - candidate_pt_near_pts[i];
			move_dir /= move_dir.norm();
			float move_dist = m_grid_size+radius;
			pt = candidate_pt_near_pts[i] + move_dir*move_dist;
			{
				for(int j=0; j<10; j++)
				{
					pt = candidate_pt_near_pts[i] + move_dir*move_dist;
					std::vector<float> query;
					for(int j=0; j<3; j++)
						query.push_back(pt[j]);
					std::vector<std::vector<float>> resultIndices;
					std::vector<float> distances;
					octree_obstacle.knnNeighbors_eigen(query, 1, resultIndices, distances);
					if(distances.size()>0)
					{
						float dist = sqrt(distances[0]);
						if(dist<radius) // 起码距离目标点较远
						{
							move_dist += radius/2;
							if(move_dist>4*radius) break;
						}
						else
							break;
					}
				}
				if(move_dist>4*radius) continue;
			}
			{
				std::vector<float> query;
				for(int j=0; j<3; j++)
					query.push_back(pt[j]);
				std::vector<std::vector<float>> resultIndices;
				std::vector<float> distances;
				octree_object.knnNeighbors_eigen(query, 1, resultIndices, distances);
				if(distances.size()>0)
				{
					float dist = sqrt(distances[0]);
					if(dist>radius) // 起码距离目标对象点较远
					{
						if(octree_circle_centers.size()>0)
						{
							std::vector<std::vector<float>> resultIndices1;
							std::vector<float> distances1;
							octree_circle_centers.knnNeighbors_eigen(query, 1, resultIndices1, distances1);
							if(distances1.size()>0)
							{
								float dist1 = sqrt(distances1[0]);
								if(dist1>2*radius) // 不在已有的圆形范围内
								{
									candidate_pts_move.push_back(pt);
									std::vector<Eigen::Vector3f> candidate_pts_move_;
									candidate_pts_move_.push_back(pt);
									octree_circle_centers.update_with_attr(candidate_pts_move_, tmp);
								}
							}
						}
						else
						{
							candidate_pts_move.push_back(pt);
							std::vector<Eigen::Vector3f> candidate_pts_move_;
							candidate_pts_move_.push_back(pt);
							octree_circle_centers.update_with_attr(candidate_pts_move_, tmp);
						}
					}
				}
			}
		}
		// std::cout<<"base_circle_pts!\n";
		// 基础圆形
		std::vector<Eigen::Vector3f> base_circle_pts;
		int num_base_pt = 360;
		float num_base_pt_f = 360;
		for(int i=0; i<num_base_pt; i++)
		{
			float angle = i/num_base_pt_f*2*3.14159256;
			Eigen::Vector3f pt = Eigen::Vector3f::Zero();
			pt[m_remaind_axis[0]] = radius*cos(angle);
			pt[m_remaind_axis[1]] = radius*sin(angle);
			base_circle_pts.push_back(pt);
		}
		// std::cout<<"data_tensor!\n";
		torch::Tensor data_tensor = torch::zeros({candidate_pts_move.size(), num_base_pt, 6}, torch::dtype(torch::kFloat32));
		torch::Tensor circle_tensor = torch::zeros({candidate_pts_move.size(), 3}, torch::dtype(torch::kFloat32));
		auto data_tensor_acces = data_tensor.accessor<float,3>();
		auto circle_tensor_acces = circle_tensor.accessor<float,2>();
		for(int i=0; i<candidate_pts_move.size(); i++)
		{
			std::vector<float> colors;
			colors.push_back(float(std::rand()%255)/255.0);
			colors.push_back(float(std::rand()%255)/255.0);
			colors.push_back(float(std::rand()%255)/255.0);
			for(int j=0; j<3; j++)
				circle_tensor_acces[i][j] = candidate_pts_move[i][j];
			for(int j=0; j<num_base_pt; j++)
			{
				Eigen::Vector3f pt = candidate_pts_move[i] + base_circle_pts[j];
				for(int k=0; k<3; k++)
				{
					data_tensor_acces[i][j][k] = pt[k];
					data_tensor_acces[i][j][3+k] = colors[k];
				}
			}
		}
		circle_centers = candidate_pts_move;
		return std::make_tuple(data_tensor, circle_tensor);;
	}

	std::tuple<torch::Tensor, torch::Tensor> get_object_envelope_circles( const double radius)
	{
		if(octree_object.size()<1)
		{
			std::cout<<"Empty octree!\n";
			return std::tuple<torch::Tensor, torch::Tensor>();
		}
		// std::cout<<"step!\n";
		// 优先通过网格内的稀疏点大致确定可行的点，再画圆
		circle_radius = radius;
		int step = radius/m_grid_size/3;
		std::vector<Eigen::Vector3f> candidate_pts;
		for(int idx=0; idx<m_width; idx+=step)
		{
			for(int idy=0; idy<m_width; idy+=step)
			{
				int64_t id_ = idy*m_width+idx;
				Eigen::Vector3f pt = id2pos(id_);
				candidate_pts.push_back(pt);
			}
		}
		// std::cout<<"candidate_pts!\n";
		// 初步筛选
		std::vector<Eigen::Vector3f> candidate_pts_inti;
		std::vector<float> candidate_pt_dist;
		std::vector<Eigen::Vector3f> candidate_pt_near_pts;
		thuni::Octree octree_circle_centers;
		for(int i=0; i<candidate_pts.size(); i++)
		{
			std::vector<float> query;
			for(int j=0; j<3; j++)
				query.push_back(candidate_pts[i][j]);
			std::vector<std::vector<float>> resultIndices;
			std::vector<float> distances;
			octree_object.knnNeighbors_eigen(query, 1, resultIndices, distances);
			if(distances.size()>0)
			{
				float dist = sqrt(distances[0]);
				if(dist>0.5*radius && dist<1.5*radius)
				{
					candidate_pts_inti.push_back(candidate_pts[i]);
					candidate_pt_dist.push_back(dist);
					candidate_pt_near_pts.push_back(Eigen::Vector3f(resultIndices[0][0],resultIndices[0][1],resultIndices[0][2]));
				}
			}
		}
		std::vector<float> tmp;
		// 圆形筛选
		// std::cout<<"candidate_pts_move!\n";
		std::vector<Eigen::Vector3f> candidate_pts_move;
		for(int i=0; i<candidate_pts_inti.size(); i++)
		{
			Eigen::Vector3f pt = candidate_pts_inti[i];
			Eigen::Vector3f move_dir = pt - candidate_pt_near_pts[i];
			move_dir /= move_dir.norm();
			float move_dist = m_grid_size+radius;
			pt = candidate_pt_near_pts[i] + move_dir*move_dist;
			// 圆平移
			{
				for(int j=0; j<10; j++)
				{
					pt = candidate_pt_near_pts[i] + move_dir*move_dist;
					std::vector<float> query;
					for(int j=0; j<3; j++)
						query.push_back(pt[j]);
					std::vector<std::vector<float>> resultIndices;
					std::vector<float> distances;
					octree_obstacle.knnNeighbors_eigen(query, 1, resultIndices, distances);
					if(distances.size()>0)
					{
						float dist = sqrt(distances[0]);
						if(dist<radius) // 起码距离目标点较远
						{
							move_dist += radius/2;
							if(move_dist>4*radius) break;
						}
						else
							break;
					}
				}
				if(move_dist>4*radius) continue;
			}
			// candidate_pts_move.push_back(pt);
			// 筛选
			// if(0)
			{
				std::vector<float> query;
				for(int j=0; j<3; j++)
					query.push_back(pt[j]);
				std::vector<std::vector<float>> resultIndices;
				std::vector<float> distances;
				octree_object.knnNeighbors_eigen(query, 1, resultIndices, distances);
				if(distances.size()>0)
				{
					float dist = sqrt(distances[0]);
					if(dist>radius) // 筛选1：起码距离目标点较远
					{
						if(octree_circle_centers.size()>0)
						{
							std::vector<std::vector<float>> resultIndices1;
							std::vector<float> distances1;
							octree_circle_centers.knnNeighbors_eigen(query, 1, resultIndices1, distances1);
							if(distances1.size()>0)
							{
								float dist1 = sqrt(distances1[0]);
								if(dist1>2*radius) // 筛选2：不在已有的圆形范围内
								{
									candidate_pts_move.push_back(pt);
									std::vector<Eigen::Vector3f> candidate_pts_move_;
									candidate_pts_move_.push_back(pt);
									octree_circle_centers.update_with_attr(candidate_pts_move_, tmp);
								}
							}
						}
						else
						{
							candidate_pts_move.push_back(pt);
							std::vector<Eigen::Vector3f> candidate_pts_move_;
							candidate_pts_move_.push_back(pt);
							octree_circle_centers.update_with_attr(candidate_pts_move_, tmp);
						}
					}
				}
			}
		}
		// candidate_pts_move = candidate_pts_inti;
		// std::cout<<"base_circle_pts!\n";
		// 基础圆形
		std::vector<Eigen::Vector3f> base_circle_pts;
		int num_base_pt = 360;
		float num_base_pt_f = 360;
		for(int i=0; i<num_base_pt; i++)
		{
			float angle = i/num_base_pt_f*2*3.14159256;
			Eigen::Vector3f pt = Eigen::Vector3f::Zero();
			pt[m_remaind_axis[0]] = radius*cos(angle);
			pt[m_remaind_axis[1]] = radius*sin(angle);
			base_circle_pts.push_back(pt);
		}
		// std::cout<<"data_tensor!\n";
		torch::Tensor data_tensor = torch::zeros({candidate_pts_move.size(), num_base_pt, 6}, torch::dtype(torch::kFloat32));
		torch::Tensor circle_tensor = torch::zeros({candidate_pts_move.size(), 3}, torch::dtype(torch::kFloat32));
		auto data_tensor_acces = data_tensor.accessor<float,3>();
		auto circle_tensor_acces = circle_tensor.accessor<float,2>();
		for(int i=0; i<candidate_pts_move.size(); i++)
		{
			std::vector<float> colors;
			colors.push_back(float(std::rand()%255)/255.0);
			colors.push_back(float(std::rand()%255)/255.0);
			colors.push_back(float(std::rand()%255)/255.0);
			for(int j=0; j<3; j++)
				circle_tensor_acces[i][j] = candidate_pts_move[i][j];
			for(int j=0; j<num_base_pt; j++)
			{
				Eigen::Vector3f pt = candidate_pts_move[i] + base_circle_pts[j];
				for(int k=0; k<3; k++)
				{
					data_tensor_acces[i][j][k] = pt[k];
					data_tensor_acces[i][j][3+k] = colors[k];
				}
			}
		}
		circle_centers = candidate_pts_move;
		return std::make_tuple(data_tensor, circle_tensor);;
	}

	torch::Tensor get_ground_map()
	{
		torch::Tensor data_tensor = torch::zeros({m_max_id, 6}, torch::dtype(torch::kFloat32));
		auto data_tensor_acces = data_tensor.accessor<float,2>();
		std::vector<int64_t> pc_ids_out;
		for(int id_=0; id_<m_max_id; id_++)
		{
			if(m_data[id_].size()==0) continue;
			for(int j=0; j<3; j++)
			{
				data_tensor_acces[id_][j] = m_data[id_][j];
			}
			if(m_data[id_][3] == 0.0) data_tensor_acces[id_][3+0]=0.0; // 一般障碍点 黑色
			else if(m_data[id_][3] == 1.0) data_tensor_acces[id_][3+1]=1.0; // 可行地面点 绿色
			else if(m_data[id_][3] == 2.0) data_tensor_acces[id_][3+0]=1.0; // 目标投影点 红色
			else data_tensor_acces[id_][3+2]=1.0; // 未知点 蓝色
			pc_ids_out.push_back(id_);
		}
		torch::Tensor index_tensor = torch::tensor(pc_ids_out, torch::dtype(torch::kLong));
		torch::Tensor selected_rows = data_tensor.index_select(/*dim=*/0, /*index=*/index_tensor);
		return selected_rows;
	}

	// 近邻搜索
	torch::Tensor nearest_search(torch::Tensor pts_attr, torch::Tensor params)
	{
		if(octree_feature.size()<1)
		{
			std::cout<<"Empty octree!\n";
			return torch::empty(0);
		}
		if(pts_attr.dim()<2)
		{
			std::cout<<"The dim of pts_attr should be 2!!\n";
			return torch::empty(0);
		}
		int num = pts_attr.size(0);
		auto pts_attr_acces = pts_attr.accessor<float,2>();
		auto params_acces = params.accessor<float,1>();
		if(pts_attr.size(1)<3)
		{
			std::cout<<"Must be 3D pts!\n";
			return torch::empty(0);;
		}
		int attr_n = octree_feature.get_attr_n();
		torch::Tensor data_tensor = torch::zeros({num, attr_n}, torch::dtype(torch::kFloat32));
		auto data_tensor_acces = data_tensor.accessor<float,2>();
		for(int i=0; i<num; i++)
		{
			std::vector<float> query;
			for(int j=0; j<3; j++)
				query.push_back(pts_attr_acces[i][j]);
			std::vector<std::vector<float>> resultIndices;
			std::vector<float> distances;
			octree_feature.knnNeighbors_eigen(query, 1, resultIndices, distances);
			if(distances.size()>0)
			{
				for(int j=0; j<attr_n; j++)
				{
					data_tensor_acces[i][j] = resultIndices[0][j];
				}
			}
		}
		return data_tensor;
	}

	// KNN
	std::tuple<torch::Tensor, torch::Tensor> knn_nearest_search(torch::Tensor pts_attr, int64_t k)
	{
		if(octree_feature.size()<1)
		{
			std::cout<<"Empty octree!\n";
			return std::tuple<torch::Tensor, torch::Tensor>();
		}
		if(pts_attr.dim()<2)
		{
			std::cout<<"The dim of pts_attr should be 2!!\n";
			return std::tuple<torch::Tensor, torch::Tensor>();
		}
		int num = pts_attr.size(0);
		auto pts_attr_acces = pts_attr.accessor<float,2>();
		if(pts_attr.size(1)<3)
		{
			std::cout<<"Must be 3D pts!\n";
			return std::tuple<torch::Tensor, torch::Tensor>();
		}
		int attr_n = octree_feature.get_attr_n();
		torch::Tensor data_tensor = torch::zeros({num, k, attr_n}, torch::dtype(torch::kFloat32));
		torch::Tensor dist_tensor = torch::zeros({num, k, 1}, torch::dtype(torch::kFloat32));
		auto data_tensor_acces = data_tensor.accessor<float,3>();
		auto dist_tensor_acces = dist_tensor.accessor<float,3>();
		for(int i=0; i<num; i++)
		{
			std::vector<float> query;
			for(int j=0; j<3; j++)
				query.push_back(pts_attr_acces[i][j]);
			std::vector<std::vector<float>> resultIndices;
			std::vector<float> distances;
			octree_feature.knnNeighbors_eigen(query, k, resultIndices, distances);
			if(distances.size()>0)
			{
				for(int ki=0; ki<distances.size() && ki<k; ki++)
				{
					for(int j=0; j<attr_n; j++)
					{
						data_tensor_acces[i][ki][j] = resultIndices[ki][j];
					}
					dist_tensor_acces[i][ki][0] = distances[ki];
				}
			}
		}
		return std::make_tuple(data_tensor, dist_tensor);
	}

	// KNN
	std::tuple<torch::Tensor, torch::Tensor> knn_nearest_search_ground(torch::Tensor pts_attr, int64_t k)
	{
		if(octree_ground.size()<1)
		{
			std::cout<<"Empty octree!\n";
			return std::tuple<torch::Tensor, torch::Tensor>();
		}
		if(pts_attr.dim()<2)
		{
			std::cout<<"The dim of pts_attr should be 2!!\n";
			return std::tuple<torch::Tensor, torch::Tensor>();
		}
		int num = pts_attr.size(0);
		auto pts_attr_acces = pts_attr.accessor<float,2>();
		if(pts_attr.size(1)<3)
		{
			std::cout<<"Must be 3D pts!\n";
			return std::tuple<torch::Tensor, torch::Tensor>();
		}
		int attr_n = octree_ground.get_attr_n();
		torch::Tensor data_tensor = torch::zeros({num, k, attr_n}, torch::dtype(torch::kFloat32));
		torch::Tensor dist_tensor = torch::zeros({num, k, 1}, torch::dtype(torch::kFloat32));
		auto data_tensor_acces = data_tensor.accessor<float,3>();
		auto dist_tensor_acces = dist_tensor.accessor<float,3>();
		for(int i=0; i<num; i++)
		{
			std::vector<float> query;
			for(int j=0; j<3; j++)
				query.push_back(pts_attr_acces[i][j]);
			std::vector<std::vector<float>> resultIndices;
			std::vector<float> distances;
			octree_ground.knnNeighbors_eigen(query, k, resultIndices, distances);
			if(distances.size()>0)
			{
				for(int ki=0; ki<distances.size() && ki<k; ki++)
				{
					for(int j=0; j<attr_n; j++)
					{
						data_tensor_acces[i][ki][j] = resultIndices[ki][j];
					}
					dist_tensor_acces[i][ki][0] = distances[ki];
				}
			}
		}
		return std::make_tuple(data_tensor, dist_tensor);
	}

	// KNN
	std::tuple<torch::Tensor, torch::Tensor> knn_nearest_search_obstacle(torch::Tensor pts_attr, int64_t k)
	{
		if(octree_obstacle.size()<1)
		{
			std::cout<<"Empty octree!\n";
			return std::tuple<torch::Tensor, torch::Tensor>();
		}
		if(pts_attr.dim()<2)
		{
			std::cout<<"The dim of pts_attr should be 2!!\n";
			return std::tuple<torch::Tensor, torch::Tensor>();
		}
		int num = pts_attr.size(0);
		auto pts_attr_acces = pts_attr.accessor<float,2>();
		if(pts_attr.size(1)<3)
		{
			std::cout<<"Must be 3D pts!\n";
			return std::tuple<torch::Tensor, torch::Tensor>();
		}
		int attr_n = octree_obstacle.get_attr_n();
		torch::Tensor data_tensor = torch::zeros({num, k, attr_n}, torch::dtype(torch::kFloat32));
		torch::Tensor dist_tensor = torch::zeros({num, k, 1}, torch::dtype(torch::kFloat32));
		auto data_tensor_acces = data_tensor.accessor<float,3>();
		auto dist_tensor_acces = dist_tensor.accessor<float,3>();
		for(int i=0; i<num; i++)
		{
			std::vector<float> query;
			for(int j=0; j<3; j++)
				query.push_back(pts_attr_acces[i][j]);
			std::vector<std::vector<float>> resultIndices;
			std::vector<float> distances;
			octree_obstacle.knnNeighbors_eigen(query, k, resultIndices, distances);
			if(distances.size()>0)
			{
				for(int ki=0; ki<distances.size() && ki<k; ki++)
				{
					for(int j=0; j<attr_n; j++)
					{
						data_tensor_acces[i][ki][j] = resultIndices[ki][j];
					}
					dist_tensor_acces[i][ki][0] = distances[ki];
				}
			}
		}
		return std::make_tuple(data_tensor, dist_tensor);
	}

    torch::Tensor pts_voxel_filter(torch::Tensor pts_attr, double voxel_size = 0.01, bool no_sort = false)
	{
		if(pts_attr.dim()!=2)
		{
			std::cout<<"The dim of pts_attr should be 2!!\n";
			return torch::empty(0);
		}
		if(pts_attr.size(1)<3)
		{
			std::cout<<"Must be 3D pts!\n";
			return torch::empty(0);
		}
		int num = pts_attr.size(0);
		int attr_n = pts_attr.size(1);
		auto pts_attr_acces = pts_attr.accessor<float,2>();
		std::unordered_map<VOXEL_LOC, std::vector<int>> feat_map_tmp;
		auto data_ptr = pts_attr.data_ptr<float>();
		for (int64_t i = 0; i < num; ++i)
		{
			// for (int64_t j = 0; j < attr_n; ++j) {
			// 	float value = data_ptr[i * attr_n + j];
			// }
			int64_t x = std::round(data_ptr[i * attr_n + 0]/voxel_size);
			int64_t y = std::round(data_ptr[i * attr_n + 1]/voxel_size);
			int64_t z = std::round(data_ptr[i * attr_n + 2]/voxel_size);
			VOXEL_LOC position(x, y, z);
			feat_map_tmp[position].push_back(i);
		}
		// 两种方式差别也不是很大。。。。直接指针的方式快一点点
		// for (int i=0; i<num; i++)
		// {
		// 	int64_t x = std::round(pts_attr_acces[i][0]/voxel_size);
		// 	int64_t y = std::round(pts_attr_acces[i][1]/voxel_size);
		// 	int64_t z = std::round(pts_attr_acces[i][2]/voxel_size);
		// 	VOXEL_LOC position(x, y, z);
		// 	feat_map_tmp[position].push_back(i);
		// }
		std::vector<int64_t> pc_ids_out;
		for (auto iter = feat_map_tmp.begin(); iter != feat_map_tmp.end(); ++iter)
		{
			if(no_sort)
			{
				pc_ids_out.push_back(iter->second[0]);
				continue;
			}
			int best_id = 0;
			float min_dist = 1e8;
			int pt_n = iter->second.size();
			Eigen::Vector3f center(iter->first.x, iter->first.y, iter->first.z);
			center *= voxel_size;
			for(int i=0; i<pt_n; i++)
			{
				int id = iter->second[i];
				// float dist = (center - Eigen::Vector3f(pts_attr_acces[id][0], pts_attr_acces[id][1], pts_attr_acces[id][2])).norm();
				float dist = (center - Eigen::Vector3f(data_ptr[id * attr_n + 0], data_ptr[id * attr_n + 1], data_ptr[id * attr_n + 2])).norm();
				if(dist<min_dist)
				{
					min_dist = dist;
					best_id = id;
				}
			}
			pc_ids_out.push_back(best_id);
		}
		torch::Tensor index_tensor = torch::tensor(pc_ids_out, torch::dtype(torch::kLong));
		torch::Tensor selected_rows = pts_attr.index_select(/*dim=*/0, /*index=*/index_tensor);
		return selected_rows;
	}

	// radiusNeighbors
	std::vector<std::tuple<torch::Tensor, torch::Tensor>> radius_neighbors(torch::Tensor pts_attr, double radius)
	{
		if(octree_feature.size()<1)
		{
			std::cout<<"Empty octree!\n";
			return std::vector<std::tuple<torch::Tensor, torch::Tensor>>();
		}
		if(pts_attr.dim()<2)
		{
			std::cout<<"The dim of pts_attr should be 2!!\n";
			return std::vector<std::tuple<torch::Tensor, torch::Tensor>>();
		}
		int num = pts_attr.size(0);
		auto pts_attr_acces = pts_attr.accessor<float,2>();
		if(pts_attr.size(1)<3)
		{
			std::cout<<"Must be 3D pts!\n";
			return std::vector<std::tuple<torch::Tensor, torch::Tensor>>();
		}
		std::vector<std::tuple<torch::Tensor, torch::Tensor>> results;
		int attr_n = octree_feature.get_attr_n();
		for(int i=0; i<num; i++)
		{
			std::vector<float> query;
			for(int j=0; j<3; j++)
				query.push_back(pts_attr_acces[i][j]);
			std::vector<std::vector<float>> resultIndices;
			std::vector<float> distances;
			octree_feature.radiusNeighbors_eigen(query, radius, resultIndices, distances);
			int k = distances.size();
			torch::Tensor data_tensor = torch::zeros({k, attr_n}, torch::dtype(torch::kFloat32));
			torch::Tensor dist_tensor = torch::zeros({k, 1}, torch::dtype(torch::kFloat32));
			auto data_tensor_acces = data_tensor.accessor<float,2>();
			auto dist_tensor_acces = dist_tensor.accessor<float,2>();
			if(k>0)
			{
				for(int ki=0; ki<k; ki++)
				{
					for(int j=0; j<attr_n; j++)
					{
						data_tensor_acces[ki][j] = resultIndices[ki][j];
					}
					dist_tensor_acces[ki][0] = distances[ki];
				}
			}
			results.push_back(std::make_tuple(data_tensor, dist_tensor));
		}
		return results;
	}

	//添加点数据
	void add_pts_with_attr_cpu(torch::Tensor pts_attr)
	{
		// params[0] min_depth
		// std::cout<<"pts_attr.type(): "<<pts_attr.type()<<std::endl; // CPUFloatType
		int num = pts_attr.size(0);
		const int attr_n = pts_attr.size(1)-3;
		auto pts_attr_acces = pts_attr.accessor<float,2>();
		// torch::Tensor params_cpu = params.to(torch::kCPU);
		// auto params_acces = params.accessor<float,1>();
		// std::cout<<"num: "<<num<<std::endl;
		// std::cout<<"attr_n: "<<attr_n<<std::endl;
		if(attr_n<0)
		{
			std::cout<<"Must be 3D pts!\n";
			return;
		}
		std::vector<std::vector<float>> pts(num), extra_attr(num);
		for(int i=0; i<num; i++)
		{
			for(int j=0; j<3; j++)
				pts[i].push_back(pts_attr_acces[i][j]);
			for(int j=0; j<attr_n; j++)
				extra_attr[i].push_back(pts_attr_acces[i][j+3]);
		}
		if(octree_feature.size()==0)
			octree_feature.initialize_with_attr(pts, extra_attr);
		else
			octree_feature.update_with_attr(pts, extra_attr);
		// std::cout<<"octree_feature.get_size(): "<<octree_feature.get_size()<<std::endl;
	}

	int64_t get_size(){return octree_feature.get_size();}

	torch::Tensor get_data()
	{
		std::vector<std::vector<float>> orig_data= octree_feature.get_orig_data();
		int num = orig_data.size();
		if(num<1)
		{
			return torch::empty(0);
		}
		int attr_n = orig_data[0].size();
		torch::Tensor data_tensor = torch::zeros({num, attr_n}, torch::dtype(torch::kFloat32));
		auto data_tensor_acces = data_tensor.accessor<float,2>();
		for(int i=0; i<num; i++)
		{
			for(int j=0; j<attr_n; j++)
			{
				data_tensor_acces[i][j] = orig_data[i][j];
			}
		}
		return data_tensor;
	}

	template <typename PCType = std::vector<std::vector<float>> >
    void pc_voxel_filter(const PCType & pc_in, int size, std::vector<int> & pc_ids_out, float voxel_size = 0.2, bool no_sort = false)
	{
		std::unordered_map<VOXEL_LOC, std::vector<int>> feat_map_tmp;
		for (int i=0; i<size; i++)
		{
			int64_t x = std::round(pc_in[i][0]/voxel_size);
			int64_t y = std::round(pc_in[i][1]/voxel_size);
			int64_t z = std::round(pc_in[i][2]/voxel_size);
			VOXEL_LOC position(x, y, z);
			feat_map_tmp[position].push_back(i);
		}
		pc_ids_out.clear();
		for (auto iter = feat_map_tmp.begin(); iter != feat_map_tmp.end(); ++iter)
		{
			if(no_sort)
			{
				pc_ids_out.push_back(iter->second[0]);
				continue;
			}
			int best_id = 0;
			float min_dist = 1e8;
			int pt_n = iter->second.size();
			Eigen::Vector3f center(iter->first.x, iter->first.y, iter->first.z);
			center *= voxel_size;
			for(int i=0; i<pt_n; i++)
			{
				int id = iter->second[i];
				float dist = (center - Eigen::Vector3f(pc_in[id][0], pc_in[id][1], pc_in[id][2])).norm();
				if(dist<min_dist)
				{
					min_dist = dist;
					best_id = id;
				}
			}
			// pc_out.push_back(pc_in[best_id]);
			pc_ids_out.push_back(best_id);
		}
	}

	void debug_print()
	{
		std::cout<<"=============================="<<std::endl;
		std::cout<<"This is a debug print in OctreeMap C++!"<<std::endl;
		std::cout<<"=============================="<<std::endl;
	}

};

#endif


