
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

#include "octree_map.hpp"


TORCH_LIBRARY(octree_map, m)
{
	m.class_<OctreeMap>("OctreeMap")
        .def(torch::init())
        .def("nearest_search", &OctreeMap::nearest_search)
        .def("knn_nearest_search", &OctreeMap::knn_nearest_search)
        .def("knn_nearest_search_ground", &OctreeMap::knn_nearest_search_ground)
        .def("knn_nearest_search_obstacle", &OctreeMap::knn_nearest_search_obstacle)
        .def("radius_neighbors", &OctreeMap::radius_neighbors)
        .def("clear", &OctreeMap::clear)
		.def("get_data", &OctreeMap::get_data)
        .def("add_pts_with_attr_cpu", &OctreeMap::add_pts_with_attr_cpu)
		.def("debug_print", &OctreeMap::debug_print)
		.def("get_size", &OctreeMap::get_size)
		.def("get_ground_map", &OctreeMap::get_ground_map)
		.def("label_occupied_grid", &OctreeMap::label_occupied_grid)
		.def("check_bounding_box_valid", &OctreeMap::check_bounding_box_valid)
		.def("get_object_envelope_circles", &OctreeMap::get_object_envelope_circles)
		.def("update_object_envelope_circles", &OctreeMap::update_object_envelope_circles)
		.def("check_obstacle_along_dir", &OctreeMap::check_obstacle_along_dir,"", {torch::arg("center"), torch::arg("dir"), torch::arg("only_obstacle") = false})
		.def("pts_voxel_filter", &OctreeMap::pts_voxel_filter, "", {torch::arg("pts_attr"), torch::arg("voxel_size")=0.01, torch::arg("no_sort") = false})
        .def("add_pts_to_ground_map", &OctreeMap::add_pts_to_ground_map, "", {torch::arg("pts_attr"), torch::arg("attr")=1.0})
        .def("ground_map_setup", &OctreeMap::ground_map_setup, "", {torch::arg("center"), torch::arg("radius")=1.0, torch::arg("grid_size")=0.01, torch::arg("ignore_axis")=1})
    ;
}
