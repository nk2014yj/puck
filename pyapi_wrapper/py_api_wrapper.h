//Copyright (c) 2023 Baidu, Inc.  All Rights Reserved.
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

/**
 * @file py_puck_api_wrapper.h
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2021/8/18 14:30
 * @brief
 *
 **/
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <memory>
#include "puck/index.h"
#include "puck/gflags/puck_gflags.h"
namespace py_puck_api {
namespace py = pybind11;

void update_gflag(const char* gflag_key, const char* gflag_val);
class PySearcher {
public:
    PySearcher();
    void show();
    int init();
    int build(uint32_t n);
    int search(uint32_t n, py::array_t<float>& query_fea,const uint32_t topk, py::array_t<float>& distance, py::array_t<uint32_t>& labels);
    ~PySearcher();
private:
    std::unique_ptr<puck::Index> _index;
    uint32_t _dim;
};
PYBIND11_MODULE(py_puck, m){
        m.doc() = "puck";
        pybind11::class_<PySearcher>(m, "PySearcher")
        .def(pybind11::init())
        .def("show", &PySearcher::show)
        .def("init", &PySearcher::init)
        .def("build", &PySearcher::build)
        .def("search", &PySearcher::search)
        .def("show", &PySearcher::show);

        m.def("update_gflag", &update_gflag);
}

};//namespace py_puck_api

