include_directories(..)
file(GLOB PY_USED_CPP
        "../puck/*.cpp"
        "../puck/base/*.cpp"
        "../puck/gflags/*.cpp"
        "../puck/hierarchical_cluster/*.cpp"
        "../puck/puck/puck_index.cpp"
        "../puck/puck/quantization.cpp"
        "../puck/puck/distances_simd.cpp"
        "../puck/tinker/method/hnsw.cc"
        "../puck/tinker/method/tinker_distfunc_opt.cc"
        "../puck/tinker/tinker_index.cpp"
        "./py_api_wrapper.cpp")

 # find python3
 find_package(Python3 ${PYTHON_MINIMUM_VERSION} REQUIRED COMPONENTS Interpreter Development)
 find_package(pybind11 REQUIRED) #找到依赖
 
 message(STATUS "Found Python: ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}.${Python3_VERSION_PATCH}")
 if (NOT Python3_SITELIB)
     message(FATAL_ERROR "site-packages not found. ")
 else ()
     message(STATUS "site-packages: ${Python3_SITELIB}")
 endif ()

 find_package(PythonLibs "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}" EXACT REQUIRED)
message(STATUS "Found PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}")
message(STATUS "Found PYTHON_LIBRARY: ${PYTHON_LIBRARY}")
message(STATUS "Found CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

set(PYTHON_EXECUTABLE "$/home/peisi.yj//miniconda3/envs/py38/bin/python3")

pybind11_add_module(py_puck_api py_api_wrapper.cpp)#重点
target_link_libraries(py_puck_api PRIVATE puck)