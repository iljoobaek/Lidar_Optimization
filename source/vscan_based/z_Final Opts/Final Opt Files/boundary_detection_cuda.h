#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
#include <iostream>


#ifndef FILE_DATA_STRUCTS
#define FILE_DATA_STRUCTS
#include "data_structs.h"
#endif

void rearrange_pc_cuda(std::vector<PC_Point>& pointcloud, std::vector<int>& index_mapping, std::vector<std::vector<int>>& ranges);