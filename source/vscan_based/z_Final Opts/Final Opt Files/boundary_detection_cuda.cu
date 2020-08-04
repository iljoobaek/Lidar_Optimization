#include "boundary_detection_cuda.h"

/*
host:   create thrust sequence
device: use sequence to get all indices of pointcloud that pass predicate
host:   sequentially append indices that failed predicate
            get size of indces that passed predicate to update ranges
            update cur_idx
device: use new index array to index into index_mapping_copy to update index_mapping
device: use new index array to index into pointcoud_copy to update pointcloud

*/

// __global__
// void init_simple_virtualscan() {
//     // svs.resize(beamnum); // svs is qvector<qvector<simple_virtual_scan>>
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
// 	int stride = blockDim.x * gridDim.x;
// }

void rearrange_pc_cuda(std::vector<PC_Point>& pointcloud, std::vector<int>& index_mapping, std::vector<std::vector<int>>& ranges) {
	const int pc_size = pointcloud.size();

	// thrust::device_vector<PC_Point> d_pc(pointcloud);
	// thrust::device_vector<int> d_im;
	// thrust::device_vector<int> seq_indices(pc_size);
	// thrust::sequence(thrust::device, seq_indices.begin(), seq_indices.end(), 0);
	// thrust::host_vector<PC_Point> v(d_pc);

	// printf("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n");
	// printf("%f %f\n", v[10][4], pointcloud[10][4]);

	// struct is_even {
	// 	int idx;
	// 	thrust::device_vector<PC_Point> dpc;
	// 	__device__
	// 		bool operator()(const int x) {
	// 			return dpc[x][4] == static_cast<float>(idx) && dpc[x][6] > 0;
	// 	}
	// }isEven;

	// isEven.idx = 0;
	// isEven.dpc = d_pc;
	// thrust::copy_if(thrust::device, seq_indices.begin(), seq_indices.end(), d_im.begin(), is_even());

	// int blockSize = 256;
	// int numBlocks = (pc_size*pc0_size + blockSize - 1) / blockSize;
	// fill_dvec<<<numBlocks, blockSize>>>(d_pc, thrust::raw_pointer_cast(pointcloud.data()), pc_size, pc0_size);
	// cudaDeviceSynchronize();
		
	// thrust::device_vector<int> d_im(index_mapping);
	// // thrust::device_vector<thrust::device_vector<int>> d_ranges(ranges);
	
	// thrust::device_vector<double> indices;
	// thrust::copy_if(seq_indices.begin(), seq_indices.end(), indices.begin(), )
}