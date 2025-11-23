#include "globals.cuh"


void copy_matrix_wrapper(cudaLibXtDesc *st_in, cudaLibXtDesc *st_out) {

	int device;

        for (int gpu_i = 0; gpu_i<num_gpus; gpu_i++){
                dim3 dimGrid( grid_size.x, grid_size.y, grid_size.z );
                dim3 dimBlock( block_size.x, block_size.y, block_size.z );

                //printf("GPUID: %d dimGrid: (%u, %u, %u)\n", gpu_i, dimGrid.x, dimGrid.y, dimGrid.z);
                //printf("GPUID: %d dimBlock: (%u, %u, %u)\n", gpu_i, dimBlock.x, dimBlock.y, dimBlock.z);

                device = st_in->descriptor->GPUs[gpu_i];
                cudaSetDevice(device);

                copy_matrix<<<dimGrid, dimBlock>>>( (cufftDoubleComplex*)st_in->descriptor->data[gpu_i], (cufftDoubleComplex*)st_out->descriptor->data[gpu_i], gpu_i, Mx, num_gpus);
                cudaDeviceSynchronize();
        }

        for (int gpu_i = 0; gpu_i < num_gpus; gpu_i++) {
                device = st_out->descriptor->GPUs[gpu_i];
                cudaSetDevice(device);
                cudaDeviceSynchronize(); // Wait for memory transfer to complete before accessing data
                //printf("copy_matrix Printing GPU: %d \n",gpu_i);
                // Check if kernel execution generated and error
                getLastCudaError("Kernel execution failed [ apply_power ]");
        }

}


__global__ void copy_matrix(cufftDoubleComplex *psi_in, cufftDoubleComplex *psi_out, int gpu_i, int Mx, int num_gpus) {

	int xgrid_idx = threadIdx.x + blockDim.x*blockIdx.x;
	int xgrid_stride = gridDim.x*blockDim.x;

	int ygrid_idx = threadIdx.y + blockIdx.y*blockDim.y;
        int ygrid_stride = gridDim.y*blockDim.y;

	for(int i = xgrid_idx; i < Mx; i+=xgrid_stride) {
		for(int j = ygrid_idx; j < Mx/num_gpus; j+=ygrid_stride) {
			psi_out[i+j*Mx].x = psi_in[i+j*Mx].x;
			psi_out[i+j*Mx].y = 0.0;
		}
	}

}














