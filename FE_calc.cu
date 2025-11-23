#include "globals.cuh"

void FE_calc_wrapper( int n, cudaLibXtDesc *d_psi, cudaLibXtDesc *d_psi2, cudaLibXtDesc *d_psi3, cudaLibXtDesc *d_psi4, cudaLibXtDesc *d_convl, double **ff, double *gpu_reduced_array, double *reduced_array, ncclComm_t *comms ) {
	
	int device;

        for (int gpu_i = 0; gpu_i<num_gpus; gpu_i++){
                dim3 dimGrid( grid_size.x, grid_size.y, grid_size.z );
                dim3 dimBlock( block_size.x, block_size.y, block_size.z );

                device = d_psi -> descriptor -> GPUs[gpu_i];
                cudaSetDevice(device);

                FE_calc<<<dimGrid, dimBlock>>>( n, a, b, (cufftDoubleComplex*)d_psi->descriptor->data[gpu_i], (cufftDoubleComplex*)d_psi2->descriptor->data[gpu_i], (cufftDoubleComplex*)d_psi3->descriptor->data[gpu_i], (cufftDoubleComplex*)d_psi4->descriptor->data[gpu_i], (cufftDoubleComplex*)d_convl->descriptor->data[gpu_i], gpu_i, Mx, M2, num_gpus, ff[gpu_i] );
		

		//cudaDeviceSynchronize();
        }

	
        for (int gpu_i = 0; gpu_i < num_gpus; gpu_i++) {
                device = d_psi->descriptor->GPUs[gpu_i];
                cudaSetDevice(device);
               	 
		ncclReduce(
           		ff[gpu_i],         // Input buffer for GPU i
            		gpu_reduced_array,     // Output buffer on the root GPU
            		chunk_size_real,   // Total number of elements
            		ncclDouble,         // Data type
            		ncclSum,           // Reduction operation
            		root_GPU,          // Root GPU
            		comms[gpu_i],      // NCCL communicator
            		0                  // Default CUDA stream
        	);	
		
		//printf("FE_Calc Printing GPU: %d \n",gpu_i);
                // Check if kernel execution generated and error
                getLastCudaError("Kernel execution failed [ apply_power ]");
        }
	

	for (int gpu_i = 0; gpu_i<num_gpus; gpu_i++){
                device = d_psi -> descriptor -> GPUs[gpu_i];
                cudaSetDevice(device);
                cudaDeviceSynchronize();
        }


	deviceReduceKernel<<<sr_blocks, sr_threads>>>( gpu_reduced_array, reduced_array, chunk_size_real );
	deviceReduceKernelFinalCall<<<1, 1024>>>(reduced_array, reduced_array, sr_blocks, n, ff_factor);
}


__global__ void FE_calc( int n, double a, double b, cufftDoubleComplex *psi, cufftDoubleComplex *psi_2, cufftDoubleComplex *psi_3, cufftDoubleComplex *psi_4, cufftDoubleComplex *convl_r, int gpu_i, int Mx, int M2, int num_gpus, double* ff ) {

	int xgrid_idx = threadIdx.x + blockIdx.x*blockDim.x; // column index (index this way for finite difference grid)
        int xgrid_stride = gridDim.x*blockDim.x;

        int ygrid_idx = threadIdx.y + blockIdx.y*blockDim.y; // row index (index this way for finite difference grid)
        int ygrid_stride = gridDim.y*blockDim.y;

	for (int i = xgrid_idx; i < Mx; i+= xgrid_stride){
                for (int j = ygrid_idx; j < Mx/num_gpus; j+= ygrid_stride){
			ff[i + j*Mx] = 0.5*psi_2[i + j*Mx].x - (a/3.0)*psi_3[i + j*Mx].x + (b/4.0)*psi_4[i + j*Mx].x - 0.5*psi[i + j*Mx].x*convl_r[i + j*Mx].x;///(float)M2/(float)M2; // ff1 - c, ff2 - c, ff3 - c, ff4 - c  
			//printf( "gpu_id %d index %d row %d column %d ff = %0.06e\n", gpu_i, i + j*Mx, j + (Mx/num_gpus)*gpu_i, i, ff[i + j*Mx] ); // For debugging purposes // ff1 , ff2 , ff3 , ff4 
    		}
	}
}

