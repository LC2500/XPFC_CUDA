#include "globals.cuh"

void convolution_wrapper( cudaLibXtDesc *st, double **c2_k, cudaLibXtDesc *st_convl) {
	
	int device;

	for (int gpu_i = 0; gpu_i<num_gpus; gpu_i++){
		dim3 dimGrid( grid_size.x, grid_size.y, grid_size.z );
                dim3 dimBlock( block_size.x, block_size.y, block_size.z );

		device = st -> descriptor -> GPUs[gpu_i];
		cudaSetDevice(device);

		convolve_fields<<<dimGrid, dimBlock>>>( (cufftDoubleComplex*)st->descriptor->data[gpu_i], c2_k[gpu_i], (cufftDoubleComplex*)st_convl->descriptor->data[gpu_i], gpu_i, Mx, num_gpus);

	}

	for (int gpu_i = 0; gpu_i < num_gpus; gpu_i++) {
                device = st_convl->descriptor->GPUs[gpu_i];
                cudaSetDevice(device);
                cudaDeviceSynchronize(); // Wait for memory transfer to complete before accessing data
                //printf("convolve_fields Printing GPU: %d \n",gpu_i);
                // Check if kernel execution generated and error
                getLastCudaError("Kernel execution failed [ apply_power ]");
        }

}


__global__ void convolve_fields( cufftDoubleComplex *psi_k, double *c2_k, cufftDoubleComplex *convl, int gpu_i, int Mx, int num_gpus) {
	
	int xgrid_idx = threadIdx.x + blockIdx.x*blockDim.x; // column index (index this way for finite difference grid)
        int xgrid_stride = gridDim.x*blockDim.x;

        int ygrid_idx = threadIdx.y + blockIdx.y*blockDim.y; // row index (index this way for finite difference grid)
        int ygrid_stride = gridDim.y*blockDim.y;

	for (int i = xgrid_idx; i < Mx; i+= xgrid_stride){
                for (int j = ygrid_idx; j < Mx/num_gpus; j+= ygrid_stride){
			
                        convl[i + j*Mx].x = c2_k[i + j*Mx]*psi_k[i + j*Mx].x;
			convl[i + j*Mx].y = c2_k[i + j*Mx]*psi_k[i + j*Mx].y;
			
			//printf( "gpu_id %d index %d row %d column %d convl = (%0.06e,%0.06e)\n", gpu_i, i + j*Mx, j + (Mx/num_gpus)*gpu_i, i,  convl[i + j*Mx].x, convl[i * j*Mx].y ); // For debugging purposes



    		}
	}

}



