#include "globals.cuh"


void scaling_step_wrapper ( cudaLibXtDesc *st) {
	
	int device;

	for(int gpu_i=0; gpu_i<num_gpus; gpu_i++) {
		dim3 dimGrid( grid_size.x, grid_size.y, grid_size.z );
                dim3 dimBlock( block_size.x, block_size.y, block_size.z );
		
		device = st -> descriptor -> GPUs[gpu_i];
		cudaSetDevice(device);

		scaling_step<<<dimGrid, dimBlock>>>( (cufftDoubleComplex *)st -> descriptor -> data[gpu_i], gpu_i, Mx, M2, num_gpus );

	}
	
	for (int gpu_i = 0; gpu_i < num_gpus; gpu_i++) {

                device = st->descriptor->GPUs[gpu_i];
                cudaSetDevice(device);
                cudaDeviceSynchronize(); // Wait for memory transfer to complete before accessing data
                //printf("Scaling Step Printing GPU: %d \n",gpu_i);
                // Check if kernel execution generated and error
                getLastCudaError("Kernel execution failed [ apply_power ]");        
	}


}



__global__ void scaling_step( cufftDoubleComplex *f, int gpu_i, int Mx, int M2, int num_gpus ) {
	
	int xgrid_idx = threadIdx.x + blockIdx.x*blockDim.x; // column index (index this way for finite difference grid)
        int xgrid_stride = gridDim.x*blockDim.x;

        int ygrid_idx = threadIdx.y + blockIdx.y*blockDim.y; // row index (index this way for finite difference grid)
        int ygrid_stride = gridDim.y*blockDim.y;

        int cols = Mx;
	int rows = Mx/num_gpus;

        for( int i = xgrid_idx; i < cols; i += xgrid_stride ) { // i - columns, contiguous in memory and the fastest changing dimension
                for( int j = ygrid_idx; j < rows; j += ygrid_stride ) { // j - rows
			int idx = i + j*Mx;
			f[ idx ].x = f[ idx ].x/(double)M2;
			f[ idx ].y = f[ idx ].y/(double)M2;
		}
	}

}
