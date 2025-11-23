#include "globals.cuh"


void power_function_wrapper(cudaLibXtDesc *st, cudaLibXtDesc *stN, int pow_i ) {

        int device;

        for (int gpu_i = 0; gpu_i<num_gpus; gpu_i++){
                dim3 dimGrid( grid_size.x, grid_size.y, grid_size.z );
                dim3 dimBlock( block_size.x, block_size.y, block_size.z );

                //printf("GPUID: %d dimGrid: (%u, %u, %u)\n", gpu_i, dimGrid.x, dimGrid.y, dimGrid.z);
                //printf("GPUID: %d dimBlock: (%u, %u, %u)\n", gpu_i, dimBlock.x, dimBlock.y, dimBlock.z);

                device = st->descriptor->GPUs[gpu_i];
                cudaSetDevice(device);

                apply_power<<<dimGrid, dimBlock>>>( (cufftDoubleComplex*)st->descriptor->data[gpu_i], (cufftDoubleComplex*)stN->descriptor->data[gpu_i], gpu_i, Mx, chunk_size_real, num_gpus, pow_i);
                cudaDeviceSynchronize();
        }

        for (int gpu_i = 0; gpu_i < num_gpus; gpu_i++) {
                device = stN->descriptor->GPUs[gpu_i];
                cudaSetDevice(device);
                cudaDeviceSynchronize(); // Wait for memory transfer to complete before accessing data
                //printf("apply_power Printing GPU: %d \n",gpu_i);
                // Check if kernel execution generated and error
                getLastCudaError("Kernel execution failed [ apply_power ]");
        }


}

__global__ void apply_power(cufftDoubleComplex *psi, cufftDoubleComplex *psiNm, int gpu_i, int Mx, int chunk_size_real, int num_gpus, int pow_i) {

        int xgrid_idx = threadIdx.x + blockIdx.x*blockDim.x; // column index (index this way for finite difference grid)
        int xgrid_stride = gridDim.x*blockDim.x;

        int ygrid_idx = threadIdx.y + blockIdx.y*blockDim.y; // row index (index this way for finite difference grid)
        int ygrid_stride = gridDim.y*blockDim.y;

        for (int i = xgrid_idx; i < Mx; i+= xgrid_stride){
                for (int j = ygrid_idx; j < Mx/num_gpus; j+= ygrid_stride){

                        // printf("gpu_id %d index %d row %d column %d psi%d = %0.6f\n", gpu_i, i*Mx + j, i + (Mx/num_gpus)*gpu_i, j, pow_i, psi[i*Mx + j] ); For debugging purposes
                        // Works, uncomment when resuming code
                        double a = powf(psi[i + j*Mx].x, pow_i);
                        psiNm[i + j*Mx].x = a;
			psiNm[i + j*Mx].y = 0.0;
                }
        }

}


