#include "globals.cuh"

void copy2cpu( cudaLibXtDesc *d_var, Complex *h_var ) {

	for(int gpu_i=0;gpu_i<num_gpus;gpu_i++) {

                cudaSetDevice( gpu_i );
                lower = chunk_size_real*gpu_i;
                upper = min(lower+chunk_size_real,M2);
                width = upper-lower;

                //DCF and Wave function magnitude initialized on CPU -> xfer GPU
                cudaMemcpy(h_var+lower, (cufftDoubleComplex *)d_var -> descriptor -> data[gpu_i], sizeof( Complex )*width , cudaMemcpyDeviceToHost);
                //cudaMemcpy(k2[gpu_i], k2_host+lower, sizeof( float )*width , cudaMemcpyHostToDevice);
		//cudaDeviceSynchronize();
        }
	
}
