#include "globals.cuh"

void SSE_calc_wrapper( int n, cufftHandle planComplex, cudaLibXtDesc* res_out, cudaLibXtDesc* res_k, cudaLibXtDesc* psi_kp1, cudaLibXtDesc* psi_k, cudaLibXtDesc* psi_2k, cudaLibXtDesc* psi_3k, double** c2_k, double** k2, double** ff, double *gpu_reduced_array, double *reduced_array, ncclComm_t *comms ) {
	
	int device;
	dim3 dimGrid( grid_size.x, grid_size.y, grid_size.z );
        dim3 dimBlock( block_size.x, block_size.y, block_size.z );
	
	//printf("\n Psi_r Euler step %d:\n", n);
        for (int gpu_i = 0; gpu_i<num_gpus; gpu_i++){
                device = psi_kp1 -> descriptor -> GPUs[gpu_i];
                cudaSetDevice(device);
		
                residual_calc<<<dimGrid, dimBlock>>>( n, 
		(cufftDoubleComplex*)res_k->descriptor->data[gpu_i],
		(cufftDoubleComplex*)psi_kp1->descriptor->data[gpu_i], 
		(cufftDoubleComplex*)psi_k->descriptor->data[gpu_i], 
		(cufftDoubleComplex*)psi_2k->descriptor->data[gpu_i], 
		(cufftDoubleComplex*)psi_3k->descriptor->data[gpu_i], 
		c2_k[gpu_i], k2[gpu_i], Mo, dt, a, b, gpu_i, Mx, num_gpus);
		//cudaDeviceSynchronize();
		// dpsi_dt, d2mu_k_dx2, 
	}	
	
        for (int gpu_i = 0; gpu_i < num_gpus; gpu_i++) {
                device = res_k->descriptor->GPUs[gpu_i];
                cudaSetDevice(device);
                cudaDeviceSynchronize(); // Wait for memory transfer to complete before accessing data
                // Check if kernel execution generated and error
                getLastCudaError("Kernel execution failed [ residual_calc ]");
        }	
	
	result = cufftXtExecDescriptor( planComplex, res_k, res_k, CUFFT_INVERSE );
	scaling_step_wrapper( res_k );
		
	//result = cufftXtMemcpy( planComplex, res_out, res_k, CUFFT_COPY_DEVICE_TO_DEVICE );
	res_k -> subFormat = 2;
	power_function_wrapper(res_k, res_k, 2);
	
	/*	
	copy2cpu( res_out, psi_host_out );
	double sse = 0.0;
        for (int i=0;i<Mx;i++){
        	for (int j=0;j<My;j++){
			sse += psi_host_out[i*Mx + j].x;
			//printf( "Array[%d,%d] = (%0.5f, %0.5f) \n",i, j, psi_host_out[i*Mx + j].x, psi_host_out[i*Mx + j].y );
                }
      	}	
	printf( "SSE_1 = %0.5e \n", sse );	

	exit(EXIT_SUCCESS);		
	*/	

	for(int gpu_i = 0; gpu_i < num_gpus; gpu_i++) {
                device = res_out->descriptor->GPUs[gpu_i];
                cudaSetDevice( device );
		copy_struct<<<dimGrid, dimBlock>>>( (cufftDoubleComplex*) res_k->descriptor->data[gpu_i], ff[gpu_i], gpu_i, Mx, num_gpus );
		cudaDeviceSynchronize( ); // Wait for memory transfer to complete before accessing data
	}
	
	/*	
	 // For Debugging //
	for(int gpu_i=0;gpu_i<num_gpus;gpu_i++) {

                cudaSetDevice( gpu_i );
                lower = chunk_size_real*gpu_i;
                upper = min(lower+chunk_size_real,M2);
                width = upper-lower;

                //DCF and Wave function magnitude initialized on CPU -> xfer GPU
                cudaError_t result = cudaMemcpy( SUB1_host + lower, ff[gpu_i], sizeof( double )*width , cudaMemcpyDeviceToHost );
		//printf("CUDA error: %s\n", cudaGetErrorString(result));
		cudaDeviceSynchronize();
        }

	double sse2 = 0.0;
        for (int i=0;i<Mx;i++){
                for (int j=0;j<My;j++){
                        //printf( "Array[%d,%d] = (%0.5e) \n",i, j, SUB1_host[i*Mx + j] );
			sse2 += SUB1_host[i*Mx + j];
                }
        }
	
	printf( "SSE_2 = %0.5e \n", sse );
	*/
	
	for (int gpu_i = 0; gpu_i < num_gpus; gpu_i++) {
                device = res_k->descriptor->GPUs[gpu_i];
                cudaSetDevice(device);
                //cudaDeviceSynchronize(); // Wait for memory transfer to complete before accessing data
               		
                ncclReduce(
                        ff[gpu_i],         // Input buffer for GPU i
                        gpu_reduced_array, // Output buffer on the root GPU
                        chunk_size_real,   // Total number of elements
                        ncclDouble,        // Data type
                        ncclSum,           // Reduction operation
                        root_GPU,          // Root GPU
                        comms[gpu_i],      // NCCL communicator
                        0                  // Default CUDA stream
                );
		
		// Check if kernel execution generated and error
                getLastCudaError("Kernel execution failed [ apply_power (SSE) ]");
        }

	for (int gpu_i = 0; gpu_i<num_gpus; gpu_i++){
                device = res_k -> descriptor -> GPUs[gpu_i];
                cudaSetDevice(device);
                cudaDeviceSynchronize();
        }

	deviceReduceKernel<<<sr_blocks, sr_threads>>>( gpu_reduced_array, reduced_array, chunk_size_real );
        deviceReduceKernelFinalCallSSE<<<1, 1024>>>( reduced_array, reduced_array, sr_blocks, M2, n );
	
}



__global__ void residual_calc( int n, cufftDoubleComplex *res_k, cufftDoubleComplex *psi_kp1, cufftDoubleComplex *psi_k, cufftDoubleComplex *psi_2k, cufftDoubleComplex *psi_3k, double *c2_k, double *k2, int Mo, double dt, double a, double b, int gpu_i, int Mx, int num_gpus ) {

	int xgrid_idx = threadIdx.x + blockIdx.x*blockDim.x;
	int xgrid_stride = gridDim.x*blockDim.x; 

	int ygrid_idx = threadIdx.y + blockIdx.y*blockDim.y;
	int ygrid_stride = gridDim.y*blockDim.y;
	
	double d2muk_dx2_real, dpsik_dt_real, d2muk_dx2_cplx, dpsik_dt_cplx;
	
	for( int i = xgrid_idx; i < Mx; i+= xgrid_stride ) {
		for( int j = ygrid_idx; j < Mx/num_gpus; j+= ygrid_stride ) {
			
			d2muk_dx2_real = -Mo*k2[i + j*Mx]*( psi_k[i + j*Mx].x - a*psi_2k[i + j*Mx].x + b*psi_3k[i + j*Mx].x - c2_k[i + j*Mx]*psi_k[i + j*Mx].x );
			dpsik_dt_real = (psi_kp1[i + j*Mx].x - psi_k[i + j*Mx].x)/dt;
			
			d2muk_dx2_cplx = -Mo*k2[i + j*Mx]*( psi_k[i + j*Mx].y - a*psi_2k[i + j*Mx].y + b*psi_3k[i + j*Mx].y - c2_k[i + j*Mx]*psi_k[i + j*Mx].y );
                        dpsik_dt_cplx = (psi_kp1[i + j*Mx].y - psi_k[i + j*Mx].y)/dt;
			
			res_k[i + j*Mx].x = dpsik_dt_real - d2muk_dx2_real;
			res_k[i + j*Mx].y = dpsik_dt_cplx - d2muk_dx2_cplx;			 
	
			//printf( "Row %d Column %d d2muk_dx2real: ( %0.16f , %0.16f )\n", j + (Mx/num_gpus)*gpu_i, i, dpsik_dt_real, dpsik_dt_cplx );
			//printf( "Row %d Column %d Residual: ( %0.16f , %0.16f )\n", j + (Mx/num_gpus)*gpu_i, i, res_k[i + j*Mx].x, res_k[i + j*Mx].y );
		} 
	}	

}

__global__ void copy_struct( cufftDoubleComplex *res_st, double *res_arr, int gpu_i, int Mx, int num_gpus ) {
	int xgrid_idx = threadIdx.x + blockIdx.x*blockDim.x;
	int xgrid_stride = gridDim.x*blockDim.x;
	int ygrid_idx = threadIdx.y + blockIdx.y*blockDim.y;
	int ygrid_stride = gridDim.y*blockDim.y;

	for( int i = xgrid_idx; i < Mx; i+= xgrid_stride ) {
		for( int j = ygrid_idx; j < Mx/num_gpus; j+= ygrid_stride ) {
			res_arr[i + j*Mx] = res_st[i + j*Mx].x;
			//printf( "Row %d Column %d Residual: ( %0.5e , %0.5e )\n", j + (Mx/num_gpus)*gpu_i, i, res_arr[i + j*Mx], res_st[i + j*Mx].x );
		}
	}
}


