#include "globals.cuh"

void euler_step_wrapper( int n, cudaLibXtDesc *d_psikp1, cudaLibXtDesc *d_psik, cudaLibXtDesc *d_psi2k, cudaLibXtDesc *d_psi3k, double **c2_k, double **k2 ) {
	int device; 

	for(int gpu_i=0; gpu_i<num_gpus; gpu_i++){
		dim3 dimGrid( grid_size.x, grid_size.y, grid_size.z );
                dim3 dimBlock( block_size.x, block_size.y, block_size.z );
		
		device = d_psik -> descriptor -> GPUs[gpu_i];
		cudaSetDevice(device);

		euler_step<<<dimGrid,dimBlock>>>(n, a, b, Mo,(cufftDoubleComplex *)d_psikp1->descriptor->data[gpu_i], (cufftDoubleComplex *)d_psik->descriptor->data[gpu_i], (cufftDoubleComplex *)d_psi2k->descriptor->data[gpu_i], (cufftDoubleComplex *)d_psi3k->descriptor->data[gpu_i], c2_k[gpu_i], k2[gpu_i], dt, eps, gpu_i, Mx, num_gpus ); 
		//cudaDeviceSynchronize();
	}
	// printf("eps = %0.5f\n", eps);



	for (int gpu_i = 0; gpu_i < num_gpus; gpu_i++) {                                                              
		device = d_psik->descriptor->GPUs[gpu_i];
                cudaSetDevice(device);
                cudaDeviceSynchronize(); // Wait for memory transfer to complete before accessing data                
		//printf("Euler Step Printing iteration: %d GPU: %d \n", n, gpu_i);
                // Check if kernel execution generated and error
                getLastCudaError("Kernel execution failed [ apply_power ]");
        }

}



__global__ void euler_step( int n, double a, double b, double Mo, cufftDoubleComplex *psi_kp1, cufftDoubleComplex *psi_k, cufftDoubleComplex *psi_2k, cufftDoubleComplex *psi_3k, double *c2_k, double *k2, double dt, double eps, int gpu_i, int Mx, int num_gpus ) {

	double tmp_num_real, tmp_num_cplx, tmp_den; 
	
	int xgrid_idx = threadIdx.x + blockDim.x*blockIdx.x; // looping across threads and blocks in x
	int xgrid_stride = gridDim.x*blockDim.x; // striding over different grids in y

	int ygrid_idx = threadIdx.y + blockDim.y*blockIdx.y; // looping across threads and blocks in x
	int ygrid_stride = gridDim.y*blockDim.y; // striding over different grids in y
	
	int rows = Mx/num_gpus;
	int cols = Mx; 

	for( int i = xgrid_idx; i < cols; i += xgrid_stride ) { // i - columns, contiguous in memory and the fastest changing dimension
		for( int j = ygrid_idx; j < rows; j += ygrid_stride ) { // j - rows
			//tmp_num_real = 1.0f; tmp_num_cplx = 1.0f;
			int idx = i + j*Mx;
			tmp_num_real = psi_k[ idx ].x + a*k2[ idx ]*Mo*dt*psi_2k[ idx ].x - b*k2[ idx ]*dt*Mo*psi_3k[ idx ].x;
			tmp_num_cplx =  psi_k[ idx ].y + a*k2[ idx ]*Mo*dt*psi_2k[ idx ].y - b*k2[ idx ]*dt*Mo*psi_3k[ idx ].y;
			
			//tmp_den = (1 + k2[ i + j*Mx ]*dt*Mo*(1-eps) - k2[ i + j*Mx ]*Mo*dt*c2_k[ i + j*Mx ]);
			tmp_den = (1.0 + k2[ idx ]*dt*Mo*(1.0-eps) - k2[ idx ]*Mo*dt*c2_k[ idx ]);

			//psi_k[ i + j*Mx ].x = tmp_den;
			psi_kp1[ idx ].x = tmp_num_real/tmp_den;
			
			psi_kp1[ idx ].y = tmp_num_cplx/tmp_den;
			// printf("GPU id %d row %d col %d: %0.5f \n",gpu_i, j, i, psi_k[ idx ].x );
		}
	}
}


