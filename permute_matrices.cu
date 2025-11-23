#include "globals.cuh"
//#include "helpers_header.cuh"

void random_input( Complex *mat1 ) {
	srand( 1000 );
	
	for(int i = 0; i < My; i++) { 
		for(int j = 0; j < Mx; j++) {
			mat1[ i*Mx + j ].x = (double)rand()/RAND_MAX;
			mat1[ i*Mx + j ].y = (double)rand()/RAND_MAX;

			//mat2[ i*Mx + j ].x = mat1[ i*Mx + j ].x;
                        //mat2[ i*Mx + j ].y = mat1[ i*Mx + j ].y;
		}
	}
}


void permute_matrices(cufftHandle planComplex, Complex *h_mat1, Complex *h_mat2 ) {
	// h_mat1 is the original unpermuted matrix fue to cufft
	// h_mat2 is the permuted matrix due to cufft
	
	cudaSetDevice(0);	
	random_input( h_mat1 );
	
	cudaLibXtDesc *d_mat1, *d_mat1_out; 
	cudaLibXtDesc *d_mat2;
	cufftResult result;

	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_mat1, CUFFT_XT_FORMAT_INPLACE );
        //check_cufft_result(result, "cufftXtMalloc (d_psiC2C)");
	
	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_mat1_out, CUFFT_XT_FORMAT_INPLACE );
        //check_cufft_result(result, "cufftXtMalloc (d_psiC2C)");

	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_mat2, CUFFT_XT_FORMAT_INPLACE );
        //check_cufft_result(result, "cufftXtMalloc (d_psiC2C)");

	result = cufftXtMemcpy(planComplex, d_mat1, h_mat1, CUFFT_COPY_HOST_TO_DEVICE);
        result = cufftXtMemcpy(planComplex, d_mat2, h_mat1, CUFFT_COPY_HOST_TO_DEVICE);
	//check_cufft_result(result, "cufftXtMemcpy (H2D)");
	cudaDeviceSynchronize();

	result = cufftXtExecDescriptor( planComplex, d_mat1, d_mat1, CUFFT_FORWARD );
	result = cufftXtExecDescriptor( planComplex, d_mat2, d_mat2, CUFFT_FORWARD );

	result = cufftXtMemcpy(planComplex, d_mat1_out, d_mat1, CUFFT_COPY_DEVICE_TO_DEVICE);

	
	//result = cufftXtMemcpy( planComplex, h_mat1, d_mat1_out, CUFFT_COPY_DEVICE_TO_HOST );
        //result = cufftXtMemcpy( planComplex, h_mat2, d_mat2, CUFFT_COPY_DEVICE_TO_HOST );
	copy2cpu( d_mat1_out, h_mat1 );
	copy2cpu( d_mat2, h_mat2 );	
	
	cudaDeviceSynchronize();

	// very expensive for loop, but saves multiple memcpy calls every iteration in time loop
	for(int i = 0; i < Mx; i++){
		for(int j = 0; j < My; j++){
			int idx_1 = i*Mx + j;

			for(int k = 0; k < Mx; k++){
				for(int l = 0; l < My; l++){
					int idx_2 = k*Mx + l;
					
					if( h_mat1[idx_1].x == h_mat2[idx_2].x &&  h_mat1[idx_1].y == h_mat2[idx_2].y ) {
					
						c2_k_hostP[ idx_2 ] = c2_k_host[ idx_1 ];
						k2_hostP[ idx_2 ] = k2_host[ idx_1 ];

					}
				}
			}
		}
	}
	
 	/*	
	for(int i = 0; i < Mx; i++){
                for(int j = 0; j < My; j++){
			//printf("row %d col %d h_mat1 = (%0.5f, %0.5f) h_mat2 = (%0.5f, %0.5f) \n", i,j, h_mat1[ i*Mx+j ].x, h_mat1[ i*Mx+j ].y, h_mat2[ i*Mx+j ].x, h_mat2[ i*Mx+j ].y  );
			printf("row %d col %d k2 = (%0.5f) k2P (%0.5f) \n", i, j, k2_host[i*Mx + j] , k2_hostP[i*Mx+j] );
		}
	}
	*/

	cufftXtFree(d_mat1);
        cufftXtFree(d_mat1_out);
        cufftXtFree(d_mat2);



}




