#define MAIN
#include "globals.cuh"
#include "helpers_header.cuh"

int main( int argc , char** argv ) {
	
	auto start = chrono::high_resolution_clock::now();

        int i, j, n;	

	Timer timer;

	// Obtaining GPU device(s) info <start> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	cudaGetDeviceCount(&num_gpus);
	printf("Number of GPUs on NODE:: %d \n", num_gpus);
	size_t *threads_per_block = (size_t*) malloc(sizeof(size_t) * num_gpus);
	size_t *number_of_blocks = (size_t*) malloc(sizeof(size_t) * num_gpus);
	int *which_gpus = (int*) malloc(sizeof(int) * num_gpus);
	size_t *worksize;
  	worksize = (size_t *)malloc(sizeof(size_t) * num_gpus);	
	root_GPU = 0; // Sets root gpu for reducing elements, should not need to be modified

	for (int gpu_i=0; gpu_i<num_gpus; gpu_i++){
		int computeCapabilityMajor;
                int computeCapabilityMinor;
                int multiProcessorCount;
                int warpSize;
                int maxBlocksSM;
                int maxThreadsBlock;
		double maxMemory;
		cudaDeviceProp device_props;

		cudaSetDevice(gpu_i);
		which_gpus[gpu_i] = gpu_i; // Contaings array for the GPU_ids 
		checkCudaErrors(cudaGetDeviceProperties(&device_props, gpu_i));

		computeCapabilityMajor = device_props.major;
                computeCapabilityMinor = device_props.minor;
                multiProcessorCount = device_props.multiProcessorCount;
                warpSize = device_props.warpSize;
                maxBlocksSM = device_props.maxBlocksPerMultiProcessor;
                maxThreadsBlock = device_props.maxThreadsPerBlock;
		maxMemory = double( device_props.totalGlobalMem )/1E9;

                printf( "GPU Device: %d \"%s\"\n"
			"Max Memory: %0.1f GB\n"
                        "Number of SMs: %d\n"
                        "Compute Capability Major: %d\n"
                        "Compute Capability Minor: %d\n"
                        "Warp Size: %d\n"
                        "Max Blocks per SM: %d\n"
                        "Max Threads per Block: %d\n"
                        "Max Blocks Total: %d\n\n",
                        gpu_i, maxMemory, device_props.name, multiProcessorCount,
                        computeCapabilityMajor, computeCapabilityMinor,
                        warpSize, maxBlocksSM, maxThreadsBlock, multiProcessorCount*maxBlocksSM);

		threads_per_block[gpu_i] = maxThreadsBlock;
        	number_of_blocks[gpu_i] = maxBlocksSM*multiProcessorCount;
	}
	
	grid_size.x = 32; // Number of blocks in x grid
	grid_size.y = 32; // Number of blocks in y grid
	grid_size.z = 1;  // Number of blocks in z grid
	block_size.x = 32; // Number of threads in x blocks
	block_size.y = 32; // Number of threads in y blocks
	block_size.z = 1;  // Number of threads in z blocks
	
	dim3 dimGrid( grid_size.x, grid_size.y, grid_size.z );                                                                                                      dim3 dimBlock( block_size.x, block_size.y, block_size.z ); 

	// Obtaining GPU device(s) info <end> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	// Initializing NCCL Communicators <start> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	ncclComm_t comms[num_gpus];
	ncclCommInitAll(comms, num_gpus, nullptr);

	// Initializing NCCL Communicators <end> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	// Reading Inputs and allocating memory on the CPUs <start> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	read_input( ) ; // >>>>>>>>>> CPU

	initialize( ) ; // >>>>>>>>>> CPU
	
	chunk_size_real = sdiv(M2,num_gpus);
        chunk_size_cplx = sdiv(M2_k,num_gpus);
	sr_threads = TSIZE; 
	sr_blocks = min((chunk_size_real + sr_threads - 1) / sr_threads, 1024);	
	// Reading Inputs and allocating memory on the CPUs <end> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	// Creating cuFFT plans on the MGPU setup <start> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	//TODO 
	// 1. Convert all doubles to floats - c 3/14/2025
	// 2. Convert to complex to complex transform and copy the Poisson tutorial - c 3/14/2025
	// 3. Check if convolution is correct by comparing to MATLAB code - c 3/15/2025
	// 4. Complete Free Energy Calculation code - c 3/18/2025
	// 5. Correctly permute c2_k and k2 
	// 6. Complete time loop 
	// 7. Change all grid-stride loops to the correct format - c 3/19/2025 
	// 8. Optimize the convolution and euler steps by FT k2 and c2k -> device2device copy -> iFFT for permuted order
	// 9. Complete the write .vtk and .txt code, copy compute overlap or write directly from gpu

	cufftHandle planComplex;
	cufftCreate(&planComplex);

    	cufftResult result = cufftXtSetGPUs(planComplex, num_gpus, which_gpus);
    	check_cufft_result(result, "cufftXtSetGpus (C2C)");
	
	result = cufftMakePlan2d(planComplex, Mx, My, CUFFT_Z2Z, worksize);
        check_cufft_result(result, "cufftMakePlanXd (C2C)");

	// Creating cuFFT plans on the MGPU setup <end> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	// Obtaining cufft shuffling to apply to k-space operators and kernels <start> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// permute_matrices( planComplex, test_arr1, test_arr2 );
        //exit(EXIT_SUCCESS); // Uncomment to debug
	// Obtaining cufft shuffling to apply to k-space operators and kernels <end> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	// Allocating memory on the GPUs <start> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	// For Device, be selective which memory is allocated to GPU
	cudaSetDevice(0);	
	
	double *c2_k[num_gpus], *k2[num_gpus], *ff[num_gpus];
	double *d_gpu_reduced_array, *d_reduced_array;
	
	// Structs for housing the multi-gpu FFT data
	// *d_psiN, d_convl for inputs 
	// *<>_out varibles for natural order
        // *<>_np1 variables for next time step 	

	cudaLibXtDesc *d_psiC2C,  *d_psiC2C_out; // equivalent to f_in and f_out, respectively
	cudaLibXtDesc *d_psiC2C_np1,  *d_psiC2C_np1_out;
	cudaLibXtDesc *d_convl, *d_convl_out; 
	cudaLibXtDesc *d_psi2C2C, *d_psi2C2C_out; 
       	cudaLibXtDesc *d_psi3C2C, *d_psi3C2C_out;	
	cudaLibXtDesc *d_psi4C2C;
	cudaLibXtDesc *res_k, *res_out; 	

	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_psiC2C, CUFFT_XT_FORMAT_INPLACE );
  	check_cufft_result(result, "cufftXtMalloc (d_psiC2C)");
	
	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_psiC2C_out, CUFFT_XT_FORMAT_INPLACE );
        check_cufft_result(result, "cufftXtMalloc (d_psiC2C_out)");

	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_psiC2C_np1, CUFFT_XT_FORMAT_INPLACE );
        check_cufft_result(result, "cufftXtMalloc (d_psiC2C_np1)");

	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_psiC2C_np1_out, CUFFT_XT_FORMAT_INPLACE );
        check_cufft_result(result, "cufftXtMalloc (d_psiC2C_np1)");

	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_convl, CUFFT_XT_FORMAT_INPLACE );
        check_cufft_result(result, "cufftXtMalloc (d_convl)");

	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_convl_out, CUFFT_XT_FORMAT_INPLACE );
        check_cufft_result(result, "cufftXtMalloc (d_convl_out)");

	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_psi2C2C, CUFFT_XT_FORMAT_INPLACE );
        check_cufft_result(result, "cufftXtMalloc (d_psi2C2C)");
	
        result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_psi2C2C_out, CUFFT_XT_FORMAT_INPLACE );
        check_cufft_result(result, "cufftXtMalloc (d_psi2C2C_out)");

	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_psi3C2C, CUFFT_XT_FORMAT_INPLACE );
        check_cufft_result(result, "cufftXtMalloc (d_psi3C2C)");

        result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_psi3C2C_out, CUFFT_XT_FORMAT_INPLACE );
        check_cufft_result(result, "cufftXtMalloc (d_psi3C2C_out)");
	
	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&d_psi4C2C, CUFFT_XT_FORMAT_INPLACE );
        check_cufft_result(result, "cufftXtMalloc (d_psi3C2C)");
	
	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&res_k, CUFFT_XT_FORMAT_INPLACE );
        check_cufft_result(result, "cufftXtMalloc (d_psi3C2C)");

	result = cufftXtMalloc(planComplex, (cudaLibXtDesc **)&res_out, CUFFT_XT_FORMAT_INPLACE );
        check_cufft_result(result, "cufftXtMalloc (d_psi3C2C)");

	mem_use_gpu = 0.;
	
	cudaMalloc((void**)&F, N_t_steps * sizeof( double )); // GPU	
	mem_use_gpu += N_t_steps * sizeof( double );
	cudaMalloc((void**)&d_gpu_reduced_array, chunk_size_real*sizeof( double )); //GPU
	cudaMalloc((void**)&d_reduced_array, chunk_size_real*sizeof( double )); //GPU
	mem_use_gpu += 2*chunk_size_real*sizeof( double );

	for (int gpu_i = 0; gpu_i < num_gpus; gpu_i++) {
		cudaSetDevice(gpu_i);

		lower = chunk_size_real*gpu_i;
		upper = min(lower+chunk_size_real,M2);		
		width = upper-lower;

		cudaMalloc((void **)&c2_k[gpu_i], width * sizeof( double ) ); // GPU
        	cudaMalloc((void **)&k2[gpu_i], width * sizeof( double ) ); // GPU		
		cudaMalloc((void**)&ff[gpu_i], width * sizeof( double )); // GPU

		mem_use_gpu += 13 * width * sizeof( cufftDoubleComplex ); // for cufftXtMalloc
		mem_use_gpu += 3 * width * sizeof( double ); // for cudaMalloc
		
		printf("Memory allocated GPU %d: %lf GB\n", gpu_i, mem_use_gpu/1E9) ;  fflush( stdout ) ;	

		mem_use_gpu = 0.;	
	}	

	// Allocating memory on the GPUs <end> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	// Memory transfer from CPU to GPU (major bottleneck) <start> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	for(int gpu_i=0;gpu_i<num_gpus;gpu_i++) {
		
		cudaSetDevice(which_gpus[gpu_i]);
		lower = chunk_size_real*gpu_i;
                upper = min(lower+chunk_size_real,M2);
                width = upper-lower;

		//DCF and Wave function magnitude initialized on CPU -> xfer GPU
		cudaMemcpy(c2_k[gpu_i], c2_k_host+lower, sizeof( double )*width , cudaMemcpyHostToDevice);
		cudaMemcpy(k2[gpu_i], k2_host+lower, sizeof( double )*width , cudaMemcpyHostToDevice);
	}
		
	
	result = cufftXtMemcpy(planComplex, d_psiC2C, psi_host, CUFFT_COPY_HOST_TO_DEVICE);
	check_cufft_result(result, "cufftXtMemcpy (H2D)");
        cudaDeviceSynchronize();	

	// Memory transfer from GPU to CPU (major bottleneck) <end> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
	

	// Applying power to order-parameters before FFT <start> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	power_function_wrapper(d_psiC2C, d_psi2C2C, 2 );
	power_function_wrapper(d_psiC2C, d_psi3C2C, 3 );
	power_function_wrapper(d_psiC2C, d_psi4C2C, 4 );
	// Applying power to order-parameters before FFT <end> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	// Iteration Zero <Start> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
	printf("Subformat (d_psiC2C) after H2D memory allocation: %d\n", d_psiC2C->subFormat);
		
	/*
	copy2cpu( d_psiC2C, psi_host_out );
	for (i=0;i<Mx;i++){
        	for (j=0;j<My;j++){
                	printf( "Array[%d,%d] = (%0.5f, %0.5f) \n",i, j, psi_host_out[i*Mx + j].x, psi_host_out[i*Mx + j].y );
                }
        }
	*/
	

	printf("Forward 2d FFT on multiple GPUs\n");
	timer.start();
	result = cufftXtExecDescriptor( planComplex, d_psiC2C, d_psiC2C, CUFFT_FORWARD );
  	timer.stop("FFT time on GPU");
	check_cufft_result(result, "cufftXtExec Forward (psi)");
	printf("Iteration 0: Transformed psi_0.\n");		

	printf("Subformat (d_psiC2C) after Forward Transform: %d\n", d_psiC2C->subFormat);

	result = cufftXtMemcpy( planComplex, d_psiC2C_out, d_psiC2C, CUFFT_COPY_DEVICE_TO_DEVICE ); // recovering natural order of order-parameter
	
	
	printf("Subformat (d_psiC2C) after D2D memory allocation: %d\n", d_psiC2C->subFormat);
	printf("Subformat (d_psiC2C_out) after D2D memory allocation: %d\n", d_psiC2C_out->subFormat);
	
	// Free Energy Calculation <start> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	convolution_wrapper( d_psiC2C_out, c2_k, d_convl );	
	
	result = cufftXtExecDescriptor( planComplex, d_convl, d_convl, CUFFT_INVERSE ); // permuted order
        check_cufft_result(result, "cufftXtExec Inverse (psi_k*c2_k)");
	
	scaling_step_wrapper( d_convl );	
	
	result = cufftXtMemcpy( planComplex, d_convl_out, d_convl, CUFFT_COPY_DEVICE_TO_DEVICE ); // recovering natural order of convolution
	d_convl -> subFormat = 2;	

	result = cufftXtExecDescriptor( planComplex, d_psiC2C, d_psiC2C, CUFFT_INVERSE ); // returns to natural order
        check_cufft_result(result, "cufftXtExec Inverse (psi)");
	scaling_step_wrapper( d_psiC2C );
	
	printf("Subformat (d_psiC2C) after Inverse Transform: %d\n", d_psiC2C->subFormat);	

	//exit(EXIT_SUCCESS);	
	
	FE_calc_wrapper( 0, d_psiC2C, d_psi2C2C, d_psi3C2C, d_psi4C2C, d_convl_out, ff, d_gpu_reduced_array, d_reduced_array, comms );	
	// Free Energy Calculation <end> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// d_psiC2C - c d_psi2C2C - c d_psi3C2C - c d_psi4C2C - c  d_convl_out	

	// Writing Iteration Zero to .vtk file <start> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	writeVTKFile(psi_host, Mx, My, Lx, Ly, 0);
	//exit(EXIT_SUCCESS);	
	// Writing Iteration Zero to .vtk file <end> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	// Iteration Zero <end> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	
	// writeVTKFile(psi, Mx, My, Lx, Ly, 0);
	cudaDeviceSynchronize();

	// Iteration One <start> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	n = 1;
	timer.start();
	for( n = 0; n < N_t_steps; n++) {
		// Uncomment Loop Comments for Debugging

		result = cufftXtExecDescriptor( planComplex, d_psi2C2C, d_psi2C2C, CUFFT_FORWARD );
		//check_cufft_result(result, "Time loop cufftXtExec Forward (d_psi2C2C)");		
		
		//printf("Subformat (d_psi2C2C) after Forward Transform: %d\n", d_psi2C2C->subFormat);
		
		result = cufftXtExecDescriptor( planComplex, d_psi3C2C, d_psi3C2C, CUFFT_FORWARD );
                //check_cufft_result(result, "Time loop cufftXtExec Forward (d_psi3C2C)");

		result = cufftXtMemcpy(planComplex, d_psi2C2C_out, d_psi2C2C, CUFFT_COPY_DEVICE_TO_DEVICE); // Recovers natural order in k-space
		//check_cufft_result(result, "Time loop Memcpy D2D (d_psi2C2C)"); // d_psi2C2C needs to be reset 

		result = cufftXtMemcpy(planComplex, d_psi3C2C_out, d_psi3C2C, CUFFT_COPY_DEVICE_TO_DEVICE); // Recovers natural order in k-space	
		//check_cufft_result(result, "Time loop Memcpy D2D (d_psi3C2C)"); // d_psi3C2C needs to be reset 

		euler_step_wrapper( n, d_psiC2C_np1, d_psiC2C_out, d_psi2C2C_out, d_psi3C2C_out, c2_k, k2 );	
		
			
		if( n % t_sample == 0 ) {
			SSE_calc_wrapper( n, planComplex, res_out, res_k, d_psiC2C_np1, d_psiC2C_out, d_psi2C2C_out, d_psi3C2C_out, c2_k, k2, ff, d_gpu_reduced_array, d_reduced_array, comms );
		}	
	

		//printf("Subformat (d_psiC2C_np1) after Euler Step: %d\n", d_psiC2C_np1->subFormat);
		
		/*	
		printf("\n Psi_k Euler step %d:\n", n);
		copy2cpu( d_psiC2C_np1, psi_host_out );
        	for (i=0;i<Mx;i++){
                	for (j=0;j<My;j++){
                        	printf( "Array[%d,%d] = (%0.16f, %0.16f) \n",i, j, psi_host_out[i*Mx + j].x, psi_host_out[i*Mx + j].y );
                	}
        	}
		*/
		
		result = cufftXtExecDescriptor( planComplex, d_psiC2C_np1, d_psiC2C_np1, CUFFT_INVERSE );
		//check_cufft_result(result, "Time loop cufftXtExec Inverse (d_psiC2C_np1)");
				

		//printf("Subformat (d_psiC2C_np1) after Inverse Transform: %d\n", d_psiC2C_np1->subFormat);

		scaling_step_wrapper( d_psiC2C_np1 );	
		result = cufftXtMemcpy(planComplex, d_psiC2C_np1_out, d_psiC2C_np1, CUFFT_COPY_DEVICE_TO_DEVICE); // Recovers natural order in real space
		//check_cufft_result(result, "Time loop Memcpy D2D (d_psiC2C_np1)"); // d_psiC2C_np1 needs to be reset

		//printf("Subformat (d_psiC2C_np1_out) after D2D memory allocation: %d\n", d_psiC2C_np1_out->subFormat);	
		
		/*
		printf("\n Psi_r Euler step %d:\n", n);		
		copy2cpu( d_psiC2C_np1_out, psi_host_out );
        	for (i=0;i<Mx;i++){
                	for (j=0;j<My;j++){
                        	printf( "Array[%d,%d] = (%0.5f, %0.5f) \n",i, j, psi_host_out[i*Mx + j].x, psi_host_out[i*Mx + j].y );
                	}
        	}
		*/		

		/*
		exit(EXIT_SUCCESS);
		*/
		
		copy_matrix_wrapper(d_psiC2C_np1_out, d_psiC2C);
		
		power_function_wrapper(d_psiC2C, d_psi2C2C, 2 );
        	power_function_wrapper(d_psiC2C, d_psi3C2C, 3 );				
		

		/*
		if (n == 1000){
		
		copy2cpu( d_psiC2C, psi_host_out );
		//cufftXtMemcpy(planComplex, psi_host_out, d_psiC2C, CUFFT_COPY_DEVICE_TO_HOST);

		printf("\n Euler step %d:\n", n);
		        	
		for (i=0;i<Mx;i++){
        		for (j=0;j<My;j++){
        			printf( "Array[%d,%d] = (%0.10f, %0.10f) \n",i, j, psi_host_out[i*Mx + j].x, psi_host_out[i*Mx + j].y );
                	}
        	
		}
		exit(EXIT_SUCCESS);
		} else if  (n == 999){
		copy2cpu( d_psiC2C, psi_host_out );
                //cufftXtMemcpy(planComplex, psi_host_out, d_psiC2C, CUFFT_COPY_DEVICE_TO_HOST);

                printf("\n Euler step %d:\n", n);

                for (i=0;i<Mx;i++){
                        for (j=0;j<My;j++){
                                printf( "Array[%d,%d] = (%0.10f, %0.10f) \n",i, j, psi_host_out[i*Mx + j].x, psi_host_out[i*Mx + j].y );
                        }

                }
		}
		*/

		result = cufftXtExecDescriptor( planComplex, d_psiC2C, d_psiC2C, CUFFT_FORWARD );
		//check_cufft_result(result, "Time loop cufftXtExec Forward (d_psiC2C)");
		
		result = cufftXtMemcpy(planComplex, d_psiC2C_out, d_psiC2C, CUFFT_COPY_DEVICE_TO_DEVICE); // Recovers natural order in k-space
		//check_cufft_result(result, "Time loop Memcpy D2D (d_psiC2C)"); //  d_psiC2C needs to be reset

		
                if ( (n+1) % t_sample == 0) {
			
                	convolution_wrapper( d_psiC2C_out, c2_k, d_convl );

        		result = cufftXtExecDescriptor( planComplex, d_convl, d_convl, CUFFT_INVERSE );
        		//check_cufft_result(result, "cufftXtExec Inverse (psi_k*c2_k)");

        		scaling_step_wrapper( d_convl );
        		result = cufftXtMemcpy(planComplex, d_convl_out, d_convl, CUFFT_COPY_DEVICE_TO_DEVICE);
        		//check_cufft_result(result, "cufftXtExec cufftXtMemcpy (d_convl)");
        		d_convl -> subFormat = 2;
        	

        		power_function_wrapper(d_psiC2C_np1_out, d_psi4C2C, 4 );
        		FE_calc_wrapper( n+1, d_psiC2C_np1_out, d_psi2C2C, d_psi3C2C, d_psi4C2C, d_convl_out, ff, d_gpu_reduced_array, d_reduced_array, comms );	
		}
                

		//cufftXtSetSubformatDefault(planComplex, d_psi2C2C -> subFormat );
		d_psiC2C -> subFormat = 2;
		d_psi2C2C -> subFormat = 2;
		d_psi3C2C -> subFormat = 2;
		d_psiC2C_np1 -> subFormat = 2;
	}

	timer.stop("Full iteration time.");
	
	convolution_wrapper( d_psiC2C_out, c2_k, d_convl );

        result = cufftXtExecDescriptor( planComplex, d_convl, d_convl, CUFFT_INVERSE );
        //check_cufft_result(result, "cufftXtExec Inverse (psi_k*c2_k)");

        scaling_step_wrapper( d_convl );
        result = cufftXtMemcpy(planComplex, d_convl_out, d_convl, CUFFT_COPY_DEVICE_TO_DEVICE);
        //check_cufft_result(result, "cufftXtExec cufftXtMemcpy (d_convl)");
	d_convl -> subFormat = 2;

	power_function_wrapper(d_psiC2C_np1_out, d_psi4C2C, 4 );
	FE_calc_wrapper( n, d_psiC2C_np1_out, d_psi2C2C, d_psi3C2C, d_psi4C2C, d_convl_out, ff, d_gpu_reduced_array, d_reduced_array, comms );
	
	
	copy2cpu( d_psiC2C_np1_out, psi_host_out );	
	
	writeVTKFile(psi_host_out, Mx, My, Lx, Ly, n);
	// Iteration One <end> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	
	// List of variables to free memory: 
	/*
	Host:
		F_host
		c2_k_host
		c2_k_hostP
		k2_host
		k2_hostP
		k_arr
		kx
		ky
		test_arr1
		test_arr2
		X
		x
		Y
		y
		psi_host - Pinned 
		psu_host_out
		reduced_array_host
		SUB1_host - Pinned

	Device:
	non fft:
		F - root gpu
		c2_k
		k2
		ff
		d_gpu_reduced_array - root gpu
		d_reduced_array	- root gpu
	fft: 
		cudaLibXtDesc *d_psiC2C,  *d_psiC2C_out;
        	cudaLibXtDesc *d_psiC2C_np1,  *d_psiC2C_np1_out;
        	cudaLibXtDesc *d_convl, *d_convl_out;
        	cudaLibXtDesc *d_psi2C2C, *d_psi2C2C_out;
        	cudaLibXtDesc *d_psi3C2C, *d_psi3C2C_out;
        	cudaLibXtDesc *d_psi4C2C;
        	cudaLibXtDesc *res_k, *res_out;

	*/	
	
        free(c2_k_host); free(c2_k_hostP);
        free(k2_host); free(k2_hostP);
        free(k_arr);
        free(kx); free(ky);
        free(test_arr1); free(test_arr2);
        free(X); free(x); free(Y); free(y);
        free(psi_host_out);
        free(reduced_array_host);        
	cudaFreeHost(psi_host); cudaFreeHost(SUB1_host);
	cudaFreeHost(F_host);	

	cudaFree(F);
	cudaFree(d_gpu_reduced_array);
       	cudaFree(d_reduced_array);
	
	for(int gpu_i=0; gpu_i < num_gpus; gpu_i++) {
		cudaFree(c2_k[gpu_i]);
                cudaFree(k2[gpu_i]);
                cudaFree(ff[gpu_i]);
	}

	result = cufftXtFree(d_psiC2C);  	cufftXtFree(d_psiC2C_out);
        cufftXtFree(d_psiC2C_np1); 	cufftXtFree(d_psiC2C_np1_out);
        cufftXtFree(d_convl); 		cufftXtFree(d_convl_out);
        cufftXtFree(d_psi2C2C); 	cufftXtFree(d_psi2C2C_out);
        cufftXtFree(d_psi3C2C); 	cufftXtFree(d_psi3C2C_out);
        cufftXtFree(d_psi4C2C);
        cufftXtFree(res_k); 		cufftXtFree(res_out);

	check_cufft_result(result, "\ncufftXtFree (d_psiC2C)");

	return 0;
}

// Uncomment and use code below for debugging: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// cout << "Debugging complete..." << endl;
// exit(1);
/*
copy2cpu( d_psiC2C_np1_out, psi_host_out );
for (i=0;i<Mx;i++){
	for (j=0;j<My;j++){
		printf( "Array[%d,%d] = (%0.5f, %0.5f) \n",i, j, psi_host_out[i*Mx + j].x, psi_host_out[i*Mx + j].y );
	}
}
exit(EXIT_SUCCESS);
*/
//printf("Subformat (d_psiC2C) after Resetting subFormat: %d\n", d_psiC2C->subFormat);                                          //printf("Subformat (d_psi2C2C) after Resetting subFormat: %d\n", d_psi2C2C->subFormat);                                        //printf("Subformat (d_psi3C2C) after Resetting subFormat: %d\n", d_psi3C2C->subFormat);                                        //printf("Subformat (d_psiC2C_np1) after Resetting subFormat: %d\n", d_psiC2C_np1->subFormat);
//printf("here\n");
//printf("\nEuler step %d:\n", n);
