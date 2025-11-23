#include "globals.cuh"
void allocate_cpu( void );
void fft_init( void );


void initialize() {
	int i, j ;

	// DCF parameters <start>
	k_10 = 2.0*PI; 
	k_11 = 2.0*PI*sqrt(2.0);
	rho_10 = 1.0;
	rho_11 = 1.0/sqrt(2.0);
	B_10 = 4.0;
	B_11 = 4.0;
       	al_10 = 1.0;
	al_11 = 1.0; 
	a_tri = 2.0/sqrt(3.0);
	a_sq = 1.0;
	// DCF parameters <end>
	v = 0.0;

	if (iconfig == 0) {
		// Triangular Phase
		Lx = Lx*a_tri; Ly = Ly*a_tri;
	} else if (iconfig == 1) {
		// Square Phase
                Lx = Lx*a_sq; Ly = Ly*a_sq;
	} else if (iconfig == 2) {
		// Liquid Phase
	} else if (iconfig == 10) {
		obtain_box_size( );	
	}

	cout << "Lx*a = " << Lx << " Ly*a = " << Ly << endl;

	I = complex<double>( 0.0 , 1.0 ) ;

	// Grid Parameters
        M2 = Mx*My;
        M_k = (My/2 + 1);
        M2_k = Mx*M_k;
	Mx_seg = Mx - 1; // Number of x segments
	My_seg = My - 1; // Number of y segments
	dxy = Lx/Mx_seg; // Gridspacing
	cout << "dxy = " << dxy << endl;
	
	double dt_crit = 2/(Mx-2);
       	cout << "dt_crit = " << dt_crit << endl;	
	
	/*
	if (dt >= dt_crit) {
		printf("Timestep is too large, will become unstable, choose dt < %f, exiting...\n",dt_crit) ;
		exit(EXIT_SUCCESS); // exits program if time step is too large
	} else {
		printf("Timestep is okay, proceeding...\n") ;
	}
	*/

	N_t_steps = t/dt + 1;
	printf( "Num Steps: %lf \n", N_t_steps ) ;

	ff_factor = ( double(Lx*Ly))/(double(Mx_seg*My_seg));
	printf("ff_factor = %0.6f\n", ff_factor);	
	mem_use = 0.;

	//fft_init();
	//printf("FFTW-MPI Initialized\n") ;

	allocate_cpu( );
        
	linspace(0,Lx,Mx,x) ;
        linspace(0,Ly,My,y) ;
        FFT_freq(Mx, Lx, kx);
        FFT_freq(My, Ly, ky);	

	for (i = 0; i < Mx; i++) {
                for (j = 0; j < My; j++){
                        k_arr[i*Mx + j] = sqrt(pow2(kx[i]) + pow2(ky[j]));
			k2_host[i*Mx + j] = pow2(kx[i]) + pow2(ky[j]);
		        X[i*Mx + j] = x[i];
			Y[i*Mx + j] = y[j];	
                       // cout << "k_arr row: " << i << " column: " << j << " = " << k2_host[i*My + j] << endl;
                }
        }

	// cout << " Debugging Complete" << endl;
	// exit(1);	

	// Initial Condition <start> 
	k_sq = 2*PI/a_sq;
	k_tri = 2*PI/a_tri;
	cout << "k_sq = " << k_sq << " k_tri = " << k_tri << endl;	

	if (iconfig == 1) {
        	auto in_func = [](double x, double y){
        	// return A_10*( (3*sqrt(2)-2)*cos(k_sq*y) - 2*cos(k_sq*x) ) - A_11*cos(k_sq*x)*cos(k_sq*y);
        	return A_10*( (3.0*sqrt(2.0)-2.0)*exp(I*k_sq*y).real() - 2.0*exp(I*k_sq*x).real() ) - A_11*exp(I*k_sq*x).real()*exp(I*k_sq*y).real();
		};
		
		r = roa*a_sq; // Radius for initial seed 

		for (i = 0 ; i < Mx; i++){
                	for (j = 0; j < My; j++){
                        psi_host[i*My + j].x =  in_func( X[i*My + j], Y[i*My + j] ) + psi_0;
			psi_host[i*My + j].y = 0.0;
                        // cout << "X row: " << i << " column: " << j << " = " << X[i*My + j] << endl;
                        // cout << "Y row: " << i << " column: " << j << " = " << Y[i][j] << endl;
                	// cout << "psi row: " << i << " column: " << j << " = " << psi[i*My + j] << endl;
                	}
        	}


        
        } else if(iconfig == 0) {
                auto in_func = [](double x, double y){
          	return A_tri*( cos(k_tri*x)/2.0 + cos(k_tri*x/2.0)*cos(sqrt(3.0)*k_tri*y/2.0) );
		};
		
		r = roa*a_tri; // Radius for initial seed

		for (i = 0 ; i < Mx; i++){
                        for (j = 0; j < My; j++){
                        psi_host[i*My + j].x =  in_func( X[i*My + j], Y[i*My + j] ) + psi_0;
			psi_host[i*My + j].y = 0.0;
                        // cout << "X row: " << i << " column: " << j << " = " << X[i*My + j] << endl;
                        // cout << "Y row: " << i << " column: " << j << " = " << Y[i][j] << endl;
                        // cout << "psi row: " << i << " column: " << j << " = " << psi_host[i*My + j].x << endl;
                        }
                }

        } else if(iconfig == 2) {
                auto in_func = [](double x, double y){
                return 0.0;
                };
		
		for (i = 0 ; i < Mx; i++){
                        for (j = 0; j < My; j++){
                        psi_host[i*My + j].x =  in_func( X[i*My + j], Y[i*My + j] ) + psi_0;
			psi_host[i*My + j].y = 0.0;
                        //cout << "X row: " << i << " column: " << j << " = " << X[i*My + j] << endl;
                        //cout << "Y row: " << i << " column: " << j << " = " << Y[i*My + j] << endl;
                	//cout << "psi row: " << i << " column: " << j << " = " << psi_host[i*My + j].x << endl;
                        }
                }
        } else if(iconfig == 10) {
		read_density_field( psi_host ); // This is where the simulation is failing
 	} else {
		printf("How did I get here?\n");
		exit(1);
	}
	// Initial Condition <end> 		
	// printf( "here" );	
	// exit(EXIT_SUCCESS);

	// Applying Seed (if applicable) <Start>	

      	if (iseed == 1) {
		cout << "r = " << r << endl;
		if (ishape == 0) {
			// 0 Circle
                	auto cir_func = [](double x, double y){
                        	return sqrt( pow2(x-h) + pow2(y-k) ); // roa = r/a
                	};
			for (i = 0 ; i < Mx; i++){
                        	for (j = 0; j < My; j++){
                                	if ( cir_func( X[i*My + j], Y[i*My + j] ) > r ) {
                                        	psi_host[i*My + j].x = psi_0;
                                	} else if ( cir_func( X[i*My + j], Y[i*My + j] ) <= r ) {
                                        	psi_host[i*My + j].x = psi_host[i*My + j].x * 1.0;
                                        	// cout << "psi row: " << i << " column: " << j << " = " << cir_func( X[i*My + j], Y[i*My + j]) << endl;
                                	}
                        	}
                	}
        	}
        	else if (ishape == 1) {
			// 1 Gaussian
                	auto cir_func = [](double x, double y, double h, double k, double roa){
                        	return exp( -(pow2(x-h) + pow2(y-k))/roa ); // roa is the variance
                	};
			for (i = 0 ; i < Mx; i++){
                                for (j = 0; j < My; j++){

					psi_host[i*My + j].x = psi_host[i*My + j].x - psi_0;
					psi_host[i*My + j].x = psi_host[i*My + j].x*cir_func( X[i*My + j], Y[i*My + j], h, k, roa ) + psi_0*(1 - cir_func( X[i*My + j], Y[i*My + j], h, k, roa ));
					//cout << "psi row: " << i << " column: " << j << " X = " <<  X[i*My + j] << " Y = " << Y[i*My + j] << endl; // Added 09/09/2025
                                	//cout << "psi row: " << i << " column: " << j << " = " << psi_host[i*My + j].x << endl;
				}
                        }

        	}
		//exit(EXIT_SUCCESS); // Added 09/09/2025
		cout << "Seed Done... " << endl;

        } else if( iseed == 0 || iseed!= 0 && iseed != 1 ) {
		cout << "No Seed, Solid Phase Only... " << endl;
        }
	// Applying Seed (if applicable) <end> 
	
	// Configuring the average density of system <start> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
	
	configure_avg_density( );	
	psi_avg = mean(psi_avg);
	cout << "psi_avg = " << psi_avg << endl;
	
	// Configuring the average density of system <end> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	// Direct Correlation Function Calculation <start> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	/* // Uncomment below for DCF envelope
	auto c2_k_fcn = [](double k) {                                                                                                                                      return max( exp(-pow2(s*k_10)/(2.0*rho_10*B_10))*exp(-pow2(k-k_10)/(2.0*pow2(al_10))), exp(-pow2(s*k_11)/(2.0*rho_11*B_11))*exp(-pow2(k-k_11)/(2.0*pow2(al_11))) );
        };
	*/ 

	// Uncomment below for DCF sum
	auto c2_k_fcn = [](double k) {
						    return exp(-pow2(s*k_10)/(2.0*rho_10*B_10))*exp(-pow2(k-k_10)/(2.0*pow2(al_10))) + exp(-pow2(s*k_11)/(2.0*rho_11*B_11))*exp(-pow2(k-k_11)/(2.0*pow2(al_11))) ;
        };


        
	//#pragma omp parallel for collapse(2)
        for (i = 0 ; i < Mx; i++){
                for (j = 0; j < My; j++){
                        c2_k_host[ i*My + j ] = c2_k_fcn( k_arr[i*My + j] ) ;
			//#pragma omp critical{
                        //cout << "c2_k row: " << i << " column: " << j << " = " << c2_k_host[i*My + j] << endl;
			//}
                }
        }
	
       // Direct Correlation Function Calculation <end> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
       //exit(EXIT_SUCCESS);

	

	printf("Memory allocated CPU: %lf MB\n", mem_use/1E6) ;  fflush( stdout ) ;

}


void allocate_cpu ( ) {
	// Allocate memory for use on the Host
	int i;

	// Allocates for FE_calc.cu for CPU %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	cudaMallocHost((void**)&F_host, N_t_steps * sizeof( double )); // CPU pinned memory 

	mem_use += N_t_steps * sizeof( double ); // x1 1D array of size N_t_steps x 1

	// Allocates DCF variables for use in main.cu and convolve_fields.cu* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// *Memory not pinned because it is only transferred once per simulation  
	c2_k_host = (double *) malloc(M2 * sizeof( double ) ); // CPU non-pinned memory	
	c2_k_hostP = (double *) malloc(M2 * sizeof( double ) );

	// Allocates k-space variables for use in the Pseudo-Spectral Method, euler_step.cu* %%%%%%%%%%%%%%%%%%%%%%%%
	k2_host = (double *) malloc(M2 * sizeof( double ) ); // CPU non-pinned memory 
	k2_hostP = (double *) malloc(M2 * sizeof( double ) );
	k_arr = (double *) malloc(M2 * sizeof( double ) ); // CPU non-pinned memory 
	kx = (double *) malloc(Mx * sizeof( double ) ); // CPU non-pinned memory
	ky = (double *) malloc(My * sizeof( double ) ); // CPU non-pinned memory

	// Allocates two test functions for permuting c2_k and k2 in the correct order %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	test_arr1 = (Complex *) malloc(M2 * sizeof( Complex ) );
	test_arr2 = (Complex *) malloc(M2 * sizeof( Complex ) );

	// Allocates real-space variables for initializing code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	//r_arr = (double *) malloc(M2 * sizeof( double ) );
	X = (double *) malloc(M2 * sizeof( double ) );
	Y = (double *) malloc(M2 * sizeof( double ) );
	x = (double *) malloc(Mx * sizeof( double ) );
	y = (double *) malloc(My * sizeof( double ) );

	// Allocates order-parameters in real-space %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	cudaMallocHost( (void**)&psi_host, M2 * sizeof( Complex )); // CPU pinned memory
	psi_host_out = (Complex *) malloc( M2 * sizeof( Complex ));
	reduced_array_host = (double *) malloc( (M2/num_gpus) * sizeof( double ) );
	cudaMallocHost( (void**)&SUB1_host, M2 * sizeof( Complex ) );	
	
	mem_use += 8*M2*sizeof( double ); // x5 real 2D arrays of size Mx x My
	mem_use += 2*Mx*sizeof( double ); // x2 1D arrays of size Mx x 1
	mem_use += 2*My*sizeof( double ); // x2 1D arrays of size My x 1
	mem_use += 4*M2*sizeof( Complex ); // x2 float2 arrays of size Mx x My
	mem_use += (M2/num_gpus)*sizeof( double ); // x1 float array size M2/num_gpus
}
