#include "globals.cuh"
////////////////////////////
/// Routines in files:   ///
/// read_input           ///
/// read_density_field   ///
////////////////////////////

void read_input( void ) {
	FILE *inp;
	int i,j;

	printf("\n######## Reading pfc.input ########\n");
	inp = fopen( "pfc.input", "r") ;
	if (inp == NULL) {
        	perror("Error opening file");
    	} 

	char tt[80] ;
	fscanf(inp, "%d", &no_val ) ;
	fgets( tt, 80, inp) ;

	fscanf(inp, "%lf %lf", &psi_0, &s) ; // initial density and sigma (Temperature ) parameter
	fgets(tt, 80, inp) ; 
	cout << "psi_0 = " << psi_0 << " s = " << s << endl; 	

	fscanf(inp, "%lf %lf %lf %lf", &a, &b, &eps, &Mo) ; // Kinetic parameters, a and b, magnitude parameter epsilon, and mobility (diffusion) parameter M
	fgets(tt, 80, inp) ;
	a = 1/a; b = 1/b;
	cout << "a = " << a << " b = " << b << " eps = "<< eps << " Mo = " << Mo << endl;

	// Blank line //
  	fgets( tt , 80 , inp ) ;

	fscanf(inp, "%d %d", &Lx, &Ly); // x and y dimensions 
  	fgets( tt , 80 , inp ) ;
	cout << "Lx = " << Lx << " Ly = " << Ly << endl;

	fscanf(inp, "%d %d", &Mx, &My ) ; // x and y grid points 
        fgets( tt, 80, inp) ;
	cout << "Mx = " << Mx << " My = " << My << endl;

	// Blank line //
        fgets( tt , 80 , inp ) ;

	fscanf(inp, "%d", &t ) ; // final time 
        fgets( tt, 80, inp) ;
	cout << "t = " << t << endl;

	fscanf(inp, "%lf", &dt ) ; // time step
	fgets( tt, 80, inp) ;
	cout << "dt = " << dt << endl;

	fscanf(inp, "%d", &t_sample ) ; // sampling time
	fgets( tt, 80, inp) ;
        cout << "t sampling = " << t_sample << endl;
	
	// Blank line //
        fgets( tt , 80 , inp ) ;
	
	fscanf(inp, "%d", &iconfig ) ; // initial configuration (tri/sq/liq)
        fgets( tt, 80, inp) ;
	
	if (iconfig == 1) { 
		cout << "Initial Configuration: Square " << endl;
	} else if(iconfig == 0) {
		cout << "Initial Configuration: Triangular " << endl;
	} else if(iconfig == 2) {
		cout << "Initial Configuration: Liquid " << endl;
	} else if(iconfig == 10){
		cout << "Initial Configuration: From .vtk file " << endl;
	} else {
		cout << "Invalid Initital Configuration " << endl;
		exit( 1 );
	}
	
	fscanf(inp, "%lf %lf", &A_10, &A_11 ) ; // Initial Square Amplitudes
        fgets( tt, 80, inp) ;
        cout << "A_10 = " << A_10 << " A_11 = " << A_11 << endl;

	fscanf(inp, "%lf", &A_tri ) ; // Initial Triangular Amplitudes
        fgets( tt, 80, inp) ;
	cout << "A_tri = " << A_tri << endl;


        fgets( tt , 80 , inp ); // Blank line 
	
	fscanf(inp, "%d", &iseed ) ; // Initial Seed (1 yes/0 no)
	fgets( tt, 80, inp) ;
	

	if (iseed == 1) {
                cout << "Seed: Yes " << endl;
        } else if(iseed == 0) {
                cout << "Seed: No " << endl;
        } else {
                cout << "Invalid Initital Configuration " << endl;
                exit( 1 );
        }

	fscanf(inp, "%d", &ishape ) ; // Initial Seed (1 yes/0 no)
        fgets( tt, 80, inp) ;


        if (ishape == 1) {
                cout << "Seed Shape: Gaussian " << endl;
        } else if(ishape == 0) {
                cout << "Seed Shape: Circle " << endl;
        } else {
                cout << "Invalid Initital Seed Shape " << endl;
                exit( 1 );
        }


	fscanf( inp, "%lf %lf %lf", &h, &k, &roa ) ; // Parameters for the circle ( h, k, r/a )
        fgets( tt, 80, inp ) ;
	
	cout << "h = " << h << " k = " << k << " r/a = " << roa << endl; // added 09/10/2025

	fclose( inp ) ;
	printf("######## Reading pfc.input Complete ########\n");
}


void obtain_box_size( void ) {
	FILE *inp;
	int Mx_t, My_t, Mz_t;
	double dx_t, dy_t, dz_t;
	printf("\n######## Obtaining Box Size ########\n");
	inp = fopen( "initial_density_field.vtk", "r");
        if (inp == NULL) {
                perror("Error opening file");
        }
	
	char tt[80];

        fgets( tt, 80, inp) ; // # vtk DataFile Version 3.0
        fgets( tt, 80, inp) ; // Density Data
        fgets( tt, 80, inp) ; // ASCII
        fgets( tt, 80, inp) ; // DATASET STRUCTURED_POINTS

	fscanf(inp, "DIMENSIONS %d %d %d", &Mx_t, &My_t, &Mz_t ) ;

	fgets( tt, 80, inp) ; // DIMENSIONS 1024 1024 1
        printf(tt);
        fgets( tt, 80, inp) ; // ORIGIN 0 0 0
        printf(tt);

	fscanf(inp, "SPACING %lf %lf %lf", &dx_t, &dy_t, &dz_t ) ;
	
	Lx = round(Mx_t*dx_t); Ly = round(My_t*dy_t); Lz = round(Mz_t*dz_t);
	
	printf("obtain_box_size: Lx = %d Ly = %d Lz = %d", Lx, Ly, Lz );
 	fclose( inp ) ;
	 printf("\n######## Box Size Complete ########\n");
}





void read_density_field( Complex *psi ) {
	FILE *inp;
	int i,j;
	int Mx_t, My_t, Mz_t; // Test values for grid size
	
	inp = fopen( "initial_density_field.vtk", "r");
	if (inp == NULL) {
		perror("Error opening file");
        }

	printf("\n######## Reading initial density field vtk ########\n");

        char tt[80];
      
        fgets( tt, 80, inp) ; // # vtk DataFile Version 3.0
	printf(tt);	
	fgets( tt, 80, inp) ; // Density Data
	printf(tt);
	fgets( tt, 80, inp) ; // ASCII
	printf(tt);
	fgets( tt, 80, inp) ; // DATASET STRUCTURED_POINTS
	fscanf(inp, "DIMENSIONS %d %d %d", &Mx_t, &My_t, &Mz_t ) ;	
	printf(tt);

	if (Mx_t != Mx) {
		printf("Incompatible x dimension: Mx_t = %d and Mx = %d \n", Mx_t, Mx);
	} else if (My_t != My) {
		printf("Incompatible y dimension: My_t = %d and My = %d \n", My_t, My);
	} else if (Dim == 2 && Mz_t != 1) {
		printf("Incompatible z dimension: Mz_t = %d and Mz = %d \n", Mz_t, 1);
	}
                
	printf("Dim: %d Mx_t = %d My_t = %d Mz_t = %d\n", Dim, Mx_t, My_t, Mz_t );

	fgets( tt, 80, inp) ; // DIMENSIONS 1024 1024 1
        printf(tt);
	fgets( tt, 80, inp) ; // ORIGIN 0 0 0
        printf(tt);
	fgets( tt, 80, inp) ; // SPACING 0.143555 0.143555 1
        printf(tt);
	fgets( tt, 80, inp) ; // POINT_DATA 1048576
	printf(tt);
	fgets( tt, 80, inp) ; // SCALARS DensityData float
	printf(tt);
	fgets( tt, 80, inp) ; // LOOKUP_TABLE default
        printf(tt);

	for(i=0;i<Mx_t;i++){
		for(j=0;j<My_t;j++){
			if (fgets(tt, sizeof(tt), inp) == NULL) {
				printf(tt);
				fprintf(stderr, "Unexpected end of file at index %d\n", i*My_t + j);
				exit(1);
			}

			double val;
			if (sscanf(tt, "%lf", &val) == 1) {
				psi[i*My_t + j].x = val;
			} else {
				fprintf(stderr, "Error parsing value at index %d\n", i*My_t + j);
				break;
			}
			
			/*
			if (fscanf(inp, "%lf", &psi[i].x) != 1) {
        			fprintf(stderr, "Error reading data at index %d\n", i);
				exit(1);
    			}
			*/
		}
	}

	

	fclose( inp ) ;
	
	printf("######## Initial density field vtk complete ########\n\n");

	//exit(EXIT_SUCCESS);

}


