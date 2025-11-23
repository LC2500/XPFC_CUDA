#include "globals.cuh"



void linspace(double x1, double x2, double N, double* vec){
	// function that creates a vector of linearly spaced values
	// x1 is the starting value
	// x2 is the ending value
	// N is the number of points
	// vec is the pre-allocated array for containing the evenly spaced array
	int i;
	double du = x2/(N-1) ; 
	vec[ 0 ] = x1 ;
	for(i=0; i<N; ++i) {
		vec[ i + 1] = vec[i] + du;
	}

}






