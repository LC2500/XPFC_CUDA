#include "globals.cuh"


void FFT_freq(int N, int L, double* k) {
	int i ;
	int rem = N % 2; // checks if the number of elements is even or odd 
	if (rem != 0) {
		for(i = 0; i < (N+1)/2; i++) {
			k[i] = (2*PI/double(L))*double(i) ;
		       // cout << "k[" << i << "] = " << k[i] << endl;
		}
		for(i = (N+1)/2; i < N; i++) {
			k[i] = (2*PI/L)*(i - N);
			// cout << "k[" << i << "] = " << k[i] << endl;
		}
	}
	else {
		for(i = 0; i < (N+2)/2; i++) {
                        k[i] = (2*PI/L)*i ;
                }
                for(i = (N+2)/2; i < N+1; i++) {
                        k[i] = (2*PI/L)*(i - N);

                }
	}
}
