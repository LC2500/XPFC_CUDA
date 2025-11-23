#include "globals.cuh"
// Original Source: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
///////////////////////////////////////////
/// Routines from Original Source:	///
/// warpReduceSum			///
/// blockReduceSum			///
/// deviceReduceKernel			///
///////////////////////////////////////////
/// Modified Routines:			///
/// deviceReduceKernelFinalCall		///
/// deviceReduceKernelFinalCallSSE	///
///////////////////////////////////////////

__inline__ __device__ double warpReduceSum(double val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  return val;
}

__inline__ __device__ double blockReduceSum(double val) {
        static __shared__ double shared[32]; // Shared mem for 32 partial sums
        int lane = threadIdx.x % warpSize;
        int wid = threadIdx.x / warpSize;
        val = warpReduceSum( val );     // Each warp performs partial reduction

        if (lane==0){
                shared[wid]=val; // Write reduced value to shared memory
	}

        __syncthreads(); // Wait for all partial reductions

	// Ternary conditional operator (conpact if statements)
	// syntax: condition ? value_if_true : value_if_false;
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0; //read from shared memory only if that warp existed

        if (wid==0) { 
		val = warpReduceSum(val); //Final reduce within first warp
	}

        return val;
}

__global__ void deviceReduceKernel(double *in, double* out, int N) {
        double sum = 0.0;
        //reduce multiple elements per thread
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
                sum += in[i];
        }
        sum = blockReduceSum(sum);
        if (threadIdx.x==0) {
                out[blockIdx.x]=sum;
	}
}

// Modified from the Original Source //
__global__ void deviceReduceKernelFinalCall(double *in, double* out, int N, int it, double ff_factor) {
        double sum = 0.0;
        //reduce multiple elements per thread
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
                sum += in[i];
        }
        sum = blockReduceSum(sum);
        if (threadIdx.x==0) {
                out[blockIdx.x]=sum; printf("iteration: %d F = %0.10f", it, sum*ff_factor);
        }
}

// Modified from the Original Source //
__global__ void deviceReduceKernelFinalCallSSE( double *in, double* out, int N, int M2, int it ) {
        double sum = 0.0;
        //reduce multiple elements per thread
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
                sum += in[i];
        }
        sum = blockReduceSum(sum);
        if (threadIdx.x==0) {
                out[blockIdx.x]= sum; printf( " SSE_gpu = %0.5e \n", sqrt(sum/M2) );

        }
}


