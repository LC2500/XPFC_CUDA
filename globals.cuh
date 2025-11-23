#include <complex>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <cufftXt.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include "omp.h"

using namespace std ;

#define PI M_PI
#define pow2(x) ((x)*(x))
#define pow3(x) ((x)*(x)*(x))
#define pow4(x) ((x)*(x)*(x)*(x))
#define min(A,B) ((A)<(B) ? (A) : (B) )
#define Kdelta(i,j) ((i==j) ? 1 : 0 )
#define Dim 2
#define Maxsp 6
#define TSIZE 512
#define FULL_MASK 0xffffffff

typedef struct{ 
	double x;
	double y;	
} Complex;

// Allocating int variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#ifndef MAIN
extern
#endif
int t, t_sample, Lx, Ly, Lz, Mx, My, Mz, Mx_seg, My_seg, M2, M_k, M2_k, nthreads, no_val, iconfig, iseed, ishape, num_gpus, Bsz_x, Bsz_y, root_GPU, sr_threads, sr_blocks;

// Allocating double variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#ifndef MAIN
extern
#endif
double s, a, b, Mo, eps, v, dt, N_t_steps, ff_factor, dxy, k_10, k_11, rho_10, rho_11, B_10, B_11, al_10, al_11, a_tri, a_sq, A_10, A_11, k_sq, A_tri, k_tri, h, k, roa, r, mem_use, mem_use_gpu, psi_avg, psi0_vtk;

// Allocating double variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#ifndef MAIN
extern
#endif
double psi_0;

// Allocating double arrays %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#ifndef MAIN
extern
#endif
double *den, *ff, *k_arr, *k2_host, *kx, *ky, *x, *y, *kX, *kY, *X, *Y, *r_arr, *F, *ele_arr, *tmp_vec_r, *convl_r, *psi_kp1_r, *fin, *F_host, *c2_k_host, *reduced_array_host, *c2_k_hostP, *k2_hostP, *SUB1_host; 

// Allocating complex double arrays %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#ifndef MAIN
extern
#endif
complex<double> *num, *tmp_vec_i, I, *convl;

// Allocating chunk sizes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#ifndef MAIN
extern
#endif
uint64_t chunk_size_real, chunk_size_cplx;

// Allocating uint64_t variables
#ifndef MAIN
extern
#endif
uint64_t lower, upper, width, width_k;

#ifndef MAIN
extern
#endif
cufftResult result;

#ifndef MAIN
extern
#endif
cufftHandle ft_fwd, ft_back; // ft_fwd is R2C, ft_back is C2R

#ifndef MAIN
extern
#endif
Complex *psi_host, *psi_host_out, *test_arr1, *test_arr2;

#ifndef MAIN
extern
#endif
int3 grid_size, block_size; 

// Device Routines //
__global__ void convolve_fields( cufftDoubleComplex *, double *, cufftDoubleComplex *, int, int, int ) ;
__global__ void euler_step( int, double, double, double, cufftDoubleComplex *, cufftDoubleComplex *, cufftDoubleComplex *, cufftDoubleComplex *, double *, double *, double, double, int, int, int ) ;
__global__ void scaling_step( cufftDoubleComplex *, int, int, int, int ) ;
__global__ void FE_calc( int, double, double, cufftDoubleComplex *, cufftDoubleComplex *, cufftDoubleComplex *, cufftDoubleComplex *, cufftDoubleComplex *, int, int, int, int, double*);
__global__ void apply_power( cufftDoubleComplex *, cufftDoubleComplex *, int, int, int, int, int );
__inline__ __device__ double warpReduceSum( double );
__inline__ __device__ double blockReduceSum( double );
__global__ void deviceReduceKernel( double *, double *, int );
__global__ void deviceReduceKernelFinalCall( double *, double *, int, int, double );
__global__ void deviceReduceKernelFinalCallSSE( double *, double *, int, int, int );
__global__ void copy_matrix(cufftDoubleComplex *, cufftDoubleComplex *, int, int, int);
__global__ void copy_struct( cufftDoubleComplex *, double *, int, int, int );
__global__ void residual_calc( int, cufftDoubleComplex *, cufftDoubleComplex *, cufftDoubleComplex *, cufftDoubleComplex *, cufftDoubleComplex *,double *, double *, int M, double dt, double a, double b, int gpu_i, int Mx, int num_gpus );

// Host Routines //
void fftw_fwd( double* , complex<double>* );
void fftw_back( complex<double>* , double* );

double mean( double );
void configure_avg_density( void ); 

void linspace( double,double,double, double* );
void FFT_freq( int,int, double* );

void read_input( void );
void obtain_box_size( void );
void read_density_field( Complex * );

void initialize( void );
double randn( void );
void die(const char *kill);

void writeVTKFile( Complex*, int, int, int, int, int );
void writeTXTFile( double*, int, int, int, int, double, double );

void copy2cpu( cudaLibXtDesc *, Complex *);
void convolution_wrapper( cudaLibXtDesc *, double **, cudaLibXtDesc * );
void power_function_wrapper( cudaLibXtDesc *, cudaLibXtDesc *, int );
void FE_calc_wrapper( int, cudaLibXtDesc *, cudaLibXtDesc *, cudaLibXtDesc *, cudaLibXtDesc *, cudaLibXtDesc *, double **, double*, double *, ncclComm_t * );
void euler_step_wrapper( int, cudaLibXtDesc *, cudaLibXtDesc *, cudaLibXtDesc *, cudaLibXtDesc *, double **, double ** );
void scaling_step_wrapper( cudaLibXtDesc * );
void copy_matrix_wrapper(cudaLibXtDesc *, cudaLibXtDesc *);
void SSE_calc_wrapper( int, cufftHandle, cudaLibXtDesc *, cudaLibXtDesc *, cudaLibXtDesc *, cudaLibXtDesc *, cudaLibXtDesc *, cudaLibXtDesc *, double **, double **, double **, double *, double *, ncclComm_t * ); 
// n, planComplex, res_out, res_k, psi_kp1, psi_k, psi_2k, psi_3k, c2_k, k2, gpu_reduced_array, reduced_array //

void permute_matrices(cufftHandle, double *, double *);
void random_input( Complex * );


