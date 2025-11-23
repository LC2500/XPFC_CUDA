# XPFC_CUDA
The following CUDA/C++ program is a GPU accelerated code based on the structural phase field crystal method (XPFC), PRL 105, 045702 (2010) and PRE 83, 031601 (2011). The code makes use of cuFFT and NCCL from the NVIDIA HPC SDK by NVIDIA. 


## Requirements
This program requires the NVIDIA CUDA Toolkit (or CUDA HPC SDK) to compile and run.
It can be downloaded from the official NVIDIA website:

- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk)

These libraries are proprietary software provided by NVIDIA Corporation under their
respective End User License Agreements (EULA). This project does not include or
redistribute any part of those libraries.


## References
See https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/ for GPU kernel reduction. The original article was written by Justin Luitjens. Some of the code was modified from the original source, see file: sum_reduction.cu.


## File Creation
A makefile needs to be adjusted to the user's system.
