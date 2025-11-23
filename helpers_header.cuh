#pragma once

#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>

uint64_t sdiv (uint64_t a, uint64_t b) {
    return (a+b-1)/b;
}

class Timer {

    float time;
    const uint64_t gpu;
    cudaEvent_t ying, yang;

public:

    Timer (uint64_t gpu=0) : gpu(gpu) {
        cudaSetDevice(gpu);
        cudaEventCreate(&ying);
        cudaEventCreate(&yang);
    }

    ~Timer ( ) {
        cudaSetDevice(gpu);
        cudaEventDestroy(ying);
        cudaEventDestroy(yang);
    }

    void start ( ) {
        cudaSetDevice(gpu);
        cudaEventRecord(ying, 0);
    }

    void check_hours( ) {
	cudaSetDevice(gpu);
    }

    void stop (std::string label) {
        cudaSetDevice(gpu);
        cudaEventRecord(yang, 0);
        cudaEventSynchronize(yang);
        cudaEventElapsedTime(&time, ying, yang);
        std::cout << "TIMING: " << time << " ms (" << label << ")" << std::endl;
    }
};

void check_cufft_result(cufftResult result, const char* task_description) {
    if (result != CUFFT_SUCCESS) {
        const char* errorString;
        switch (result) {
            case CUFFT_INVALID_PLAN:
                errorString = "CUFFT_INVALID_PLAN";
                break;
            case CUFFT_ALLOC_FAILED:
                errorString = "CUFFT_ALLOC_FAILED";
                break;
            case CUFFT_INVALID_TYPE:
                errorString = "CUFFT_INVALID_TYPE";
                break;
            case CUFFT_INVALID_VALUE:
                errorString = "CUFFT_INVALID_VALUE";
                break;
            case CUFFT_INTERNAL_ERROR:
                errorString = "CUFFT_INTERNAL_ERROR";
                break;
            case CUFFT_EXEC_FAILED:
                errorString = "CUFFT_EXEC_FAILED";
                break;
            case CUFFT_SETUP_FAILED:
                errorString = "CUFFT_SETUP_FAILED";
                break;
            case CUFFT_INVALID_SIZE:
                errorString = "CUFFT_INVALID_SIZE";
                break;
            case CUFFT_UNALIGNED_DATA:
                errorString = "CUFFT_UNALIGNED_DATA";
                break;
            default:
                errorString = "Unknown CUFFT error";
        }
        cout << task_description << " error: " << errorString << " code: " << result << "" << std::endl;
        exit(EXIT_FAILURE);
    } else {
        cout << task_description << " successful." << std::endl;
    }
}



