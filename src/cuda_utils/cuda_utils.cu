//
// Created by chnlkw on 11/14/17.
//

#include "cuda_utils.h"

#ifdef USE_CUDA

#define REPORT_CUDA_SUCCESS 0

bool cudaEnsureSuccess(cudaError_t status, const char *status_context_description,
                       bool die_on_error, const char *filename, unsigned line_number) {
    if (status_context_description == NULL)
        status_context_description = "";
    if (status == cudaSuccess) {
#if REPORT_CUDA_SUCCESS
        std::cerr <<  "Succeeded: " << status_context_description << std::endl << std::flush;
#endif
        return true;

    }
    const char *errorString = cudaGetErrorString(status);
    std::cerr << "CUDA Error: ";
    if (status_context_description != NULL) {
        std::cerr << status_context_description << ": ";
    }
    if (errorString != NULL) {
        std::cerr << errorString;
    } else {
        std::cerr << "(Unknown CUDA status code " << status << ")";
    }
    if (filename != NULL) {
        std::cerr << " at " << filename << ":" << line_number;
    }

    std::cerr << std::endl << std::flush;
    if (die_on_error) {
        abort();
        //exit(EXIT_FAILURE);
        // ... or cerr << "FATAL ERROR" << etc. etc.

    }
    return false;

}

#endif
