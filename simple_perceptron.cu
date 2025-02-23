#include <iostream>
#include <cuda_runtime.h>

// Defining the activation function kernel
// __global__ is used to define a kernel function that will be executed on the GPU
__global__ void stepFunctionKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// Defining a funtion to print an array
// This function is used to print the elements of an array
void printArray(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

// Move kernel definition outside of main()
__global__ void computeWeightedSumKernel(float* inputs, float* weights, float* outputs, int numSamples, int numFeatures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        float dotProduct = 0.0f;
        for (int j = 0; j < numFeatures; ++j) {
            dotProduct += inputs[idx * numFeatures + j] * weights[j];
        }
        outputs[idx] = dotProduct;
    }
}

// Defining the main function
int main() {
    // Parameters
    const int numFeatures = 3;
    const int numSamples = 4;

    // Host arrays
    float h_weights[numFeatures] = {0.5f, -0.6f, 0.2f};
    float h_inputs[numSamples * numFeatures] = {
        1.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        0.0f, 0.0f, 0.0f
    };

    float h_outputs[numSamples];

    // Device arrays (GPU)
    float *d_weights, *d_inputs, *d_outputs;

    // Allocate memory on the GPU
    cudaMalloc(&d_weights, numFeatures * sizeof(float));
    cudaMalloc(&d_inputs, numSamples * numFeatures * sizeof(float));
    cudaMalloc(&d_outputs, numSamples * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_weights, h_weights, numFeatures*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs, h_inputs, numSamples*numFeatures*sizeof(float), cudaMemcpyHostToDevice);

    // Compute weighted sum for each sample on GPU
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (numSamples + threadsPerBlock - 1) / threadsPerBlock;

    computeWeightedSumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_inputs, d_weights, d_outputs, numSamples, numFeatures);
    cudaDeviceSynchronize();
    
    // Apply the step function
    stepFunctionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_outputs, d_outputs, numSamples);
    cudaDeviceSynchronize();

    // Copy the results back to the host
    cudaMemcpy(h_outputs, d_outputs, numSamples*sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print the results
    printArray(h_outputs, numSamples);

    // Free the allocated memory
    cudaFree(d_weights);
    cudaFree(d_inputs);
    cudaFree(d_outputs);

    return 0;
}