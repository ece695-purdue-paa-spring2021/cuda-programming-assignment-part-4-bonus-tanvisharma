

#ifndef CUDA_LIB_H
#define CUDA_LIB_H

	#include "cpuLib.h"

	#include <cuda.h>
	#include <curand_kernel.h>

	#define TILE_WIDTH 8
	#define TILE_HEIGHT 8
	#define TILE_BREADTH 8

	// Uncomment this to suppress console output
	#define DEBUG_PRINT_DISABLE

	#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }
	extern void gpuAssert(cudaError_t code, const char *file, int line, bool abort);

	/**
	 * @brief 
	 * 
	 * @param argc 
	 * @param argv 
	 * @return int 
	 */
	extern int runAlexNet (int argc, char ** argv);

	/**
	 * @brief CPU entrypoint for GPU based conv operation for a layer
	 * 
	 * @param iShape 	TensorShape
	 * @param fShape 	TensorShape
	 * @param oShape 	TensorShape &		output tensor dimensions - reference
	 * @param args 		ConvLayerArgs
	 * @return 			uint64_t	 	number of errors
	 */
	extern void evaluateGpuConvLayer (float * input, TensorShape iShape, 
		float * filter, TensorShape fShape, 
		float * bias, float * output, TensorShape oShape, 
		ConvLayerArgs convArgs);
	
	extern void Dampen(float * output, TensorShape shape, float factor);
	
	/**
	 * @brief 
	 * 
	 * @param chanTile  int            input channels accessed at a time
	 * @param input 	float *
	 * @param iShape 	TensorShape
	 * @param filter 	float *
	 * @param fShape 	TensorShape
	 * @param bias 		float *
	 * @param output 	float *
	 * @param oShape 	TensorShape		dimensions of output tensor
	 * @param args 		ConvLayerArgs	parameters for convolution operation	
	 * @return int 
	 */
	extern __global__ void convLayer_gpu ( int chanTile, float * input, TensorShape iShape, 
		float * filter, TensorShape fShape, 
		float * bias, float * output, TensorShape oShape, 
		ConvLayerArgs args);

	extern void evaluateGpuPoolLayer(float * input, TensorShape inShape,
			float * output, TensorShape outShape, PoolLayerArgs poolArgs);

	/**
	 * @brief GPU kernel to perform 2D Pool operation
	 * 
	 * @param input 	float *			pointer to input tensor
	 * @param inShape 	TensorShape		dimensions of input tensor
	 * @param output 	float *			pointer to output tensor
	 * @param outShape 	TensorShape		dimensions of output tensor
	 * @param args 		PoolLayerArgs	parameters of pool operation
	 * @return int 
	 */
	 extern __global__ void poolLayer_gpu (float * input, TensorShape inShape,
		float * output, TensorShape outShape, PoolLayerArgs args);

	extern void evaluateGpuFCLayer(float * a, TensorShape aShape, 
		float * b, TensorShape bShape, 
		float * c, TensorShape cShape, 
		GemmLayerArgs args);

	extern __global__ void gemmLayer_gpu (float * a, TensorShape aShape, 
		float * b, TensorShape bShape,
		float * c, TensorShape cShape,
		GemmLayerArgs args);

#endif
