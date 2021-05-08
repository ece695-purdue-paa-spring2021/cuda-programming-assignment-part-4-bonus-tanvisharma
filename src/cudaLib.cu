
#include "cudaLib.cuh"
#include<cassert>

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


int runAlexNet (int argc, char ** argv) {

	// ------------ Layer1 ---------------
	TensorShape iShape = AlexL1_InShape; 
	TensorShape fShape = AlexL1_FilterShape;
	ConvLayerArgs convArgs = AlexL1_ConvArgs; 
	TensorShape oShape;

	oShape.height 	= (iShape.height + 2 * convArgs.padH - fShape.height) / convArgs.strideH + 1;
	oShape.width	= (iShape.width  + 2 * convArgs.padW - fShape.width)  / convArgs.strideW + 1;
	oShape.channels	= (fShape.count);
	oShape.count 	= (iShape.count); // batch size

	std::cout << "Evaluate AlexNet-Layer1 : \n";
	std::cout << "Input : " << iShape << " \n";
	std::cout << "Filter : " << fShape << " \n";
	std::cout << "Output : " << oShape << " \n";
	std::cout << "ConvArgs : " << convArgs << " \n";
	
	// Initialize the inputs
	float * in1 = nullptr;
	float * filter1 = nullptr;
	float * bias1 = nullptr; 
	float * out1 = nullptr;

	int returnVal;
	returnVal = makeTensor(&in1, iShape);
	// assert(returnVal == 0);  //can be used for debugging
	returnVal = makeTensorBin(&filter1, fShape);
	returnVal = makeVector(&bias1, oShape.channels);
	out1 = (float *) malloc (tensorSize(oShape) * sizeof(float));
	// for debugging
	// for (int i = 0; i < oShape.channels; ++i){
	// 	bias1[i] = 0.5;
	// }

	// Run the required operation
	evaluateGpuConvLayer(in1, iShape, filter1, fShape, bias1, out1, oShape, convArgs);
	printTensor(out1, oShape);

	// Pass the output to MaxPool
	PoolLayerArgs poolArgs = Alex_PoolArgs;
	std::cout << "MaxPool Args : "<< poolArgs << "\n";
	float * out1pool;
	TensorShape oShapePool = {oShape.count, oShape.channels, (oShape.height - poolArgs.poolH)/poolArgs.strideH +1, (oShape.width - poolArgs.poolW)/poolArgs.strideW +1};
	out1pool = (float *) malloc(tensorSize(oShapePool) * sizeof(float));

	evaluateGpuPoolLayer(out1, oShape, out1pool, oShapePool, poolArgs);

	std::cout << oShapePool << "\n";
	printTensor(out1pool, oShapePool);

	free(filter1); free(bias1);

	// ---------- Layer2 ------------
	TensorShape fShape2 = AlexL2_FilterShape;
	ConvLayerArgs convArgs2 = AlexL2_ConvArgs; 
	TensorShape oShape2;
	
	oShape2.height 	= (oShapePool.height + 2 * convArgs2.padH - fShape2.height) / convArgs2.strideH + 1;
	oShape2.width	= (oShapePool.width  + 2 * convArgs2.padW - fShape2.width)  / convArgs2.strideW + 1;
	oShape2.channels	= (fShape2.count);
	oShape2.count 	= (oShapePool.count); // batch size

	std::cout << "Evaluate AlexNet-Layer2 : \n";
	std::cout << "Input : " << oShapePool << " \n";
	std::cout << "Filter : " << fShape2 << " \n";
	std::cout << "Output : " << oShape2 << " \n";
	std::cout << "ConvArgs : " << convArgs2 << " \n";
	
	// Initialize the inputs
	float * filter2 = nullptr;
	float * bias2 = nullptr; 
	float * out2 = nullptr;

	// assert(returnVal == 0);  //can be used for debugging
	returnVal = makeTensorBin(&filter2, fShape2);
	returnVal = makeVector(&bias2, oShape2.channels);
	out2 = (float *) malloc (tensorSize(oShape2) * sizeof(float));

	// Run the required operation
	evaluateGpuConvLayer(out1pool, oShapePool, filter2, fShape2, bias2, out2, oShape2, convArgs2);
	printTensor(out2, oShape2);

	// Pass the output to MaxPool
	std::cout << "MaxPool Args : "<< poolArgs << "\n";
	float * out2pool;
	TensorShape oShapePool2 = {oShape2.count, oShape2.channels, (oShape2.height - poolArgs.poolH)/poolArgs.strideH +1, (oShape2.width - poolArgs.poolW)/poolArgs.strideW +1};
	out2pool = (float *) malloc(tensorSize(oShapePool2) * sizeof(float));

	evaluateGpuPoolLayer(out2, oShape2, out2pool, oShapePool2, poolArgs);

	std::cout << oShapePool2 << "\n";
	printTensor(out2pool, oShapePool2);
	free(filter2); free(bias2);

	// ---------- Layer3 ------------
	TensorShape fShape3 = AlexL3_FilterShape;
	ConvLayerArgs convArgs3 = AlexL3_ConvArgs;
	TensorShape oShape3;

	oShape3.height 	= (oShapePool2.height + 2 * convArgs3.padH - fShape3.height) / convArgs3.strideH + 1;
	oShape3.width	= (oShapePool2.width  + 2 * convArgs3.padW - fShape3.width)  / convArgs3.strideW + 1;
	oShape3.channels	= (fShape3.count);
	oShape3.count 	= (oShapePool2.count); // batch size

	std::cout << "Evaluate AlexNet-Layer3 : \n";
	std::cout << "Input : " << oShapePool << " \n";
	std::cout << "Filter : " << fShape3 << " \n";
	std::cout << "Output : " << oShape3 << " \n";
	std::cout << "ConvArgs : " << convArgs3 << " \n";
	
	// Initialize the inputs
	float * filter3 = nullptr;
	float * bias3 = nullptr; 
	float * out3 = nullptr;

	// assert(returnVal == 0);  //can be used for debugging
	returnVal = makeTensorBin(&filter3, fShape3);
	returnVal = makeVector(&bias3, oShape3.channels);
	out3 = (float *) malloc (tensorSize(oShape3) * sizeof(float));

	// Run the required operation
	evaluateGpuConvLayer(out2pool, oShapePool2, filter3, fShape3, bias3, out3, oShape3, convArgs3);
	printTensor(out3, oShape3);

	// ---------- Layer4 ------------
	TensorShape fShape4 = AlexL4_FilterShape;
	ConvLayerArgs convArgs4 = AlexL4_ConvArgs;
	TensorShape oShape4;

	oShape4.height 	= (oShape3.height + 2 * convArgs4.padH - fShape4.height) / convArgs4.strideH + 1;
	oShape4.width	= (oShape3.width  + 2 * convArgs4.padW - fShape4.width)  / convArgs4.strideW + 1;
	oShape4.channels	= (fShape4.count);
	oShape4.count 	= (oShape3.count); // batch size

	std::cout << "Evaluate AlexNet-Layer4 : \n";
	std::cout << "Filter : " << fShape4 << " \n";
	std::cout << "Output : " << oShape4 << " \n";
	std::cout << "ConvArgs : " << convArgs4 << " \n";
	
	// Initialize the inputs
	float * filter4 = nullptr;
	float * bias4 = nullptr; 
	float * out4 = nullptr;

	// assert(returnVal == 0);  //can be used for debugging
	returnVal = makeTensorBin(&filter4, fShape4);
	returnVal = makeVector(&bias4, oShape4.channels);
	out4 = (float *) malloc (tensorSize(oShape4) * sizeof(float));

	// Run the required operation
	evaluateGpuConvLayer(out3, oShape3, filter4, fShape4, bias4, out4, oShape4, convArgs4);
	printTensor(out4, oShape4);

	// ---------- Layer5 ------------
	TensorShape fShape5 = AlexL5_FilterShape;
	ConvLayerArgs convArgs5 = AlexL5_ConvArgs;
	TensorShape oShape5;

	oShape5.height 	= (oShape4.height + 2 * convArgs5.padH - fShape5.height) / convArgs5.strideH + 1;
	oShape5.width	= (oShape4.width  + 2 * convArgs5.padW - fShape5.width)  / convArgs5.strideW + 1;
	oShape5.channels	= (fShape5.count);
	oShape5.count 	= (oShape4.count); // batch size

	std::cout << "Evaluate AlexNet-Layer5 : \n";
	std::cout << "Filter : " << fShape5 << " \n";
	std::cout << "Output : " << oShape5 << " \n";
	std::cout << "ConvArgs : " << convArgs5 << " \n";
	
	// Initialize the inputs
	float * filter5 = nullptr;
	float * bias5 = nullptr; 
	float * out5 = nullptr;

	// assert(returnVal == 0);  //can be used for debugging
	returnVal = makeTensorBin(&filter5, fShape5);
	returnVal = makeVector(&bias5, oShape5.channels);
	out5 = (float *) malloc (tensorSize(oShape5) * sizeof(float));

	// Run the required operation
	evaluateGpuConvLayer(out4, oShape4, filter5, fShape5, bias5, out5, oShape5, convArgs5);
	printTensor(out5, oShape5);

	// TensorShape oShape5 = {1, 256, 13, 13};
	// float * out5 = nullptr;

	// int returnVal;
	// returnVal = makeTensor(&out5, oShape5);
	// // // assert(returnVal == 0);  //can be used for debugging

	// PoolLayerArgs poolArgs = Alex_PoolArgs;

	// Pass the output to MaxPool
	std::cout << "MaxPool Args : "<< poolArgs << "\n";
	float * out5pool;
	TensorShape oShapePool5 = {oShape5.count, oShape5.channels, (oShape5.height - poolArgs.poolH)/poolArgs.strideH +1, (oShape5.width - poolArgs.poolW)/poolArgs.strideW +1};
	out5pool = (float *) malloc(tensorSize(oShapePool5) * sizeof(float));

	evaluateGpuPoolLayer(out5, oShape5, out5pool, oShapePool5, poolArgs);

	std::cout << oShapePool5 << "\n";
	printTensor(out5pool, oShapePool5);
	free(filter5); free(bias5);

	// ---------- Layer6 ------------
	TensorShape inShapeFC1 = {1, 1, oShapePool5.count, oShapePool5.width * oShapePool5.height * oShapePool5.channels};
	TensorShape fShapeFC1 = {1, 1, oShapePool5.width * oShapePool5.height * oShapePool5.channels, 4096};
	GemmLayerArgs FCargs = {16, 16, 1};
	TensorShape oShape6   = {1, 1, oShapePool5.count, 4096};
	
	std::cout << "Evaluate AlexNet-Layer6 : \n";
	std::cout << "Filter : " << fShapeFC1 << " \n";
	std::cout << "Output : " << oShape6 << " \n";
	
	// Initialize the inputs
	float * filter6 = nullptr;
	float * out6 = nullptr;

	returnVal = makeTensorBin(&filter6, fShapeFC1);
	out6 = (float *) malloc(tensorSize(oShape6) * sizeof(float));
	
	evaluateGpuFCLayer(out5pool, inShapeFC1, filter6, fShapeFC1, out6, oShape6, FCargs);
	printTensor(out6, oShape6);

	// ---------- Layer7 ------------
	TensorShape inShapeFC2 = oShape6;
	TensorShape fShapeFC2 = {1, 1, 4096, 1000};
	TensorShape oShape7   = {1, 1, oShape6.height, 1000};
	
	std::cout << "Evaluate AlexNet-Layer7 : \n";
	std::cout << "Filter : " << fShapeFC2 << " \n";
	std::cout << "Output : " << oShape7 << " \n";
	
	// Initialize the inputs
	float * filter7 = nullptr;
	float * out7 = nullptr;

	returnVal = makeTensorBin(&filter7, fShapeFC2);
	out7 = (float *) malloc(tensorSize(oShape7) * sizeof(float));
	
	evaluateGpuFCLayer(out6, inShapeFC2, filter7, fShapeFC2, out7, oShape7, FCargs);
	printTensor(out7, oShape7);

	return 0;
}

void evaluateGpuConvLayer(float * input, TensorShape iShape, 
	float * filter, TensorShape fShape, 
	float * bias, float * output, TensorShape oShape, 
	ConvLayerArgs convArgs)
{
	//Copy the tensors to GPU
	float * in_d, * filter_d, * bias_d, * out_d;
	cudaMalloc((void **) &in_d, tensorSize(iShape) * sizeof(float));
	cudaMemcpy(in_d, input, tensorSize(iShape) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **) &filter_d, tensorSize(fShape) * sizeof(float));
	cudaMemcpy(filter_d, filter, tensorSize(fShape) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **) &bias_d, oShape.channels * sizeof(float));
	cudaMemcpy(bias_d, bias, oShape.channels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **) &out_d, tensorSize(oShape)*sizeof(float));

	//Calculate the shared memory size
	int inp_chan_tile = 4; 
	int inpReqX = fShape.width + (TILE_WIDTH - 1)*convArgs.strideW;
	int inpReqY = fShape.height + (TILE_HEIGHT - 1)*convArgs.strideH;
	int inputReq = (inpReqX) * inpReqY * inp_chan_tile ;
	int filterReq = fShape.width * fShape.height * inp_chan_tile * TILE_BREADTH;
	int biasSize  = TILE_BREADTH;
	int sharedmemSize = (inputReq + filterReq + biasSize) * sizeof(float);
	#ifndef DEBUG_PRINT_DISABLE
		std::cout << "Input Shared mem size = " << inpReqX << ", " << inpReqY << ", " << inp_chan_tile << "\n";
		std::cout << "Shared mem size = " << sharedmemSize << "B\n";
	#endif

	//Calculate the grid,block dimensions
	dim3 dimBlock(TILE_WIDTH,TILE_HEIGHT,TILE_BREADTH);
	int gridSizeX = ceil(1.0f*oShape.width/TILE_WIDTH);
	int gridSizeY = ceil(1.0f*oShape.height/TILE_HEIGHT);
	int gridSizeZ = ceil(1.0f*oShape.channels/TILE_BREADTH); //Parallelizing over output channels
	#ifndef DEBUG_PRINT_DISABLE
		std::cout << "TB size: "<<TILE_WIDTH<<","<<TILE_HEIGHT<<","<<TILE_BREADTH<<"\n";
		std::cout << "Grid size: "<<gridSizeX<<","<<gridSizeY<<","<<gridSizeZ<<"\n";
	#endif
	dim3 dimGrid(gridSizeX,gridSizeY,gridSizeZ);

	//Launch the kernel
	convLayer_gpu<<<dimGrid,dimBlock,sharedmemSize>>>(inp_chan_tile, in_d, iShape, filter_d, fShape, bias_d, out_d, oShape, convArgs);
	
	cudaDeviceSynchronize();

	//Copy back the variables
	cudaMemcpy(output, out_d, tensorSize(oShape)*sizeof(float), cudaMemcpyDeviceToHost );

	if (convArgs.activation){
		Dampen(output, oShape, 1);
	}

	cudaFree(in_d); cudaFree(out_d); cudaFree(filter_d);

}

void Dampen(float * output, TensorShape shape, float factor){
	int offset;
	for (uint32_t count = 0; count < shape.count; ++ count) {
		for (uint32_t chIdx = 0; chIdx < shape.channels; ++ chIdx ) {
			for (uint32_t rowIdx = 0; rowIdx < shape.height; ++ rowIdx) {
				for (uint32_t colIdx = 0; colIdx < shape.width; ++ colIdx) {
					offset = count*shape.channels*shape.height*shape.width +chIdx * shape.height * shape.width + rowIdx * shape.width + colIdx;
					if (output[offset] < 0.0)
						output[offset] = 0.0;
                    else if (output[offset] > 1.0)
						output[offset] = 1.0 * factor;
				}
			}
		}
	}

}


void evaluateGpuPoolLayer(float * input, TensorShape inShape,
	float * output, TensorShape outShape, PoolLayerArgs poolArgs){

	//Copy the required variables to device mem
	float * inMatrix_d, * outMatrix_d;
	cudaMalloc((void **) &inMatrix_d, tensorSize(inShape)*sizeof(float));
	cudaMemcpy(inMatrix_d, input,  tensorSize(inShape)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **) &outMatrix_d, tensorSize(outShape)*sizeof(float));

	//Calculate shared memory size per threadblock
	int inputsizeH, inputsizeW, inputsizeC;
	inputsizeW = (min(TILE_WIDTH,outShape.width) - 1)*poolArgs.strideW + poolArgs.poolW;
	inputsizeH = (min(TILE_HEIGHT,outShape.height) - 1)*poolArgs.strideH + poolArgs.poolH;
	inputsizeC = TILE_BREADTH;

	unsigned int sharedmemSize = ( inputsizeH*inputsizeW*inputsizeC ) * sizeof(float);
	
	
	//Launch 2d kernel function
	dim3 dimBlock(TILE_WIDTH,TILE_HEIGHT,TILE_BREADTH);
	int gridSizeX = ceil(1.0f*outShape.width/TILE_WIDTH);
	int gridSizeY = ceil(1.0f*outShape.height/TILE_HEIGHT);
	int gridSizeZ = ceil(1.0f*outShape.channels/TILE_BREADTH);

	#ifndef DEBUG_PRINT_DISABLE
		std::cout << "InMatrix dimension is "<<inShape.height<<","<<inShape.width<<","<<inShape.channels<<"\n";
		std::cout << "OutMatrix dimension is "<<outShape.height<<","<<outShape.width<<","<<outShape.channels<<"\n";
		std::cout << "Input size per thread block: ("<<inputsizeH<<","<<inputsizeH<<","<<inputsizeC<<")\n";
		std::cout << "Shared mem size: "<<sharedmemSize<<"B\n";
		std::cout << "GridX, GridY, GridZ : ("<<gridSizeX<<","<<gridSizeY<< "," << gridSizeZ <<")\n";
		std::cout << "Tile Size: " << TILE_WIDTH << "," << TILE_HEIGHT << "," << TILE_BREADTH <<"\n";
	#endif
	
	dim3 dimGrid(gridSizeX, gridSizeY, gridSizeZ);
	poolLayer_gpu<<<dimGrid,dimBlock,sharedmemSize>>>(inMatrix_d, inShape, outMatrix_d, outShape, poolArgs);

	cudaDeviceSynchronize();

	//Copy back the output to the host memory and free dev mem
	cudaMemcpy(output, outMatrix_d, tensorSize(outShape)*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(inMatrix_d); cudaFree(outMatrix_d);

}

void evaluateGpuFCLayer(float * a, TensorShape aShape, 
	float * b, TensorShape bShape, 
	float * c, TensorShape cShape, 
	GemmLayerArgs args){
		//Copy the tensors to gpu
		float * a_d, * b_d, * c_d;
		cudaMalloc((void **) &a_d, tensorSize(aShape) * sizeof(float) );
		cudaMemcpy(a_d, a, tensorSize(aShape) * sizeof(float), cudaMemcpyHostToDevice );
		cudaMalloc((void **) &b_d, tensorSize(bShape) * sizeof(float) );
		cudaMemcpy(b_d, b, tensorSize(bShape) * sizeof(float), cudaMemcpyHostToDevice );
		cudaMalloc((void **) &c_d, tensorSize(cShape) * sizeof(float) );

		//Define shared memory size
		int a_req = args.tileH * args.tileH;
		int b_req = args.tileW * args.tileH;
		int sharedmemSize = (a_req + b_req) * sizeof(float);

		//Launch the kernel
		dim3 dimBlock(args.tileW, args.tileH, 1);
		int gridSizeY = ceil(1.0f*cShape.height/args.tileH);
		int gridSizeX = ceil(1.0f*cShape.width/args.tileW);
		dim3 dimGrid(gridSizeX, gridSizeY, 1);

		#ifndef DEBUG_PRINT_DISABLE
		std::cout << "GridX,GridY and block dimension: ("<<gridSizeX<<","<<gridSizeY<<" and "<<args.tileW<<","<<args.tileH<<")\n";
		std::cout << "output: " << cShape << "\n";
		#endif
		
		gemmLayer_gpu<<<dimGrid,dimBlock,sharedmemSize>>>(a_d, aShape, b_d, bShape, c_d, cShape, args);

		cudaDeviceSynchronize();
		gpuErrchk(cudaMemcpy(c, c_d, tensorSize(cShape) * sizeof(float), cudaMemcpyDeviceToHost ));
		cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);

}

__global__
void convLayer_gpu (int chanTile, float * input, TensorShape iShape, 
	float * filter, TensorShape fShape, 
	float * bias, float * output, TensorShape oShape, 
	ConvLayerArgs args) {
	
	//Thread and block variables
	int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
	int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;

	//Output variables
	int outRow = by * blockDim.y + ty; //output row
	int outCol = bx * blockDim.x + tx; //output column


	// with shared memory
	/* tz calculates the output in the output channel direction
	 */
	int outChan = bz * blockDim.z + tz; //output channels

	//define shared memory blocks
	extern __shared__ float s[]; //shared memory array
	float *inp_data = s; //pointer to shared memory reserved for input per TB
	int inpChan = chanTile; 
	int inpW = fShape.width + (blockDim.x - 1) * args.strideW; //input shared memory width
	int inpH = fShape.height + (blockDim.y - 1) * args.strideH; //input shared memory height
	int inpSize = inpW * inpH * inpChan; //input shared memory total size
	float *filter_data = &(inp_data)[inpSize]; //pointer to shared memory reserved for filter per TB
	int filterSize = fShape.width * fShape.height * inpChan * blockDim.z; //filter shared memory total size
	float *bias_data = &(filter_data)[filterSize]; //pointer to shared memory reserved for bias per TB

	int inpRowOffset = (by * blockDim.y) * args.strideH - args.padH; //row-offset for actual input (can be negative)
	int inpColOffset = (bx * blockDim.x) * args.strideW - args.padW; //col-offset for actual input (can be negative)
	int subChannels = ceilf(1.0f*iShape.channels/inpChan); //#chunks for input channels required to compute final output value
	if (iShape.channels < inpChan){
		inpChan = iShape.channels;
	}

	int outIndex;
	
	//Load bias to shared memory
	//Reuse across output batch and within an output channel
	if (outRow < oShape.height && outCol < oShape.width && outChan < oShape.channels){
		bias_data[tz] = bias[bz * blockDim.z + tz];		
		for (int batch = 0; batch < iShape.count; ++batch){
			outIndex = (outRow*oShape.width + outCol) + outChan*oShape.width*oShape.height + batch*oShape.channels*oShape.width*oShape.height;
			output[outIndex] = bias_data[tz];	
		}
	}

	
	
	for (int chan = 0; chan < subChannels; ++chan ){ // #inpChan per subChannel
		int inpChanOffset = inpChan*chan; //channel-offset for actual input
		// if (tx == 1 && ty == 7 && bx ==0 && by == 0 && tz == 0 && bz ==0){
		// 	printf("inpChanOffset:%d\n", inpChanOffset);
		// }

		//Load weight to shared memory
		//Reuse across output batch and within an output channel
		int fshared_row, fshared_col, fshared_chan, fshared_count, fil_chan, fil_count;
		for (int fblockChan = 0; fblockChan < inpChan; ++fblockChan){
			for (int fblockRow = 0; fblockRow < ceilf(1.0f*fShape.height/blockDim.y); ++fblockRow){
				for (int fblockCol = 0; fblockCol < ceilf(1.0f*fShape.width/blockDim.x); ++fblockCol){
					fshared_row  = ty + fblockRow*blockDim.y; //shared memory for filter - row
					fshared_col  = tx + fblockCol*blockDim.x; //shared memory for filter - col
					fshared_chan = fblockChan; //shared memory for filter - chan (based on subblock of inpChannels)
					fshared_count = tz; ////shared memory for filter - output channel
					fil_chan    = inpChanOffset + fshared_chan;
					fil_count   = bz * blockDim.z + tz;
					if (fil_chan < iShape.channels && fshared_row < fShape.height && fshared_col < fShape.width && fil_count < fShape.count){
							filter_data[fshared_row*fShape.width + fshared_col + fshared_chan*fShape.width*fShape.height + fshared_count*fShape.width*fShape.height*inpChan]   = filter[fshared_row*fShape.width + fshared_col + fil_chan*fShape.width*fShape.height + fil_count*fShape.width*fShape.height*fShape.channels];
					}
				}
				// if (tx == 1 && ty == 7 && bx ==0 && by == 9 && tz == 0 && bz ==0){
				// 	printf("Chan[%d] | filter_data: %f, FilCount: %d, filChan:%d, filH: %d, filW: %d\n", chan, filter_data[fshared_row*fShape.width + fshared_col + fshared_chan*fShape.width*fShape.height + fshared_count*fShape.width*fShape.height*inpChan], fil_count, fil_chan, fshared_row, fshared_col);
				// 	printf("Chan[%d] | SharedCount: %d, shared_chan:%d, shared_row: %d, shared_col: %d\n", chan, fshared_count, fshared_chan, fshared_row, fshared_col);
				// }
			}
		}

		for (int batch = 0; batch < iShape.count; ++batch){	

			// Load shared mem data for input
			// Reuse across output channels
			int shared_row, shared_col, shared_chan, inp_row, inp_col, inp_chan;
			for (int blockRow = 0; blockRow < ceilf(1.0f*inpH/blockDim.y); ++blockRow){ //row
				shared_row  = ty + blockRow*blockDim.y;
				if (shared_row < inpH){
					for (int blockCol = 0; blockCol < ceilf(1.0f*inpW/blockDim.x); ++blockCol){  //col
						shared_col  = tx + blockCol*blockDim.x;
						if (shared_col < inpW){
							for (int blockChan = 0; blockChan < ceilf(1.0f*inpChan/blockDim.z); ++blockChan){ //channel
								shared_chan = tz + blockChan*blockDim.z;
								if (shared_chan < inpChan){
									inp_chan    = inpChanOffset + shared_chan;
									inp_row     = inpRowOffset + shared_row;
									inp_col     = inpColOffset + shared_col;
									if (inp_chan < iShape.channels){
										if (inp_row < 0 or inp_row > iShape.height - 1 or inp_col < 0 or inp_col > iShape.width - 1){
											inp_data[shared_row*inpW + shared_col + shared_chan*inpW*inpH] = 0;
										}else{
											inp_data[shared_row*inpW + shared_col + shared_chan*inpW*inpH]   = input[inp_row*iShape.width + inp_col + inp_chan*iShape.width*iShape.height + batch*iShape.channels*iShape.width*iShape.height];
										}
									}
									// if ((inp_chan < iShape.channels) && (inp_row > -1) && (inp_row < iShape.height) && (inp_col > -1) && (inp_col < iShape.width)){
									// 	inp_data[shared_row*inpW + shared_col + shared_chan*inpW*inpH]   = input[inp_row*iShape.width + inp_col + inp_chan*iShape.width*iShape.height];
									// 	// inp_data[shared_row*inpW + shared_col + shared_chan*inpW*inpH]   = 1.0;
									// }	
									// if (tx == 1 && ty == 7 && bx ==0 && by == 9 && tz == 0 && bz ==0){
									// 	printf("[0]: %f\n",inp_data[0]);
									// 	printf("inpChan:%d, inpH: %d, inpW: %d, inpRowOffset: %d, inpColOffset: %d, inpChanOffset: %d\n", inpChan, inpH, inpW, inpRowOffset, inpColOffset, inpChanOffset);
									// 	printf("inp_data: %f, shared_chan:%d, shared_row: %d, shared_col: %d\n", inp_data[shared_row*inpW + shared_col + shared_chan*inpW*inpH], shared_chan, shared_row, shared_col);
									// 	printf("input[%d]: %f, inp_chan:%d, inp_row: %d, inp_col: %d\n",inp_row*iShape.width + inp_col + inp_chan*iShape.width*iShape.height, input[inp_row*iShape.width + inp_col + inp_chan*iShape.width*iShape.height],inp_chan, inp_row, inp_col);
									// 	printf ("Chan[%d] | %d, %d, %d, %d, %d\n",chan, (inp_chan < iShape.channels), (inp_row > -1),(inp_row < iShape.height),(inp_col > -1),(inp_col < iShape.width) );
									// }
								}
							}		
						}
					}			
				}
			}

		
			__syncthreads();

			//Calculate output
			if (outRow < oShape.height && outCol < oShape.width && outChan < oShape.channels){
				for (int iCh = 0; iCh < inpChan; ++iCh){
					int iRowOffset = ty * args.strideH;
					int iColOffset = tx * args.strideW;
					for (int r = 0; r < fShape.height; ++r){
						for (int c = 0; c < fShape.width; ++c){
							int iIdx = (iRowOffset + r) * inpW + iColOffset + c + iCh * inpW * inpH;
							int fIdx = r * fShape.width + c + iCh * fShape.width * fShape.height + tz * fShape.width * fShape.height * inpChan;
							output[outIndex] += inp_data[iIdx] * filter_data[fIdx];
							// if (tx == 0 && ty == 0 && bx ==0 && by == 0 && tz == 0 && bz ==0){
							// 	printf("Chan[%d] | inp (%d): %f, filter(%d): %f and output(%d): %f\n", chan, iIdx, inp_data[iIdx], fIdx, filter_data[fIdx], (outRow*oShape.width + outCol) + outChan*oShape.width*oShape.height, output[(outRow*oShape.width + outCol) + outChan*oShape.width*oShape.height]);
							// }
							
						}
					}
				}
			}
		}
	}
}

__global__
void poolLayer_gpu (float * input, TensorShape inShape,
	float * output, TensorShape outShape, PoolLayerArgs args){		

	extern __shared__ float shared_data[];

	// printf("I am in kernel fucntion\n");

	int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
	int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;
	int outRow = by * blockDim.y + ty;
	int outCol = bx * blockDim.x + tx;
	int outChan = bz * blockDim.z + tz;
	// int outWidth, outHeight;
	int inpWidth, inpHeight;
	
	//Get the input tile size
	if (outShape.width < blockDim.x)
		inpWidth = (outShape.width - 1)*args.strideW + args.poolW;
	else
		inpWidth = (blockDim.x - 1)*args.strideW + args.poolW;
	
	if (outShape.height < blockDim.y)
		inpHeight = (outShape.height - 1)*args.strideH + args.poolH;
	else
		inpHeight = (blockDim.y - 1)*args.strideH + args.poolH;

	for (int batch = 0; batch < inShape.count; ++batch){

		//load the shared memory
		int iterTotal = ceilf((1.0f*inpWidth*inpHeight)/(blockDim.x*blockDim.y));
		int index, inpRow, inpCol, inpIndex, inpChan;
		inpChan = outChan;
		for (int idx = 0; idx < iterTotal; ++idx){
			index = idx*blockDim.x*blockDim.y + ty*blockDim.x + tx;
			inpRow = by * blockDim.y * args.strideH + (index / inpWidth);
			inpCol = bx * blockDim.x * args.strideW + index % inpHeight;
			inpIndex = inpCol + inpRow* inShape.width + inpChan * inShape.width * inShape.height + batch * inShape.width * inShape.height * inShape.channels;
			if (index < inpWidth*inpHeight){
				if (inpRow < inShape.height and inpCol < inShape.width and inpChan < inShape.channels){
					shared_data[index + tz*inpWidth*inpHeight] = input[inpIndex];
				}
			}
		}
		if (ty == 0 and tx == 0 and tz ==0 and bx == 0 and by == 0 and bz == 1){
			printf("iterTotal: %d \n", iterTotal);
			printf("Number of threads in TB: %d \n", blockDim.x*blockDim.y*blockDim.z);
			printf("Input size: %d,%d,%d\n", inpWidth,inpHeight,blockDim.z);
			printf("shared memory first element: %f and channel: %d\n", shared_data[0+inpWidth*inpHeight],inpChan);
			printf("input first element: %f \n", input[0]);
		}
		
		__syncthreads();
	
		//Calculate max and store in output
		int inpOffsetY, inpOffsetX, indX, indY, inC;
		float poolPick;
		float rival;
		if (outRow < outShape.height and outCol < outShape.width and outChan < outShape.channels){ //corner cases
			inpOffsetY = tx*args.strideW;
			inpOffsetX = ty*args.strideH;
			inC = tz;
			poolPick = 0;
			for (int x = 0; x < args.poolH; ++x){
				for (int y = 0; y < args.poolW; ++y){
	
					indX = inpOffsetX + x;
					indY = inpOffsetY + y;
					rival = shared_data[inC*inpWidth*inpHeight + indX*inpWidth + indY];
					if (ty == 0 and tx == 0 and tz ==1 and bx == 0 and by == 0 and bz == 0){
						printf("outRow,Col,Chan: (%d,%d,%d), indX,indY: (%d,%d) and Data: %f\n",outRow,outCol,outChan,indX,indY,rival);
					}
					if (poolPick < rival){
						poolPick = rival;
					}
				}
			}
			output[(outRow*outShape.width + outCol) + outShape.width*outShape.height*outChan + outShape.width*outShape.height*outShape.channels*batch] = poolPick;
			// if (ty == 0 and tx == 1 and tz == 3 and bx == 0 and by == 0 and bz == 1){
			// 	printf("Output element: %f \n", output[(outRow*outShape.width + outCol) + outShape.width*outShape.height*outChan + outShape.width*outShape.height*outShape.channels*batch]);
			// 	printf("Output Index: (%d,%d, %d,%d)\n", outRow, outCol, outChan, batch);
			// 	printf("Input shared mem Index: (%d,%d)\n", inpOffsetX, inpOffsetY);
			// 	printf("pool picked value: %f\n", poolPick);
			// }
	
		}
	}

}

__global__
void gemmLayer_gpu (float * a, TensorShape aShape, 
	float * b, TensorShape bShape,
	float * c, TensorShape cShape,
	GemmLayerArgs args){

	// Calculate the output indices
	int tx = threadIdx.x; int ty = threadIdx.y;
	int bx = blockIdx.x; int by = blockIdx.y;
	int outRow = by * blockDim.y + ty;
	int outCol = bx * blockDim.x + tx;

	int subTilesAlongK = ceilf(1.0f*aShape.width/args.tileH);

	#ifndef DEBUG_PRINT_DISABLE
		if (tx == 0 and ty ==0 and bx ==0 and by == 0){
			printf("subtilesAlongK: %d and the max value: %d\n", subTilesAlongK, (subTilesAlongK - 1)* args.tileH + args.tileH - 1);
			printf("A[0], B[0]: %f, %f\n", a[0],b[0]);
		}
	#endif

	//Allocate shared memory space
	extern __shared__ float s[];
	float * a_data  = s;
	float * b_data = &(a_data)[args.tileH * args.tileH];

	//Variable storing partial sum from each subtile
	float Pvalue = 0;
	
	for (int subTile = 0; subTile < subTilesAlongK; ++ subTile){
		int aOffsetRow = by * blockDim.y;
		int aOffsetCol = subTile * args.tileH;
		int bOffsetRow = subTile * args.tileH;
		int bOffsetCol = bx * blockDim.x;
		
		//Load a
		int iter = ceilf(1.0f*args.tileH/args.tileW); //assuming tileH > tileW
		for (int idx = 0; idx < iter; ++idx){
			int col = tx + idx*args.tileW;
			if (col < args.tileH){
				if ((aOffsetRow + ty) < aShape.height and (aOffsetCol + col) < aShape.width){
					a_data[ty * args.tileH + col] = a[(aOffsetRow + ty) * aShape.width + aOffsetCol + col];
				}
			}
		}

		//Load b
		int bShindex = ty * args.tileW + tx;
		int bIndex   = (bOffsetRow + ty) * bShape.width + bOffsetCol + tx;
		if ((bOffsetRow + ty) < bShape.height && (bOffsetCol + tx) < bShape.width){
			b_data[bShindex] = b[bIndex];
		}
		__syncthreads();
		
		//Calculate pvalue
		if (outRow < cShape.height && outCol < cShape.width){
			for (int i = 0; i < args.tileH; ++i){
				Pvalue += a_data[ty*args.tileH + i] * b_data[i*args.tileW + tx];
				#ifndef DEBUG_PRINT_DISABLE
				if (subTile == 255){	
					if (bx == 62 && by == 0 && ty == 0 && tx == 0){
						// printf("%d,%d,%d,%d: b shared index: %d and b index: %d\n",bx,tx,by,ty, bShindex, bIndex);
						// printf("offsetrow: %d and offsetcol: %d\n",bOffsetRow, bOffsetCol);
						// printf("%d,%d,%d,%d: b_data: %f, b: %f, index: %d\n",bx,tx,by,ty,b_data[bShindex],b[bIndex], bIndex);
						printf("%d (subtile),%d,%d,%d,%d: b_data: %f (%d), a_data: %f, Pvalue: %f\n",subTile, bx,tx,by,ty,b_data[i*args.tileW + tx],i*args.tileW + tx,a_data[ty*args.tileH + i],Pvalue );
					}
				}
				#endif
			}
		}	
	}

	if (outRow < cShape.height && outCol < cShape.width){
		//assign the total sum to output
		c[outRow*cShape.width + outCol] = Pvalue;
		#ifndef DEBUG_PRINT_DISABLE
			if (bx == 62 and by == 0){
				printf("%d,%d,%d,%d: pvalue: %f, %d, output: %f\n",bx,tx,by,ty,Pvalue,outRow*cShape.width + outCol, c[outRow*cShape.width + outCol]);
			}
		#endif
	}	

}
