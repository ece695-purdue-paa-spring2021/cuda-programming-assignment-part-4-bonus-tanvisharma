/**
 * @file alexnet.cu
 * @author Tanvi Sharma (sharm418@purdue.edu)
 * @brief 
 * @version 0.1
 * @date 2021-05-04
 * 
 * @copyright Copyright (c) 2021
 * 
 */

 #include <iostream>
 #include "cpuLib.h"
 #include "cudaLib.cuh"
 
 int main(int argc, char** argv) {
  cudaFreeHost(0);
   #ifndef DEBUG_PRINT_DISABLE
    std::cout << "AlexNet Implementation on CUDA \n";
   #endif
    runAlexNet(argc, argv);
 
 }
 
 
 