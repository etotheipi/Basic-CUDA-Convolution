/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>
#include <stopwatch.h>
#include <cmath>
#include "convolution_kernel.cu"

using namespace std;


// Assume target memory has already been allocated, nPixels is odd
void createGaussian1D(float* targPtr, 
                      int    nPixels, 
                      float  sigma, 
                      float  ctr=0.0f);

// Assume target memory has already been allocate, nPixels is odd
void createGaussian2D(float* targPtr, 
                      int    nPixelsCol,
                      int    nPixelsRow,
                      float  sigmaCol,
                      float  sigmaRow,
                      float  ctrCol=0.0f,
                      float  ctrRow=0.0f);

// Assume diameter^2 target memory has already been allocated
void createBinaryCircle(float* targPtr,
                        int    diameter); 

// Assume diameter^2 target memory has already been allocated
void createBinaryCircle(int* targPtr,
                        int  diameter); 

////////////////////////////////////////////////////////////////////////////////
//
// Program main
//
// TODO:  Remove the CUTIL calls so libcutil is not required to compile/run
//
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
   cout << endl << "Executing GPU-accelerated convolution..." << endl;

   /////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////
   // Query the devices on the system and select the fastest
   int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
   {
		cout << "cudaGetDeviceCount() FAILED." << endl;
      cout << "CUDA Driver and Runtime version may be mismatched.\n";
      return -1;
	}

   // Check to make sure we have at least on CUDA-capable device
   if( deviceCount == 0)
   {
      cout << "No CUDA devices available.  Exiting." << endl;
      return -1;
	}

   // Fastest device automatically selected.  Can override below
   int fastestDeviceID = cutGetMaxGflopsDeviceId() ;
   //fastestDeviceID = 0;
   cudaSetDevice(fastestDeviceID);

   cudaDeviceProp gpuProp;
   cout << "CUDA-enabled devices on this system:  " << deviceCount <<  endl;
   for(int dev=0; dev<deviceCount; dev++)
   {
      cudaGetDeviceProperties(&gpuProp, dev); 
      char* devName = gpuProp.name;
      int mjr = gpuProp.major;
      int mnr = gpuProp.minor;
      if( dev==fastestDeviceID )
         cout << "\t* ";
      else
         cout << "\t  ";

      printf("(%d) %20s \tCUDA Capability %d.%d\n", dev, devName, mjr, mnr);
   }
   // End of CUDA device query & selection
   /////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////


   // I've conceded, we better just do everything in COL-MAJOR 
   // Use 17x17 because it's bigger than the block size, so we stress
   // the COPY_LIN_ARRAY_TO_SHMEM macro looping
   unsigned int imgW  = 512;
   unsigned int imgH  = 512;
   unsigned int psfW  = 17;
   unsigned int psfH  = 17;
   unsigned int nPix  = imgH*imgW;
   unsigned int nPsf  = psfW*psfH;
   // I would've expected 32x8 to reduce bank conflicts, but 8x32 is 
   // about 30% faster, and both are much faster than 16x16
   unsigned int blockDimX = 8;              // X ~ COL
   unsigned int blockDimY = 32;              // Y ~ ROW
   unsigned int gridDimX = imgW/blockDimX;   // X ~ COL
   unsigned int gridDimY = imgH/blockDimY;   // Y ~ ROW

   unsigned int imgBytes = nPix*FLOAT_SZ;
   unsigned int psfBytes = nPsf*FLOAT_SZ;

   cout << endl;
   printf("Executing convolution on %dx%d image with %dx%d PSF\n", imgW,imgH,psfW,psfH);

   // Allocate host-side memory
   float* imgIn  = (float*)malloc(imgBytes);
   float* imgOut = (float*)malloc(imgBytes);
   float* imgPsf = (float*)malloc(psfBytes);


   // Read the [very large] image
   // Data is stored in row-major, so reverse the loop to read in correctly
   // col-major, so we reverse the order of loops
   ifstream fpIn( "salt512.txt", ios::in);
   cout << "Reading image from file..." << endl;
   for(int r=0; r<imgH; r++)
      for(int c=0; c<imgH; c++)
          fpIn >> imgIn[c*imgH+r];
   fpIn.close();


   // Read in the PSF from a file
   // Here we use a highly asymmetric PSF so we can verify coordinate systems
   // PSF is read in as row-major data, yet we do all our processing in
   // col-major, so we reverse the order of loops

   //createGaussian2D(imgPsf, psfW, psfH, (float)psfW/5.5, (float)psfH/5.5);
   //createBinaryCircle(imgPsf, psfW);
   //ifstream fpPsf("se_3x3_asymm.txt", ios::in);
   //ifstream fpPsf("asymmPSF_15x15.txt", ios::in);
   ifstream fpPsf("asymmPSF_17x17.txt", ios::in);
   //ifstream fpPsf("asymmPSF_25x25.txt", ios::in);
   //cout << endl << "Point Spread Function: " << endl;
   for(int r=0; r<psfH; r++)
   {
      for(int c=0; c<psfW; c++)
      {
         fpPsf >> imgPsf[c*psfH+r];
      }
   }

   // Write out the PSF so can be checked later
   ofstream psfout("psf.txt", ios::out);
   cout << endl << "Point Spread Function:" << endl;
   for(int r=0; r<psfH; r++)
   {
      cout << "\t";
      for(int c=0; c<psfW; c++)
      {
         printf("%0.3f ", imgPsf[c*psfH+r]);
         psfout << imgPsf[c*psfH+r] << " ";
      }
      cout << endl;
      psfout << endl;
   }
   cout << endl;

   // Allocate device memory and copy data to it
   float* devIn; 
   float* devOut; 
   float* devPsf;
   cudaMalloc((void**)&devIn,  imgBytes);
   cudaMalloc((void**)&devOut, imgBytes);
   cudaMalloc((void**)&devPsf, psfBytes);

   dim3  GRID(  gridDimX,  gridDimY,   1);
   dim3  BLOCK( blockDimX, blockDimY,  1);
   printf("Grid  Dimensions: (%d, %d)\n", gridDimX,  gridDimY);
   printf("Block Dimensions: (%d, %d)\n\n", blockDimX, blockDimY);

   // GPU Timer Functions
   unsigned int timer = 0;
   cutilCheckError( cutCreateTimer( &timer));
   cutilCheckError( cutStartTimer( timer));

   cudaMemcpy(devIn,  imgIn,   imgBytes, cudaMemcpyHostToDevice);
   cudaMemcpy(devPsf, imgPsf,  psfBytes, cudaMemcpyHostToDevice);


   // Set up kernel execution geometry
   // **************************************************************************
   // The data is on the HOST, do a full round-trip calculation w/ mem copies
   cout << "Data loaded into HOST mem, executing kernel..." << endl;
   convolveBasic<<<GRID, BLOCK>>>(devIn, devOut, devPsf, 
                                                imgW, imgH, psfW/2, psfH/2);
   cutilCheckMsg("Kernel execution failed");  // Check if kernel exec failed
   cudaThreadSynchronize();

   // Copy result from device to host
   cudaMemcpy(imgOut, devOut, nPix*sizeof(float), cudaMemcpyDeviceToHost);

   cutilCheckError( cutStopTimer( timer));
   float gpuTime_w_copy = cutGetTimerValue(timer);
   cutilCheckError( cutDeleteTimer( timer));
   // **************************************************************************


   // **************************************************************************
   // With data already on the device, time just the computations over 100 runs
   int NITER = 20;
   printf("Data already on DEVICE, running %d times...\n", NITER);
   cutilCheckError( cutCreateTimer( &timer));
   cutilCheckError( cutStartTimer( timer));
   for(int i=0; i<NITER; i++)
   {
      convolveBasic<<<GRID, BLOCK>>>(devIn, devOut, devPsf, 
                                                   imgW, imgH, psfW/2, psfH/2);
      cutilCheckMsg("Kernel execution failed");  // Check if kernel exec failed
      cudaThreadSynchronize();
   }
   cutilCheckError( cutStopTimer( timer));
   float gpuTime_compute_only = cutGetTimerValue(timer)/(float)NITER;
   cutilCheckError( cutDeleteTimer( timer));
   // **************************************************************************

   float memCopyTime = gpuTime_w_copy - gpuTime_compute_only;
   float cpySpeed = 2.0 * (float)imgBytes / (float)(memCopyTime/1000. * 1024 * 1024);
   cout << "Final Timing Results:" << endl << endl;
   printf("\tMemory copies            :  %.3f ms (%0.1f MB/s)\n", memCopyTime, cpySpeed);
   printf("\tRaw computation time     :  %.3f ms\n", gpuTime_compute_only);
   printf("\t-----------------------------------\n");
   printf("\tTotal time w/ mem copies :  %.3f ms\n\n", gpuTime_w_copy);


   // Writing I/O data to file, so I can check it with MATLAB
   // Again, data is stored col-major, but files are r/w in row-major
   // so we reverse the loops
   cout << "Writing before/after image to file..." << endl;
   ofstream fpOrig("origImage.txt", ios::out);
   ofstream fpOut("gpu_solution.txt", ios::out);
   // Write fpOut
   for(int r=0; r<imgH; r++)
   {
      for(int c=0; c<imgW; c++)
      {
         fpOrig << imgIn[c*imgW+r] << " ";
         fpOut << imgOut[c*imgW+r] << " ";
      }
      fpOrig << endl;
      fpOut << endl;
   }
   fpOrig.close();
   fpOut.close();




   /*
   // We'll use an gaussian "PSF" for delta intensity in bilateral-filter 
   unsigned int ipsfBytes = 256*2*FLOAT_SZ;
   float* imgPsfIntens  = (float*)malloc(ipsfBytes);
   float* devPsfIntens;
   cudaMalloc((void**)&devPsfIntens, ipsfBytes);
   cudaMemcpy(devPsfIntens,  imgPsfIntens,  ipsfBytes, cudaMemcpyHostToDevice);

   // Create the gaussian PSF, or replace with code block below to read file
   float sigmaSq = 9;
   unsigned int psfradW  = psfW/2;
   unsigned int psfradH  = psfH/2;
   float mult = 1.0/sqrt(2.0*M_PI*sigmaSq);
   for(int c=-psfradW; c<psfradW; c++)
   {
      for(int r=-psfradH; r<psfradH; r++)
      {
         int distsq = c*c+r*r;
         int psfIdx = IDX_1D(c+psfradW,r+psfradH,psfH);
         imgPsf[psfIdx] = mult * exp(-0.5 * distsq / sigmaSq);
      }
   }


   isigmaSq = 400;
   mult = 1.0/sqrt(2.0*M_PI*isigmaSq);
   for(int di=-256; di<256; di++)
      imgPsfIntens[di+256] = mult * exp(-0.5 * di * di / isigmaSq);


   cutilCheckError( cutCreateTimer( &timer));
   cutilCheckError( cutStartTimer( timer));
   for(int i=0; i<100; i++)
   {
      bilateralConvolve<<<GRID, BLOCK>>>(devIn, devOut, devPsf, imgW, imgH, psfW/2);
      cudaThreadSynchronize();
   }
   */

   // **************************************************************************
   // Now execute the same procedure on the CPU
   // **************************************************************************
/*
   cout << "Now executing the exact same convolution on the CPU" << endl;

   unsigned int cutTimer;
   cutCreateTimer(&cutTimer);
   cutStartTimer(cutTimer);

   for(int i=0; i<imgW*imgH; i++)
      imgOut[i] = 0.0f;

   int psfRad = psfW/2;
   int colOff, rowOff;
   int psfIdx, pixIdx;
   for(int c=0; c<imgH; c++)
   {
      for(int r=0; r<imgW; r++)
      {
         float & thisPixel = imgOut[c*imgH+r];
         for(int coff=-psfRad; coff<=psfRad; coff++)
         {
            for(int roff=-psfRad; roff<=psfRad; roff++)
            {
               colOff = c + coff; 
               rowOff = r + roff; 
               if(colOff>=0 && rowOff>=0 && colOff<imgH && rowOff<imgW)
               {
                  psfIdx = IDX_1D(psfRad-coff, psfRad-roff, psfW);
                  pixIdx = IDX_1D(colOff, rowOff, imgH);
                  thisPixel += imgPsf[psfIdx] * imgIn[pixIdx];
               }
            }
         }
      }
   }
   
   cutStopTimer(cutTimer);
   float cpuTime = cutGetTimerValue(cutTimer);
   printf("CPU Processing time: %f (ms)\n", cpuTime);
   
   

   ofstream cpuOut("cpu_solution.txt", ios::out);
   // Write fpOut
   for(int c=0; c<imgW; c++)
   {
      for(int r=0; r<imgH; r++)
      {
         cpuOut << imgOut[c*imgW+r] << " ";
      }
      cpuOut << endl;
   }

   cout << endl << endl;
   cout << "Final Timing Results:" << endl;
   cout << "\tGPU Time:  " << gpuTime/1000. << " seconds" << endl;
   cout << "\tCPU Time:  " << cpuTime/1000. << " seconds" << endl;
   cout << "\tSpeed Up:  " << cpuTime/gpuTime << endl;
*/

   // cleanup memory
   free(imgIn);
   free(imgOut);
   free(imgPsf);
   //free(imgPsfIntens);
   cudaFree(devIn);
   cudaFree(devOut);
   cudaFree(devPsf);

   cudaThreadExit();

   //cutilExit(argc, argv);
}

// Assume target memory has already been allocated, nPixels is odd
void createGaussian1D(float* targPtr, 
                      int    nPixels, 
                      float  sigma, 
                      float  ctr)
{
   if(nPixels%2 != 1)
   {
      cout << "***Warning: createGaussian(...) only defined for odd pixel"  << endl;
      cout << "            dimensions.  Undefined behavior for even sizes." << endl;
   }

   float pxCtr = (float)(nPixels/2 + ctr);   
   float sigmaSq = sigma*sigma;
   float denom = sqrt(2*M_PI*sigmaSq);
   float dist;
   for(int i=0; i<nPixels; i++)
   {
      dist = (float)i - pxCtr;
      targPtr[i] = exp(-0.5 * dist * dist / sigmaSq) / denom;
   }
}

// Assume target memory has already been allocate, nPixels is odd
// Use col-row (D00_UL_ES)
void createGaussian2D(float* targPtr, 
                      int    nPixelsCol,
                      int    nPixelsRow,
                      float  sigmaCol,
                      float  sigmaRow,
                      float  ctrCol,
                      float  ctrRow)
{
   if(nPixelsCol%2 != 1 || nPixelsRow != 1)
   {
      cout << "***Warning: createGaussian(...) only defined for odd pixel"  << endl;
      cout << "            dimensions.  Undefined behavior for even sizes." << endl;
   }

   float pxCtrCol = (float)(nPixelsCol/2 + ctrCol);   
   float pxCtrRow = (float)(nPixelsRow/2 + ctrRow);   
   float distCol, distRow, distColSqNorm, distRowSqNorm;
   float denom = 2*M_PI*sigmaCol*sigmaRow;
   for(int c=0; c<nPixelsCol; c++)
   {
      distCol = (float)c - pxCtrCol;
      distColSqNorm = distCol*distCol / (sigmaCol*sigmaCol);
      for(int r=0; r<nPixelsRow; r++)
      {
         distRow = (float)r - pxCtrRow;
         distRowSqNorm = distRow*distRow / (sigmaRow*sigmaRow);
         
         targPtr[c*nPixelsRow+r] = exp(-0.5*(distColSqNorm + distRowSqNorm)) / denom;
      }
   }
}

// Assume diameter^2 target memory has already been allocated
void createBinaryCircle(float* targPtr,
                        int  diameter)
{
   float pxCtr = (float)(diameter-1) / 2.0f;
   float rad;
   for(int c=0; c<diameter; c++)
   {
      for(int r=0; r<diameter; r++)
      {
         rad = sqrt((c-pxCtr)*(c-pxCtr) + (r-pxCtr)*(r-pxCtr));
         if(rad <= pxCtr+0.5)
            targPtr[c*diameter+r] = 1.0f;
         else
            targPtr[c*diameter+r] = 0.0f;
      }
   }
}

// Assume diameter^2 target memory has already been allocated
void createBinaryCircle(int* targPtr,
                        int    diameter)
{
   float pxCtr = (float)(diameter-1) / 2.0f;
   float rad;
   for(int c=0; c<diameter; c++)
   {
      for(int r=0; r<diameter; r++)
      {
         rad = sqrt((c-pxCtr)*(c-pxCtr) + (r-pxCtr)*(r-pxCtr));
         if(rad <= pxCtr+0.5)
            targPtr[c*diameter+r] = 1;
         else
            targPtr[c*diameter+r] = 0;
      }
   }
}
