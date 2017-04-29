/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, Itseez Inc, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Itseez Inc or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __DPM_FEATURE_GPU__
#define __DPM_FEATURE_GPU__

#include "opencv2/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/gpu/gpumat.hpp"

#include <string>
#include <vector>
#include <cstdio>

namespace cv
{
namespace dpm
{

/** @brief This class contains DPM model parameters
 */

class FeatureGPUParams 
{
    public:
        int sbin; 
        int interval; 
        int maxScale; 
        int pad_x; 
        int pad_y; 
        float sfactor; 

        FeatureGPUParams()
        {
            sbin = 8; 
            interval = 5;
            maxScale = 10; 
            pad_x = 0; 
            pad_y = 0; 
            sfactor = 1.14869; 
        }
};

class FeatureGPU
{
    public:
        FeatureGPUParams params; 

        gpu::GpuMat origImage;
        Size origSize;

        std::vector<gpu::GpuMat> gpuImage; 
        std::vector<gpu::GpuMat> gpuHist; 
        std::vector<gpu::GpuMat> gpuNorm;
        std::vector<gpu::GpuMat> gpuFeat; 
        std::vector<float>       scales; 
        std::vector<Size>        scaleSizes; 

        // constructors of the Feature GPU calculator class
	FeatureGPU(const Size &size);
	// FeatureGPU(const Mat &image);
	// FeatureGPU(const gpu::GpuMat &image);

        void initialize(const FeatureGPUParams params = FeatureGPUParams()); 

        // load a new image to Mat & image, we need to create new Hist, Norm and Feat matrices. 
        void loadImage(const Mat &image); 
        void loadImage(const gpu::GpuMat &gpuImage_); 

        FeatureGPUParams getParams(); 

        void setParams(const FeatureGPUParams &p); 

        // compute all scales, saving it to scales; 
        // use in loadImage() 
        void computeScale(); 
        
        // initialize all gpuHist, gpuImage, gpuNorm, gpuFeat matrices
        // use in constuctor
        // scale should be known
        void initMats(); 
        
        // reset all Hist, Norm, Features to zero 
        // use in loadImage
        void resetMats(); 
        
        // resize images using gpu::resize function
        // use in loadImage, after computeScale()
        void createPyramid(); 

        // calculateHistograms
        void computeHistPyramid();

        // save GPU matrices to host memory. 
        void downloadFeature(std::vector<Mat> &feature); 
        void downloadHist(std::vector<Mat> &hist); 
        void downloadNorm(std::vector<Mat> &norm); 
        
        // GPU wrapper
        void computeHOG32D(int i, const int sbin,
                const int pad_x, const int pad_y);

        // for debug
        void computeImage(const Mat &img, Mat &hist, 
                int sbin, int pad_x, int pad_y);
};

} // namespace dpm
} // namespace cv

#endif // __DPM_FEATURE_GPU__
