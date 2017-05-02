#include "dpm_feature_gpu.hpp"

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>

#include <cuda_runtime.h>
#include <device_functions.hpp>

namespace cv
{
namespace dpm
{

static const int numOrient = 18;
static const int dimHOG = 32;
static __constant__ float eps = 0.0001;
static __constant__ float uu[9] = {1.000, 0.9397, 0.7660, 0.5000, 0.1736, -0.1736, -0.5000, -0.7660, -0.9397}; 
static __constant__ float vv[9] = {0.000, 0.3420, 0.6428, 0.8660, 0.9848,  0.9848,  0.8660,  0.6428,  0.3420};

__global__ void computeHOG32DHist
(const gpu::PtrStepSz<float3> imageM, gpu::PtrStepSzf histM,
 const int sbin, const int pad_x, const int pad_y)
{
    // in a block
    const int block_r = blockIdx.y;
    const int block_c = blockIdx.x; 
    const int pixel_id = threadIdx.x; 
    const int pixel_r  = pixel_id / sbin; 
    const int pixel_c = pixel_id % sbin; 
    
    const int y = block_r * sbin + int (pixel_id / sbin); 
    const int x = block_c * sbin + pixel_id % sbin; 

    const int bW = gridDim.x;
    const int bH = gridDim.y;

    const int oW = fmaxf(bW-2, 0) + 2*pad_x;
    const int oH = fmaxf(bH-2, 0) + 2*pad_y; 

    const int vW = bW * sbin;
    const int vH = bH * sbin;

    // TODO: copy data to shared memory
    const float3* sLast = y >= 1 ? (float3*)(imageM.ptr(y - 1) + MIN(x, imageM.cols - 2)) : NULL;
    const float3* s = (float3*)(imageM.ptr(y) + MIN(x, imageM.cols - 2));
    const float3* sNext = y < vH - 1 ? (float3*)(imageM.ptr(y + 1) + MIN(x, imageM.cols - 2)) : NULL;
    const size_t imStride = imageM.ptr(1) - imageM.ptr(0);
    
    float dyb, dxb, vb;
    float dyg, dxg, vg;
    float dyr, dxr, vr;
    float dy,  dx,  v;

    if (y < 1 || y >= vH - 1 || x < 1 || x >= vW - 1)
    	dy = dx = v = 0;
    else
    {
        // blue image channel ;
        dyb = (sNext)->x - (sLast)->x;
        dxb = (s+1)->x - (s-1)->x;
        vb = dxb*dxb + dyb*dyb;

        // green image channel
        dyg = (sNext)->y - (sLast)->y;
        dxg = (s+1)->y - (s-1)->y;
        vg = dxg*dxg + dyg*dyg;

        // red image channel
        dyr = (sNext)->z - (sLast)->z;
        dxr = (s+1)->z - (s-1)->z;
        vr = dxr*dxr + dyr*dyr;

        // pick the channel with the strongest gradient
        if (vr > v) { v = vr; dx = dxr; dy = dyr; }
        if (vg > v) { v = vg; dx = dxg; dy = dyg; }
        if (vb > v) { v = vb; dx = dxb; dy = dyb; }
    }

    // snap to one of the 18 orientations
    float best_dot = -1000;
    int best_o = 0;
    for (int o = 0; o < (int)numOrient/2; o++)
    {
        float dot = uu[o]*dx + vv[o]*dy;
        if (dot > best_dot)
        {
            best_dot = dot;
            best_o = o;
        }
        if (-dot > best_dot)
        {
            best_dot = -dot;
            best_o = o + (int)(numOrient/2);
        }
    }

    // add to 4 historgrams around pixel using bilinear interpolation
    float vy0 = (float)(pixel_id / sbin) / sbin;
    int tmp = pixel_id % sbin;
    float vx0 = (float) (tmp) / sbin;
    float vy1 = 1.0 - vy0;
    float vx1 = 1.0 - vx0;
    v = sqrtf(v);

    __shared__ float addHist0[64][numOrient], addHist1[64][numOrient], 
                     addHist2[64][numOrient], addHist3[64][numOrient];

    for (int i = 0; i < numOrient; i++)
    {
        addHist0[pixel_id][i] = 0; 
        addHist1[pixel_id][i] = 0; 
        addHist2[pixel_id][i] = 0; 
        addHist3[pixel_id][i] = 0; 
    }

    addHist0[pixel_id][best_o] = vy1*vx1*v; 
    addHist1[pixel_id][best_o] = vx0*vy1*v; 
    addHist2[pixel_id][best_o] = vy0*vx1*v; 
    addHist3[pixel_id][best_o] =  vy0*vx0*v;
    __syncthreads(); 

    if (pixel_id)
    {
        atomicAdd(&addHist0[0][best_o], vy1*vx1*v); 
        atomicAdd(&addHist1[0][best_o], vy1*vx0*v); 
        atomicAdd(&addHist2[0][best_o], vy0*vx1*v); 
        atomicAdd(&addHist3[0][best_o], vy0*vx0*v); 
    }
    __syncthreads(); 

    
    if (pixel_id == 0)
    {
	for (int i = 0; i < numOrient; i++)
            atomicAdd(histM.ptr(block_r) + block_c * numOrient + i, addHist0[0][i]);
        if (block_c + 1 < bW)
            for (int i = 0; i < numOrient; i++)          
                atomicAdd(histM.ptr(block_r) + (block_c + 1) * numOrient + i, addHist1[0][i]);
        if (block_r + 1 < bH)
        {
            for (int i = 0; i < numOrient; i++)
                atomicAdd(histM.ptr(block_r + 1) + block_c * numOrient + i, addHist2[0][i]);
            if (block_c + 1 < bW)
                for (int i = 0 ; i < numOrient; i++)    
                    atomicAdd(histM.ptr(block_r + 1) + (block_c + 1) * numOrient + i, addHist3[0][i]); 
        }
    }
    __syncthreads(); 
}

__global__ void computeHOG32DNorm
(const gpu::PtrStepSzf histM, gpu::PtrStepSzf normM)
{
    int row = blockIdx.x; 
    int col = threadIdx.x; 
    float* dst = normM.ptr(row) + col;
    const float* src = histM.ptr(row) + col * numOrient;
    float sum = 0; 
    for (int i = 0; i < numOrient/2; i++) {
        float tmp = *(src) + *(src + numOrient/2); 
        sum += tmp * tmp;  
        src++; 
    }
    *dst = sum; 
}


__global__ void computeHOG32DFeat
(const gpu::PtrStepSzf histM, const gpu::PtrStepSzf normM,
		gpu::PtrStepSzf featM, int pad_x, int pad_y)
{
    const int oW = featM.cols;
    const int oH = featM.rows;

    int row = blockIdx.x; 
    int col = threadIdx.x; 
    float *dst = featM.ptr(row) + col * dimHOG; 

    if (row < pad_y || row >= featM.rows - pad_y || 
        col < pad_x || col >= featM.cols / dimHOG - pad_x)
    {
        *(dst + dimHOG - 1) = 1; 
        return;
    }
    
    const float *p1 = normM.ptr(row - pad_y) + (col - pad_x);
    const float *p4 = normM.ptr(row - pad_y + 1) + (col - pad_x);
    const float *p7 = normM.ptr(row - pad_y + 2) + (col - pad_x);
    const float *p2 = p1 + 1, *p3 = p1 + 2;
    const float *p5 = p4 + 1, *p6 = p4 + 2;
    const float *p8 = p7 + 1, *p9 = p7 + 2;

    float n1, n2, n3, n4; 
    n1 = rsqrtf(*p5 + *p6 + *p8 + *p9 + eps); 
    n2 = rsqrtf(*p2 + *p3 + *p5 + *p6 + eps); 
    n3 = rsqrtf(*p4 + *p5 + *p7 + *p8 + eps); 
    n4 = rsqrtf(*p1 + *p2 + *p4 + *p5 + eps);

    float t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0; 
    const float *src = histM.ptr(row - pad_y + 1) + (col - pad_x + 1) * numOrient;
    for (int o = 0; o < numOrient; o++)
    {
        float val = *src;
        float h1 = fminf(val*n1, (float)0.2);
        float h2 = fminf(val*n2, (float)0.2);
        float h3 = fminf(val*n3, (float)0.2);
        float h4 = fminf(val*n4, (float)0.2);
        *(dst++) = 0.5 * (h1 + h2 + h3 + h4);
        src++;
        t1 += h1;
        t2 += h2;
        t3 += h3;
        t4 += h4;
    }

    // contrast-insensitive features
    src = histM.ptr(row - pad_y + 1) + (col - pad_x + 1) * numOrient; 
    for (int o = 0; o < numOrient/2; o++)
    {
        float sum = *src + *(src + numOrient/2);
        float h1 = fminf(sum * n1, (float)0.2);
        float h2 = fminf(sum * n2, (float)0.2);
        float h3 = fminf(sum * n3, (float)0.2);
        float h4 = fminf(sum * n4, (float)0.2);
        *(dst++) = 0.5 * (h1 + h2 + h3 + h4);
        src++;
    }

    *(dst++) = 0.2357 * t1;
    *(dst++) = 0.2357 * t2;
    *(dst++) = 0.2357 * t3;
    *(dst++) = 0.2357 * t4;

    *(dst) = 0; 
}

FeatureGPU::FeatureGPU(const Size &size)
{
    origSize = size; 
}

void FeatureGPU::initialize(const FeatureGPUParams params) 
{
    setParams(params); 
    computeScale(); 
    initMats(); 
}

void FeatureGPU::loadImage(const Mat &image)
{
    if (image.size() != origSize)
        CV_Error(CV_StsBadSize, "Error loaded image size!");
    origImage.upload(image);
    // origSize = Size(origImage.cols, origImage.rows); 

    resetMats(); 
    createPyramid(); 
}

void FeatureGPU::loadImage(const gpu::GpuMat &image)
{
    if (image.size() != origSize)
        CV_Error(CV_StsBadSize, "Error loaded image size!");
    origImage = image;
    // origSize = Size(origImage.cols, origImage.rows); 

    resetMats(); 
    createPyramid(); 
}

void FeatureGPU::setParams(const FeatureGPUParams &p) 
{
    params = p;   
}

FeatureGPUParams FeatureGPU::getParams()
{
    return params;
}

void FeatureGPU::downloadFeature(std::vector<Mat> &feature)
{
    feature.resize(gpuFeat.size()); 
    for (int i = 0; i < gpuFeat.size(); i++)
        gpuFeat[i].download(feature[i]);
}

void FeatureGPU::downloadHist(std::vector<Mat> &hist)
{

    hist.resize(gpuHist.size()); 
    for (int i = 0; i < gpuHist.size(); i++)
        gpuHist[i].download(hist[i]);
}

void FeatureGPU::downloadNorm(std::vector<Mat> &norm)
{
    norm.resize(gpuNorm.size()); 
    for (int i = 0; i < gpuNorm.size(); i++)
        gpuNorm[i].download(norm[i]);
}

void FeatureGPU::computeScale() 
{
    CV_Assert(params.interval > 0);
    
    params.sfactor = pow(2.0, 1.0/params.interval); 
    params.maxScale = 1 + 
        (int)floor( 
                log(min(origSize.width, origSize.height) / (float)(params.sbin * 5.0 ))
                / log(params.sfactor)
            ); 

    if (params.maxScale < params.interval)
    {
        CV_Error(CV_StsBadArg, "The image is too small to create a pyramid");     
        return; 
    }

    int interval = params.interval; 
    int maxScale = params.maxScale; 

    scales.resize(maxScale + interval); 
    scaleSizes.resize(maxScale + interval); 
    
    for (int i = 0; i < interval; i++)
    {
        //doubled
        const double scale = (double)(1.0f/pow(params.sfactor, i));  // cannot modify!
        scales[i] = 2.0 * scale; 
        scaleSizes[i] = Size(origSize.width * scales[i], origSize.height * scales[i]); 

        //self
        scales[i + interval] = scale; 
        scaleSizes[i + interval] = Size(origSize.width * scale, origSize.height * scale); 

        //zoom out
        for (int j = i + interval; j < maxScale; j += interval) 
        {
            scales[j + interval] = scales[j] * 0.5; 
            scaleSizes[j + interval] = Size(origSize.width * (scale * 0.5), origSize.height * (scale * 0.5)); 
        }
    }
    
    // for (int i = 0; i < scaleSizes.size(); i++)
        // printf("%d %d\n", scaleSizes[i].width, scaleSizes[i].height); 
}

void FeatureGPU::initMats()
{
    int interval = params.interval;
    int maxScale = params.maxScale; 

    gpuImage.resize(interval + maxScale); 
    gpuHist.resize(interval + maxScale); 
    gpuNorm.resize(interval + maxScale); 
    gpuFeat.resize(interval + maxScale); 

    for (int i = 0; i < interval + maxScale; i++)
    {
        //larger than the original size
        const int w = scaleSizes[i].width;
        const int h = scaleSizes[i].height; 

        int sbin = params.sbin; 
        // sbin = (i < interval) ? params.sbin / 2 : params.sbin; 
        const int bW = cvRound((double)w / (double)(sbin));
        const int bH = cvRound((double)h / (double)(sbin));

        const int oW = max(bW - 2, 0) + 2 * (params.pad_x + 1); 
        const int oH = max(bH - 2, 0) + 2 * (params.pad_y + 1); 

        if (i >= interval)
            gpuImage[i] = gpu::GpuMat(Size(w, h), CV_32FC3, Scalar(0)); 
        else
            gpuImage[i] = gpu::GpuMat(); 
        gpuHist[i] = gpu::GpuMat(Size(bW * numOrient, bH), CV_32F, Scalar(0)); 
        gpuNorm[i] = gpu::GpuMat(Size(bW, bH), CV_32F, Scalar(0)); 
        gpuFeat[i] = gpu::GpuMat(Size(oW * dimHOG, oH), CV_32F, Scalar(0)); 
    }
}

void FeatureGPU::resetMats()
{
    int interval = params.interval;
    int maxScale = params.maxScale; 
    
    for (int i = 0; i < interval + maxScale; i++)
    {
        gpuHist[i].setTo(0);
        gpuNorm[i].setTo(0);
        gpuFeat[i].setTo(0);
    }
}

void FeatureGPU::createPyramid()
{
    int interval = params.interval;
    int maxScale = params.maxScale; 

    for (int i = interval; i < interval + maxScale; i++)
    {
        gpu::resize(origImage, gpuImage[i], scaleSizes[i]); 
    }
}

void FeatureGPU::computeHOG32D
(int i, const int sbin, const int pad_x, const int pad_y)
{
    int bW;
    int bH;
    if (i < params.interval)
    {
        bW = cvRound((float)gpuImage[i + params.interval].cols / sbin);
        bH = cvRound((float)gpuImage[i + params.interval].rows / sbin);
    }
    else 
    {
        bW = cvRound((float)gpuImage[i].cols / sbin); 
        bH = cvRound((float)gpuImage[i].rows / sbin); 
    }

    const int oW = fmaxf(bW-2, 0) + 2*pad_x;
    const int oH = fmaxf(bH-2, 0) + 2*pad_y;

    dim3 grid(bW, bH, 1);
    dim3 threads(sbin * sbin, 1, 1);

    if (i < params.interval)
        computeHOG32DHist<<< grid, threads >>> (gpuImage[i + params.interval], gpuHist[i], sbin, pad_x, pad_y);
    else
        computeHOG32DHist<<< grid, threads >>> (gpuImage[i], gpuHist[i], sbin, pad_x, pad_y);

    computeHOG32DNorm<<< bH, bW >>> (gpuHist[i], gpuNorm[i]);
    computeHOG32DFeat<<< oH, oW >>> (gpuHist[i], gpuNorm[i], gpuFeat[i], pad_x, pad_y);
}

void FeatureGPU::computeImage(const Mat &img, Mat &hist, 
int sbin, int pad_x, int pad_y)
{
    gpu::GpuMat gpuImg(img); 
    gpu::GpuMat gpuSingleHist(Size(img.cols / sbin * 18, img.rows / sbin), CV_32F, Scalar(0)); 

    int bW = img.cols / sbin; 
    int bH = img.rows / sbin; 

    dim3 grid(bW, bH, 1);
    dim3 threads(sbin * sbin, 1, 1);

    computeHOG32DHist<<< grid, threads >>> (gpuImg, gpuSingleHist, sbin, pad_x, pad_y);
    gpuSingleHist.download(hist); 
}

void FeatureGPU::computeHistPyramid()
{
    int interval = params.interval;
    int maxScale = params.maxScale; 
    for (int i = 0; i < interval + maxScale; i++)
        if (i < interval)
            computeHOG32D(i, params.sbin / 2, params.pad_x + 1, params.pad_y + 1);
        else
            computeHOG32D(i, params.sbin, params.pad_x + 1, params.pad_y + 1); 
}


}
 
}
