#include <iostream>
#include <algorithm>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#define M_PI           3.14159265358979323846


using namespace std;
using namespace cv;
//一维高斯kernel数组
__constant__ float cGaussian[64];
//声明纹理参照系，以全局变量形式出现
texture<unsigned char, 2, cudaReadModeElementType> inTexture;


//计算一维高斯距离权重，二维高斯权重可由一维高斯权重做积得到
void updateGaussian(int r, double sd)
{
	float fGaussian[64];
	for (int i = 0; i < 2 * r + 1; i++)
	{
		float x = i - r;
		fGaussian[i] = 1 / (sqrt(2 * M_PI) * sd) * expf(-(x * x) / (2 * sd * sd));
	}
	cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float) * (2 * r + 1));
}

// 一维高斯函数，计算像素差异权重
__device__ inline double gaussian(float x, double sigma)
{
	return 1 / (sqrt(2 * M_PI) * sigma) * __expf(-(powf(x, 2)) / (2 * powf(sigma, 2)));
}

__global__ void gpuCalculation(unsigned char* input, unsigned char* output, int width ,int height, int r,double sigmaColor)
{
	int txIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int tyIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if ((txIndex < width) && (tyIndex < height))
	{
		double iFiltered = 0;
		double k = 0;
		//纹理拾取，得到要计算的中心像素点
		unsigned char centrePx = tex2D(inTexture, txIndex, tyIndex);
		//进行卷积运算
		for (int dy = -r; dy <= r; dy++) {
			for (int dx = -r; dx <= r; dx++) {
				//得到kernel区域内另一像素点
				unsigned char currPx = tex2D(inTexture, txIndex + dx, tyIndex + dy);
				// Weight = 1D Gaussian(x_axis) * 1D Gaussian(y_axis) * Gaussian(Color difference)
				double w = (cGaussian[dy + r] * cGaussian[dx + r]) * gaussian(centrePx - currPx, sigmaColor);
				iFiltered += w * currPx;
				k += w;
			}
		}
		output[tyIndex * width + txIndex] = iFiltered / k;
	}
}

void MyBilateralFilter(const Mat& input, Mat& output, int r, double sigmaColor, double sigmaSpace)
{
	//GPU计时事件
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//计算图片大小
	int gray_size = input.step * input.rows;

	//在device上开辟2维数据空间保存输入输出数据
	unsigned char* d_input = NULL;
	unsigned char* d_output;

	updateGaussian(r, sigmaSpace);

	//分配device内存
	cudaMalloc<unsigned char>(&d_output, gray_size);

	//纹理绑定
	size_t pitch;
	cudaMallocPitch(&d_input, &pitch, sizeof(unsigned char) * input.step, input.rows);
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
	cudaMemcpy2D(d_input, pitch, input.ptr(), sizeof(unsigned char) * input.step, sizeof(unsigned char) * input.step, input.rows, cudaMemcpyHostToDevice);
	//将纹理参照系绑定到一个CUDA数组
	cudaBindTexture2D(0, inTexture, d_input, desc, input.step, input.rows, pitch);
	
	dim3 block(16, 16);
	dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	cudaEventRecord(start, 0);
	gpuCalculation <<< grid, block >>> (d_input, d_output, input.cols, input.rows, r, sigmaColor);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//将device上的运算结果拷贝到host上
	cudaMemcpy(output.ptr(), d_output, gray_size, cudaMemcpyDeviceToHost);

	//释放device和host上分配的内存
	cudaFree(d_input);
	cudaFree(d_output);

	// Calculate and print kernel run time
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the GPU: %f ms\n", time);
}