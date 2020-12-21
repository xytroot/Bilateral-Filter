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
//һά��˹kernel����
__constant__ float cGaussian[64];
//�����������ϵ����ȫ�ֱ�����ʽ����
texture<unsigned char, 2, cudaReadModeElementType> inTexture;


//����һά��˹����Ȩ�أ���ά��˹Ȩ�ؿ���һά��˹Ȩ�������õ�
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

// һά��˹�������������ز���Ȩ��
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
		//����ʰȡ���õ�Ҫ������������ص�
		unsigned char centrePx = tex2D(inTexture, txIndex, tyIndex);
		//���о������
		for (int dy = -r; dy <= r; dy++) {
			for (int dx = -r; dx <= r; dx++) {
				//�õ�kernel��������һ���ص�
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
	//GPU��ʱ�¼�
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//����ͼƬ��С
	int gray_size = input.step * input.rows;

	//��device�Ͽ���2ά���ݿռ䱣�������������
	unsigned char* d_input = NULL;
	unsigned char* d_output;

	updateGaussian(r, sigmaSpace);

	//����device�ڴ�
	cudaMalloc<unsigned char>(&d_output, gray_size);

	//�����
	size_t pitch;
	cudaMallocPitch(&d_input, &pitch, sizeof(unsigned char) * input.step, input.rows);
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
	cudaMemcpy2D(d_input, pitch, input.ptr(), sizeof(unsigned char) * input.step, sizeof(unsigned char) * input.step, input.rows, cudaMemcpyHostToDevice);
	//���������ϵ�󶨵�һ��CUDA����
	cudaBindTexture2D(0, inTexture, d_input, desc, input.step, input.rows, pitch);
	
	dim3 block(16, 16);
	dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	cudaEventRecord(start, 0);
	gpuCalculation <<< grid, block >>> (d_input, d_output, input.cols, input.rows, r, sigmaColor);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//��device�ϵ�������������host��
	cudaMemcpy(output.ptr(), d_output, gray_size, cudaMemcpyDeviceToHost);

	//�ͷ�device��host�Ϸ�����ڴ�
	cudaFree(d_input);
	cudaFree(d_output);

	// Calculate and print kernel run time
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the GPU: %f ms\n", time);
}