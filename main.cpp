#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void MyBilateralFilter(const Mat& input, Mat& output, int r, double sI, double sS);

int main() {

	//高斯核直径
	int d = 8;
	double sigmaColor = 15.0, sigmaSpace = 15.0;
	//将原始图像转化为灰度图像再打开
	Mat srcImg = imread("1.jpg", IMREAD_GRAYSCALE);
	//分配host内存
	Mat dstImg(srcImg.rows, srcImg.cols, CV_8UC1);
	Mat dstImgCV;

	//在GPU上运行测速
	MyBilateralFilter(srcImg, dstImg, d/2, sigmaColor, sigmaSpace);

	//使用OpenCV bilateral filter在cpu上测速
	clock_t start_s = clock();
	bilateralFilter(srcImg, dstImgCV, d, sigmaColor, sigmaSpace);
	clock_t stop_s = clock();
	cout << "Time for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl;
	//展示图片
	imshow("原图", srcImg);
	imwrite("space2.jpg", srcImg);
	imshow("GPU加速双边滤波", dstImg);
	imwrite("space2.jpg", dstImg);
	imshow("CPU双边滤波", dstImgCV);
	imwrite("space2.jpg", dstImgCV);
	cv::waitKey();
}