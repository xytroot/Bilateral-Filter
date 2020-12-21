#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void MyBilateralFilter(const Mat& input, Mat& output, int r, double sI, double sS);

int main() {

	//��˹��ֱ��
	int d = 8;
	double sigmaColor = 15.0, sigmaSpace = 15.0;
	//��ԭʼͼ��ת��Ϊ�Ҷ�ͼ���ٴ�
	Mat srcImg = imread("1.jpg", IMREAD_GRAYSCALE);
	//����host�ڴ�
	Mat dstImg(srcImg.rows, srcImg.cols, CV_8UC1);
	Mat dstImgCV;

	//��GPU�����в���
	MyBilateralFilter(srcImg, dstImg, d/2, sigmaColor, sigmaSpace);

	//ʹ��OpenCV bilateral filter��cpu�ϲ���
	clock_t start_s = clock();
	bilateralFilter(srcImg, dstImgCV, d, sigmaColor, sigmaSpace);
	clock_t stop_s = clock();
	cout << "Time for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl;
	//չʾͼƬ
	imshow("ԭͼ", srcImg);
	imwrite("space2.jpg", srcImg);
	imshow("GPU����˫���˲�", dstImg);
	imwrite("space2.jpg", dstImg);
	imshow("CPU˫���˲�", dstImgCV);
	imwrite("space2.jpg", dstImgCV);
	cv::waitKey();
}