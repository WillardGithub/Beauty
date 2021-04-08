// BeautyAlgorithm.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
// MathFuncDll.cpp: 定义 DLL 应用程序的导出函数。
//
#include "BeautyAlgorithm.h" 
#include "facedetect-dll.h"
using namespace cv;
#define PI 3.1415926
#define DETECT_BUFFER_SIZE 0x20000

#define max_uchar(a, b)  (((a) > (b)) ? (a) : (b))
#define min_uchar(a, b)  (((a) < (b)) ? (a) : (b))

#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define CLIP3(x, a, b) MIN2(MAX2(a,x), b)


void CompMeanAndVariance(Mat& img, Vec3f& mean3f, Vec3f& variance3f)
{
	int row = img.rows;
	int col = img.cols;
	int total = row * col;
	float sum[3] = { 0.0f };
	// 均值
	uchar* pImg = img.data;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			sum[0] += pImg[3 * j + 0];
			sum[1] += pImg[3 * j + 1];
			sum[2] += pImg[3 * j + 2];
		}
		pImg += img.step;
	}

	mean3f[0] = sum[0] / total;
	mean3f[1] = sum[1] / total;
	mean3f[2] = sum[2] / total;

	memset(sum, 0, sizeof(sum));
	// 标准差
	pImg = img.data;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			sum[0] += (pImg[3 * j + 0] - mean3f[0]) * (pImg[3 * j + 0] - mean3f[0]);
			sum[1] += (pImg[3 * j + 1] - mean3f[1]) * (pImg[3 * j + 1] - mean3f[1]);
			sum[2] += (pImg[3 * j + 2] - mean3f[2]) * (pImg[3 * j + 2] - mean3f[2]);
		}
		pImg += img.step;
	}

	variance3f[0] = sqrt(sum[0] / total);
	variance3f[1] = sqrt(sum[1] / total);
	variance3f[2] = sqrt(sum[2] / total);
}


Mat BilinearInterpolation(Mat& img, int startX, int startY, int endX, int endY, float radius) //平移及拉伸（插值算法）
{
	if (radius> startX)
	{
		radius = startX - 1;
	}

	if (radius > img.cols - startX)
	{
		radius = img.cols - startX - 1;
	}

	if (radius > startY)
	{
		radius = startY - 1;
	}

	if (radius > img.rows - startY)
	{
		radius = img.rows - startY -1;
	}


	double ddradius = float(radius * radius);

	//double ddradius = radius;
	Mat copyImg;
	copyMakeBorder(img, copyImg, 0, 0, 0, 0, BORDER_REPLICATE);
	double ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY);
	//ddmc = sqrt(ddmc);
	int H = img.rows;
	int W = img.cols;
	int C = 3;
	for (int i = 0; i < W; i++)
	{
		for (int j = 0; j < H; j++)
		{
			if (fabs(i - startX) > radius || fabs(j - startY) > radius)
			{
				continue;
			}
			double distance = (i - startX) * (i - startX) + (j - startY) * (j - startY);
			//distance = sqrt(distance);
			if (distance < ddradius)
			{
				double ratio = (ddradius - distance) / (ddradius - distance + ddmc);
				ratio = ratio * ratio;
				double UX = i - ratio * (endX - startX);
				double UY = j - ratio * (endY - startY);
				int x1 = UX;
				int x2 = x1 + 1;
				int y1 = UY;
				int y2 = y1 + 1;
				float a0 = float(img.ptr<Vec3b>(y1)[x1][0]) * (float(x2) - UX) * (float(y2) - UY);
				float a1 = float(img.ptr<Vec3b>(y1)[x1][1]) * (float(x2) - UX) * (float(y2) - UY);
				float a2 = float(img.ptr<Vec3b>(y1)[x1][2]) * (float(x2) - UX) * (float(y2) - UY);

				float b0 = float(img.ptr<Vec3b>(y1)[x2][0]) * (UX - float(x1)) * (float(y2) - UY);
				float b1 = float(img.ptr<Vec3b>(y1)[x2][1]) * (UX - float(x1)) * (float(y2) - UY);
				float b2 = float(img.ptr<Vec3b>(y1)[x2][2]) * (UX - float(x1)) * (float(y2) - UY);

				float c0 = float(img.ptr<Vec3b>(y2)[x1][0]) * (float(x2) - UX) * (UY - float(y1));
				float c1 = float(img.ptr<Vec3b>(y2)[x1][1]) * (float(x2) - UX) * (UY - float(y1));
				float c2 = float(img.ptr<Vec3b>(y2)[x1][2]) * (float(x2) - UX) * (UY - float(y1));

				float d0 = float(img.ptr<Vec3b>(y2)[x2][0]) * (UX - float(x1)) * (UY - float(y1));
				float d1 = float(img.ptr<Vec3b>(y2)[x2][1]) * (UX - float(x1)) * (UY - float(y1));
				float d2 = float(img.ptr<Vec3b>(y2)[x2][2]) * (UX - float(x1)) * (UY - float(y1));
				copyImg.ptr<Vec3b>(j)[i][0] = int(a0 + b0 + c0 + d0);
				copyImg.ptr<Vec3b>(j)[i][1] = int(a1 + b1 + c1 + d1);
				copyImg.ptr<Vec3b>(j)[i][2] = int(a2 + b2 + c2 + d2);
			}
		}
	}

	return copyImg;
}

Mat BilinearInterpolation1(Mat& img, int startX, int startY, float moudxnorm, float radius) //大眼拉伸用
{
	if (radius > startX)
	{
		radius = startX - 1;
	}

	if (radius > img.cols - startX)
	{
		radius = img.cols - startX - 1;
	}

	if (radius > startY)
	{
		radius = startY - 1;
	}

	if (radius > img.rows - startY)
	{
		radius = img.rows - startY - 1;
	}

	double ddradius = float(radius * radius);
	Mat copyImg;
	img.copyTo(copyImg);
	//copyMakeBorder(img, copyImg, 0, 1, 0, 1, BORDER_REPLICATE);
	int H = img.rows;
	int W = img.cols;
	int C = 3;
	for (int i = 0; i < W; i++)
	{
		for (int j = 0; j < H; j++)
		{
			if (fabs(i - startX) > radius || fabs(j - startY) > radius)
			{
				continue;
			}
			double distance = (i - startX) * (i - startX) + (j - startY) * (j - startY);
			if (distance < ddradius)
			{
				double rnorm = sqrt(distance) / radius;
				double a = 1 - (rnorm - 1) * (rnorm - 1) * moudxnorm;
				double UX = startX + a * (i - startX);
				double UY = startY + a * (j - startY);
				//双线性插值法（或者内点法）//
				int x1 = UX;
				int x2 = x1 + 1;
				int y1 = UY;
				int y2 = y1 + 1;
				float a0 = float(img.ptr<Vec3b>(y1)[x1][0]) * (float(x2) - UX) * (float(y2) - UY);
				float a1 = float(img.ptr<Vec3b>(y1)[x1][1]) * (float(x2) - UX) * (float(y2) - UY);
				float a2 = float(img.ptr<Vec3b>(y1)[x1][2]) * (float(x2) - UX) * (float(y2) - UY);

				float b0 = float(img.ptr<Vec3b>(y1)[x2][0]) * (UX - float(x1)) * (float(y2) - UY);
				float b1 = float(img.ptr<Vec3b>(y1)[x2][1]) * (UX - float(x1)) * (float(y2) - UY);
				float b2 = float(img.ptr<Vec3b>(y1)[x2][2]) * (UX - float(x1)) * (float(y2) - UY);

				float c0 = float(img.ptr<Vec3b>(y2)[x1][0]) * (float(x2) - UX) * (UY - float(y1));
				float c1 = float(img.ptr<Vec3b>(y2)[x1][1]) * (float(x2) - UX) * (UY - float(y1));
				float c2 = float(img.ptr<Vec3b>(y2)[x1][2]) * (float(x2) - UX) * (UY - float(y1));

				float d0 = float(img.ptr<Vec3b>(y2)[x2][0]) * (UX - float(x1)) * (UY - float(y1));
				float d1 = float(img.ptr<Vec3b>(y2)[x2][1]) * (UX - float(x1)) * (UY - float(y1));
				float d2 = float(img.ptr<Vec3b>(y2)[x2][2]) * (UX - float(x1)) * (UY - float(y1));
				copyImg.ptr<Vec3b>(j)[i][0] = int(a0 + b0 + c0 + d0);
				copyImg.ptr<Vec3b>(j)[i][1] = int(a1 + b1 + c1 + d1);
				copyImg.ptr<Vec3b>(j)[i][2] = int(a2 + b2 + c2 + d2);
			}
		}
	}
	return copyImg;
}

Mat BilinearInterpolation2(Mat& img, int startX, int startY, float strength, float radius) //局部打光用
{
	if (radius > startX)
	{
		radius = startX - 1;
	}

	if (radius > img.cols - startX)
	{
		radius = img.cols - startX - 1;
	}

	if (radius > startY)
	{
		radius = startY - 1;
	}

	if (radius > img.rows - startY)
	{
		radius = img.rows - startY - 1;
	}

	double ddradius = float(radius * radius);
	Mat copyImg;
	copyMakeBorder(img, copyImg, 0, 0, 0,0, BORDER_REPLICATE);

	int H = img.rows;
	int W = img.cols;
	int C = 3;
	for (int i = 0; i < W; i++)
	{
		for (int j = 0; j < H; j++)
		{
			if (fabs(i - startX) > radius && fabs(j - startY) > radius)
			{
				continue;
			}
			double distance = (i - startX) * (i - startX) + (j - startY) * (j - startY);
			if (distance < ddradius)
			{
				double rnorm = sqrt(distance) / radius;
				int a = (1.0 - rnorm) * strength;
				int B = float(img.ptr<Vec3b>(j)[i][0]) + a;
				int G = float(img.ptr<Vec3b>(j)[i][1]) + a;
				int R = float(img.ptr<Vec3b>(j)[i][2]) + a;
				//防止越界
				copyImg.ptr<Vec3b>(j)[i][0] = min(255, max(0, B));
				copyImg.ptr<Vec3b>(j)[i][1] = min(255, max(0, G));
				copyImg.ptr<Vec3b>(j)[i][2] = min(255, max(0, R));
			}
		}
	}
	return copyImg;
}

Mat ForeHeadHandle(Mat& img, int upRow, int downRow, int leftCol, int rightCol,float radius)
{
	Mat copyImg;
	copyMakeBorder(img, copyImg, 0, 0, 0, 0, BORDER_REPLICATE);

	int moveline = (int)radius;

	//int rectRows = downRow - upRow;

	int H = img.rows;
	int W = img.cols;
	int C = 3;


	for (int j = 0; j < H; j++)
	{
		if (j <= upRow || j >= downRow)
		{
			continue;
		}
		for (int i = 0; i < W; i++)
		{
			if (i <= leftCol || i >= rightCol)
			{
				continue;
			}
			for (int k = moveline - 1; k > 0; k--)
			{
				if (j + k < downRow)
				{
					copyImg.ptr<Vec3b>(j + k)[i][0] = img.ptr<Vec3b>(j + k - 1)[i][0];
					copyImg.ptr<Vec3b>(j + k)[i][1] = img.ptr<Vec3b>(j + k - 1)[i][1];
					copyImg.ptr<Vec3b>(j + k)[i][2] = img.ptr<Vec3b>(j + k - 1)[i][2];
				}
			}

			/*double distance = (i - (leftCol+rightCol)/2) * (i - (leftCol + rightCol) / 2) + (j - (upRow+downRow)/2) * (j - (upRow + downRow) / 2);
			double ddradius = float(radius * radius);
			double ddmc = (leftCol - rightCol) * (leftCol - rightCol) + (upRow - downRow) * (upRow - downRow);
			double ratio = (ddradius - distance) / (ddradius - distance + ddmc);
			ratio = ratio * ratio;
			double UX = i - ratio * (leftCol + rightCol) / 2;
			double UY = j - ratio * (leftCol + rightCol) / 2;
			int x1 = UX;
			int x2 = x1 + 1;
			int y1 = UY;
			int y2 = y1 + 1;
			float a0 = float(img.ptr<Vec3b>(y1)[x1][0]) * (float(x2) - UX) * (float(y2) - UY);
			float a1 = float(img.ptr<Vec3b>(y1)[x1][1]) * (float(x2) - UX) * (float(y2) - UY);
			float a2 = float(img.ptr<Vec3b>(y1)[x1][2]) * (float(x2) - UX) * (float(y2) - UY);

			float b0 = float(img.ptr<Vec3b>(y1)[x2][0]) * (UX - float(x1)) * (float(y2) - UY);
			float b1 = float(img.ptr<Vec3b>(y1)[x2][1]) * (UX - float(x1)) * (float(y2) - UY);
			float b2 = float(img.ptr<Vec3b>(y1)[x2][2]) * (UX - float(x1)) * (float(y2) - UY);

			float c0 = float(img.ptr<Vec3b>(y2)[x1][0]) * (float(x2) - UX) * (UY - float(y1));
			float c1 = float(img.ptr<Vec3b>(y2)[x1][1]) * (float(x2) - UX) * (UY - float(y1));
			float c2 = float(img.ptr<Vec3b>(y2)[x1][2]) * (float(x2) - UX) * (UY - float(y1));

			float d0 = float(img.ptr<Vec3b>(y2)[x2][0]) * (UX - float(x1)) * (UY - float(y1));
			float d1 = float(img.ptr<Vec3b>(y2)[x2][1]) * (UX - float(x1)) * (UY - float(y1));
			float d2 = float(img.ptr<Vec3b>(y2)[x2][2]) * (UX - float(x1)) * (UY - float(y1));
			copyImg.ptr<Vec3b>(j)[i][0] = int(a0 + b0 + c0 + d0);
			copyImg.ptr<Vec3b>(j)[i][1] = int(a1 + b1 + c1 + d1);
			copyImg.ptr<Vec3b>(j)[i][2] = int(a2 + b2 + c2 + d2);*/

		}
	}
	return copyImg;

}

Mat guidedfilter(Mat& srcImage, Mat& srcClone, int r, double eps)  //导向滤波
{
	// 转换源图像信息
	srcImage.convertTo(srcImage, CV_64FC1);
	srcClone.convertTo(srcClone, CV_64FC1);
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;

	r = 2 * r + 1;
	Mat boxResult;
	// 步骤一: 计算均值
	//boxFilter(Mat::ones(nRows, nCols, srcImage.type()),boxResult, CV_64FC1, Size(r, r));
	// 生成导向均值mean_I    
	Mat mean_I;
	boxFilter(srcImage, mean_I, CV_64FC1, Size(r, r));
	// 生成原始均值mean_p   
	Mat mean_p;
	boxFilter(srcClone, mean_p, CV_64FC1, Size(r, r));
	// 生成互相关均值mean_Ip 
	Mat mean_Ip;
	boxFilter(srcImage.mul(srcClone), mean_Ip,CV_64FC1, Size(r, r));
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
	// 生成自相关均值mean_II 
	Mat mean_II;
	boxFilter(srcImage.mul(srcImage), mean_II, CV_64FC1, Size(r, r));
	// 步骤二：计算相关系数   
	Mat var_I = mean_II - mean_I.mul(mean_I);

	// 步骤三：计算参数系数a，b 
	Mat a = cov_Ip / (var_I + eps);
	Mat b = mean_p - a.mul(mean_I);
	// 步骤四：计算系数a，b均值    
	Mat mean_a;
	boxFilter(a, mean_a, CV_64FC1, Size(r, r));
	//mean_a = mean_a / boxResult;
	Mat mean_b;
	boxFilter(b, mean_b, CV_64FC1, Size(r, r));
	//mean_b = mean_b / boxResult;
	//步骤五：生成输出矩阵 
	Mat resultMat = mean_a.mul(srcImage) + mean_b;
	return resultMat;
}

//瘦脸
double  __stdcall SlimFace(Mat src, double ratio, Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}

	double rate = ratio * 0.01;
	float r_left1 = sqrt((p[6 + 2 * 1] - p[6 + 2 * 30]) * (p[6 + 2 * 1] - p[6 + 2 * 30]) + (p[6 + 2 * 1 + 1] - p[6 + 2 * 30 + 1]) * (p[6 + 2 * 1 + 1] - p[6 + 2 * 30 + 1]));
	float r_right1 = sqrt((p[6 + 2 * 15] - p[6 + 2 * 30]) * (p[6 + 2 * 15] - p[6 + 2 * 30]) + (p[6 + 2 * 15 + 1] - p[6 + 2 * 30 + 1]) * (p[6 + 2 * 15 + 1] - p[6 + 2 * 30 + 1]));
	Mat data = BilinearInterpolation(src, p[6 + 2 * 1], p[6 + 2 *1 + 1], p[6 + 2 * 30], p[6 + 2 * 30 + 1], rate * r_left1 * 0.35);
	Mat data1 = BilinearInterpolation(data, p[6 + 2 * 15], p[6 + 2 * 15 + 1], p[6 + 2 * 30], p[6 + 2 * 30 + 1], rate * r_right1 * 0.35);

	float r_left2 = sqrt((p[6 + 2 * 4] - p[6 + 2 * 51]) * (p[6 + 2 * 4] - p[6 + 2 * 51]) + (p[6 + 2 * 4 + 1] - p[6 + 2 * 51 + 1]) * (p[6 + 2 * 4 + 1] - p[6 + 2 * 51 + 1]));
	float r_right2 = sqrt((p[6 + 2 * 12] - p[6 + 2 * 51]) * (p[6 + 2 * 12] - p[6 + 2 * 51]) + (p[6 + 2 * 12 + 1] - p[6 + 2 * 51 + 1]) * (p[6 + 2 * 12 + 1] - p[6 + 2 * 51+ 1]));
	Mat data2 = BilinearInterpolation(data1, p[6 + 2 * 4], p[6 + 2 * 4 + 1], p[6 + 2 * 51], p[6 + 2 * 51 + 1], rate * r_left1 * 0.35);
	Mat data3 = BilinearInterpolation(data2, p[6 + 2 * 12], p[6 + 2 * 12 + 1], p[6 + 2 * 51], p[6 + 2 * 51 + 1], rate * r_right1 * 0.35);

	//额头缩放
	//Mat data4 = BilinearInterpolation(data3, p[6 + 2 * 20], 2 * p[6 + 2 * 20 + 1] - p[6 + 2 * 29 + 1], p[6 + 2 * 20], p[6 + 2 * 20 + 1], p[6 + 2 * 29 + 1] - p[6 + 2 * 20 + 1]);
	//Mat data5 = BilinearInterpolation(data4, p[6 + 2 * 25], 2 * p[6 + 2 * 20 + 1] - p[6 + 2 * 29 + 1], p[6 + 2 * 20], p[6 + 2 * 20 + 1], p[6 + 2 * 29 + 1] - p[6 + 2 * 20 + 1]);
	//Mat data6 = BilinearInterpolation(data5, p[6 + 2 * 29], 2 * p[6 + 2 * 20 + 1] - p[6 + 2 * 29 + 1], p[6 + 2 * 20], p[6 + 2 * 20 + 1], p[6 + 2 * 29 + 1] - p[6 + 2 * 20 + 1]);

	//float radius_left_eye = sqrt((p[6 + 2 * 19] - p[6 + 2 * 37]) * (p[6 + 2 * 19] - p[6 + 2 * 37]) + (p[6 + 2 * 19 + 1] - p[6 + 2 * 37 + 1]) * (p[6 + 2 *19 + 1] - p[6 + 2 * 37 + 1]));
	//float radius_right_eye = sqrt((p[6 + 2 * 24] - p[6 + 2 * 44]) * (p[6 + 2 * 24] - p[6 + 2 * 44]) + (p[6 + 2 * 24 + 1] - p[6 + 2 * 44 + 1]) * (p[6 + 2 * 24 + 1] - p[6 + 2 * 44 + 1]));

	//Mat data4 = BilinearInterpolation(data3, p[6 + 2 * 19], (int)(p[6 + 2 * 19 + 1]*0.75), p[6 + 2 * 19], (int)(p[6 + 2 * 19 + 1] + radius_left_eye*1.1), rate * radius_left_eye*1.25);
	//Mat data5 = BilinearInterpolation(data4, p[6 + 2 * 24], (int)(p[6 + 2 * 24 + 1]*0.75), p[6 + 2 * 24], (int)(p[6 + 2 * 24 + 1]+  radius_right_eye*1.1), rate * radius_right_eye*1.25);

	//Mat data6 = BilinearInterpolation(data5, p[6 + 2 * 27], (int)((p[6 + 2 * 21 + 1] + p[6 + 2 * 22 + 1])*0.5*0.8), p[6 + 2 * 27], p[6 + 2 * 27 + 1] * 0.75, rate * (r_right1+r_right2) *0.5* 0.8);
	//Mat data6 = BilinearInterpolation(data5, p[6 + 2 * 29], 2 * p[6 + 2 * 20 + 1] - p[6 + 2 * 29 + 1], p[6 + 2 * 20], p[6 + 2 * 20 + 1], rate * (r_right1+ r_left1) * 0.25);

	data3.copyTo(dst);


	return 0;
}

//大眼
double  __stdcall ZoomEyes(Mat src, double ratio, Mat& dst,short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	double rate = ratio * 0.01;

	int left_landmark = 37;  //左眼左边界点
	int left_landmark_down = 40;  //左眼右边界点
	int right_landmark = 43;  //右眼左边界点
	int right_landmark_down = 46; //右眼右边界点

	float r_left = sqrt((p[6 + 2 * left_landmark] - p[6 + 2 * left_landmark_down]) * (p[6 + 2 * left_landmark] - p[6 + 2 * left_landmark_down]) + (p[6 + 2 * left_landmark + 1] - p[6 + 2 * left_landmark_down + 1]) * (p[6 + 2 * left_landmark + 1] - p[6 + 2 * left_landmark_down + 1]));
	float r_right = sqrt((p[6 + 2 * right_landmark] - p[6 + 2 * right_landmark_down]) * (p[6 + 2 * right_landmark] - p[6 + 2 * right_landmark_down]) + (p[6 + 2 * right_landmark + 1] - p[6 + 2 * right_landmark_down + 1]) * (p[6 + 2 * right_landmark + 1] - p[6 + 2 * right_landmark_down + 1]));
	
	Mat data = BilinearInterpolation1(src, (p[6 + 2 * left_landmark] + p[6 + 2 * left_landmark_down])/2, (p[6 + 2 * left_landmark+1] + p[6 + 2 * left_landmark_down+1]) / 2, 0.8*rate,  r_left );
	Mat data1 = BilinearInterpolation1(data, (p[6 + 2 * right_landmark] + p[6 + 2 * right_landmark_down]) / 2, (p[6 + 2 * right_landmark + 1] + p[6 + 2 * right_landmark_down + 1]) / 2, 0.8*rate, r_right);
	data1.copyTo(dst);

	return 0;
}

//瘦鼻
double  __stdcall SlimNose(Mat src, double ratio, Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	double rate = ratio * 0.01;

	int left_landmark = 31;
	int left_landmark_down = 30;
	int right_landmark = 35;
	int right_landmark_down = 30;
	int endPt = 30;

	float r_left = sqrt((p[6 + 2 * left_landmark] - p[6 + 2 * left_landmark_down]) * (p[6 + 2 * left_landmark] - p[6 + 2 * left_landmark_down]) + (p[6 + 2 * left_landmark + 1] - p[6 + 2 * left_landmark_down + 1]) * (p[6 + 2 * left_landmark + 1] - p[6 + 2 * left_landmark_down + 1]));
	float r_right = sqrt((p[6 + 2 * right_landmark] - p[6 + 2 * right_landmark_down]) * (p[6 + 2 * right_landmark] - p[6 + 2 * right_landmark_down]) + (p[6 + 2 * right_landmark + 1] - p[6 + 2 * right_landmark_down + 1]) * (p[6 + 2 * right_landmark + 1] - p[6 + 2 * right_landmark_down + 1]));
	Mat data = BilinearInterpolation(src, p[6 + 2 * left_landmark], p[6 + 2 * left_landmark + 1], p[6 + 2 * endPt], p[6 + 2 * endPt + 1], rate * r_left*0.75);
	Mat data1 = BilinearInterpolation(data, p[6 + 2 * right_landmark], p[6 + 2 * right_landmark + 1], p[6 + 2 * endPt], p[6 + 2 * endPt + 1], rate * r_right*0.75);
	data1.copyTo(dst);

	return 0;
}

//锥子脸效果
//double  __stdcall AwlFace(Mat src, double ratio, Mat& dst, short* p)
//{
//	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
//	{
//		return -1;
//	}
//	double rate = ratio * 0.01;
//	/*unsigned char* pBuffer = (unsigned char*)malloc(DETECT_BUFFER_SIZE);
//	Mat gray;
//	cvtColor(src, gray, COLOR_BGR2GRAY);
//	auto pResults = facedetect_multiview_reinforce(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step, 1.2f, 2, 48, 0, 1);
//	short* p = (short*)(pResults + 1);*/
//
//	//SlimFace(src, ratio, dst);
//
//	//float r_left1 = sqrt((p[6 + 2 * 3] - p[6 + 2 * 5]) * (p[6 + 2 * 3] - p[6 + 2 * 5]) + (p[6 + 2 * 3 + 1] - p[6 + 2 * 5 + 1]) * (p[6 + 2 * 3 + 1] - p[6 + 2 * 5 + 1]));
//	//float r_right1 = sqrt((p[6 + 2 * 13] - p[6 + 2 * 15]) * (p[6 + 2 * 13] - p[6 + 2 * 15]) + (p[6 + 2 * 13 + 1] - p[6 + 2 * 15 + 1]) * (p[6 + 2 * 13 + 1] - p[6 + 2 * 15 + 1]));
//	//Mat data_slim1 = BilinearInterpolation(src, p[6 + 2 * 3], p[6 + 2 * 3 + 1], p[6 + 2 * 30], p[6 + 2 * 30 + 1],  rate*r_left1*0.9);
//	//Mat data1_slim2 = BilinearInterpolation(data_slim1, p[6 + 2 * 13], p[6 + 2 * 13 + 1], p[6 + 2 * 30], p[6 + 2 * 30 + 1],  rate*r_right1*0.9);
//
//	int left_landmark = 6;
//	int left_landmark_down = 8;
//	int right_landmark = 10;
//	int right_landmark_down = 8;
//
//	float r_left = sqrt((p[6 + 2 * left_landmark] - p[6 + 2 * left_landmark_down]) * (p[6 + 2 * left_landmark] - p[6 + 2 * left_landmark_down]) + (p[6 + 2 * left_landmark + 1] - p[6 + 2 * left_landmark_down + 1]) * (p[6 + 2 * left_landmark + 1] - p[6 + 2 * left_landmark_down + 1]));
//	float r_right = sqrt((p[6 + 2 * right_landmark] - p[6 + 2 * right_landmark_down]) * (p[6 + 2 * right_landmark] - p[6 + 2 * right_landmark_down]) + (p[6 + 2 * right_landmark + 1] - p[6 + 2 * right_landmark_down + 1]) * (p[6 + 2 * right_landmark + 1] - p[6 + 2 * right_landmark_down + 1]));
//
//	Mat data1 = BilinearInterpolation(dst, p[6 + 2 * right_landmark_down], p[6 + 2 * right_landmark_down + 1], p[6 + 2 * right_landmark_down], p[6 + 2 * right_landmark_down + 1] + 10, rate * (r_right+ r_left)*0.35);
//	Mat data2 = BilinearInterpolation(data1, p[6 + 2 * left_landmark], p[6 + 2 * left_landmark + 1], p[6 + 2 * right_landmark_down], p[6 + 2 * right_landmark_down + 1] + 10, rate * (r_right + r_left) * 0.35);
//	Mat data3 = BilinearInterpolation(data2, p[6 + 2 * right_landmark], p[6 + 2 * right_landmark + 1], p[6 + 2 * right_landmark_down], p[6 + 2 * right_landmark_down + 1] + 10, rate * (r_right + r_left)* 0.35);
//
//	Mat data4 = BilinearInterpolation(data3, p[6 + 2 * (left_landmark-1)], p[6 + 2 * (left_landmark-1) + 1], p[6 + 2 * right_landmark_down], p[6 + 2 * right_landmark_down + 1] + 10, rate * (r_right + r_left) * 0.35);
//	Mat data5 = BilinearInterpolation(data4, p[6 + 2 * (right_landmark+1)], p[6 + 2 * (right_landmark+1) + 1], p[6 + 2 * right_landmark_down], p[6 + 2 * right_landmark_down + 1] + 10, rate * (r_right + r_left) * 0.35);
//
//	Mat data6 = BilinearInterpolation(data5, p[6 + 2 * (left_landmark - 2)], p[6 + 2 * (left_landmark - 2) + 1], p[6 + 2 * right_landmark_down], p[6 + 2 * right_landmark_down + 1] + 10, rate * (r_right + r_left) * 0.35);
//	Mat data7 = BilinearInterpolation(data6, p[6 + 2 * (right_landmark + 2)], p[6 + 2 * (right_landmark + 2) + 1], p[6 + 2 * right_landmark_down], p[6 + 2 * right_landmark_down + 1] + 10, rate * (r_right + r_left) * 0.35);
//
//	data7.copyTo(dst);
//
//	free(pBuffer);
//	pBuffer = NULL;
//
//	return 0;
//}

//added at 2020-5-10
double  __stdcall GlobalBuffing(Mat src, double ratio, Mat& dst)   //整体磨皮
{
	if (src.empty())
	{
		return -1;
	}
	double rate = ratio * 0.01;
	
	//Mat srcGray(src.size(), CV_8UC1);
	//cvtColor(src, srcGray, COLOR_BGR2GRAY);
	// 通道分离 
	vector<Mat> vSrcImage, vResultImage;
	split(src, vSrcImage);
	Mat resultMat;
	for (int i = 0; i < 3; i++)
	{
		// 分通道转换成浮点型数据
		Mat tempImage;
		vSrcImage[i].convertTo(tempImage, CV_64FC1, 1.0 / 255.0);
		Mat p = tempImage.clone();
		// 分别进行导向滤波       
		Mat resultImage = guidedfilter(tempImage, p, 4, 0.01);
		vResultImage.push_back(resultImage);
	}
	// 通道结果合并   
	merge(vResultImage, resultMat);

	//区域复制
	//Mat img3;
	//resultMat.copyTo(img3);
    //cv::imwrite("filter.jpg", img3 * 255);
	//Mat img11 = imread("filter.jpg", 1);

	Mat img11;
	resultMat.copyTo(img11);
	img11 = img11 * 255;
	img11.convertTo(img11, CV_8UC3);

	IplImage* img1 = &IplImage(img11);
	////////定义矩形区域  

	Mat temp1;
	src.copyTo(temp1);
	IplImage* img2 = &IplImage(temp1);

	//////定义矩形区域  
	//CvRect roi1  = cvRect(0, 0, src.cols, src.rows);
	//////根据给定矩形设置图像的ROI(Region of Interesting)  
	//cvSetImageROI(img1, roi);
	//cvSetImageROI(img2, roi1);
	//////将第一幅图像中的ROI区域拷贝到第二幅图像的感兴趣区域中  
	cvCopy(img1, img2);
	//取消img和img1上的感兴趣区域  
	//cvResetImageROI(img1);
	//cvResetImageROI(img2);

	temp1.copyTo(dst);
	return 0;
}

double __stdcall LocalBuffing(Mat src, double ratio, Mat& dst, short* p)//局部(人脸部位)磨皮
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}

	double rate = ratio * 0.01+0.001;
;
	int x = p[0];
	int y = p[1];
	int w = p[2];
	int h = p[3];
	int expandRows = 0;

	//扩大矩形框的行数
	if (y > h) 
	{
		expandRows = h / 2;
	}
	else
	{
		expandRows = y*2 / 3;
	}

	y = y - expandRows;
	h = h + 2 * expandRows;

	if (y + h > src.rows - 1)
	{
		h = src.rows - 1 - y;
	}

	Mat faceImage = src(Rect(x, y, w, h)); //取出人脸矩形框

	if (faceImage.empty())
	{
		return -1;
	}
	// 通道分离 
	vector<Mat> vSrcImage, vResultImage;
	split(faceImage, vSrcImage);
	Mat resultMat;
	for (int i = 0; i < 3; i++)
	{
		// 分通道转换成浮点型数据
		Mat tempImage;
		vSrcImage[i].convertTo(tempImage, CV_64FC1, 1.0 / 255.0);
		Mat p = tempImage.clone();
		// 分别进行导向滤波       
		Mat resultImage = guidedfilter(tempImage, p, 4, 0.001*rate);
		vResultImage.push_back(resultImage);
	}
	// 通道结果合并   
	merge(vResultImage, resultMat);
	
	Mat img11;
	resultMat.copyTo(img11);
	img11 = img11 * 255;
	img11.convertTo(img11, CV_8UC3);

	Mat temp1;
	src.copyTo(temp1);
	img11.copyTo(temp1(Rect(x, y, w, h)));

	temp1.copyTo(dst);;
	return 0;
}

double  __stdcall GlobalWhitening(Mat src, double ratio, Mat& dst, short* p)//全局美白
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	double rate = 2.6 - 0.002 * ratio;  //美白系数，系数越小，美白效果越明显(1-10)

	Mat dstImage;
	vector<Mat> Channels;
	split(src, Channels);
	Mat B = Channels[0];
	Mat G = Channels[1];
	Mat R = Channels[2];
	double Baver = mean(B)[0];
	double Gaver = mean(G)[0];
	double Raver = mean(R)[0];

	double K = (Baver + Gaver + Raver) / rate; //计算增益系数	
	double Kb = K / Baver;	
	double Kg = K / Gaver;	
	double Kr = K / Raver;

	//白平衡处理后的通道	
	Mat dstB, dstG, dstR;
	dstB = B * Kb;
	dstG = G * Kg;
	dstR = R * Kr;
	vector<Mat> dstChanges;
	dstChanges.push_back(dstB);
	dstChanges.push_back(dstG);
	dstChanges.push_back(dstR);
	merge(dstChanges, dstImage);//合并通道
	dstImage.copyTo(dst);


	//short pResult[] = { 90,159,310,310,105,256,108,295,114,336,124,375,143,408,170,436,203,456,235,474,271,479,304,473,335,
	//	455,365,431,389,399,403,361,408,321,408,279,405,240,138,202,157,183,184,177,213,181,240,190,279,186,
	//	302,175,328,171,353,176,370,193,260,230,260,251,260,273,261,295,234,331,250,332,266,333,281,330,296,
	//	328,168,238,185,233,202,232,218,240,202,243,185,244,300,237,315,227,331,227,347,231,333,238,316,239,
	//	218,387,239,371,255,360,269,364,281,360,299,369,320,382,302,395,285,401,270,403,256,402,240,398,227,
	//	385,256,379,269,380,282,379,310,381,282,379,269,382,255,381 };
	//GlobalBuffing(src, 50, dst); //先进行磨皮
	//double rate = ratio * 0.01;
	//unsigned char* pBuffer = (unsigned char*)malloc(DETECT_BUFFER_SIZE);
	//Mat gray;
	//cvtColor(src, gray, COLOR_BGR2GRAY);
	//auto pResults = facedetect_multiview_reinforce(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step, 1.2f, 2, 48, 0, 1);
	//short* p = (short*)(pResults + 1);
	//int left_landmark = 38;
	//int left_landmark_down = 27;
	//float r_left = sqrt((p[4 + 2 * left_landmark] - p[4+ 2 * left_landmark_down]) * (p[4 + 2 * left_landmark] - p[4 + 2 * left_landmark_down]) + (p[4 + 2 * left_landmark + 1] - p[4 + 2 * left_landmark_down + 1]) * (p[4 + 2 * left_landmark + 1] - p[4 + 2 * left_landmark_down + 1]));
	//Mat data1 = BilinearInterpolation2(dst, p[4 + 2 * left_landmark], p[4 + 2 * left_landmark + 1], rate*25, 10 * r_left);
	//data1.copyTo(dst);
	//free(pBuffer);
	//pBuffer = NULL;

	return 0;

}

double  __stdcall Whitening(Mat src, double ratio, Mat& dst, short* p)//局部美白
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	//double rate = 0.01 * ratio;
	//cvtColor(src, src, COLOR_RGB2BGR);  //通道问题，取消注释即可解决

	Mat temp1;
	src.copyTo(temp1);

	//int x = p[0];
	//int y = p[1];
	//int w = p[2];
	//int h = p[3];
	//Mat faceImage = src(Range(y, y + h), Range(x, x + w)); //取出人脸矩形框
	//float radius = sqrt(faceImage.rows * faceImage.rows / 4 + faceImage.cols * faceImage.cols / 4);
	//Mat data1 = BilinearInterpolation2(faceImage, faceImage.cols / 2, faceImage.rows / 2, rate * 15, radius);
	//data1.copyTo(temp1(Rect(x, y, w, h)));
	//memcpy(dst.data, temp1.data, sizeof(unsigned char) * temp1.rows * temp1.cols * 3);	

	//add at 2021-3-23
	int x1 = p[6 + 2 * 1];
	int y1 = p[6 + 2 * 1 + 1] - 1.2 * (p[6 + 2 * 9 + 1] - p[6 + 2 * 1 + 1]);
	if (y1 < 1)
	{
		y1 = 1;
	}
	int w1 = p[6 + 2 * 16] - p[6 + 2 * 1];
	int h1 = p[6 + 2 * 9 + 1] - (p[6 + 2 * 1 + 1] - 1.2 * (p[6 + 2 * 9 + 1] - p[6 + 2 * 1 + 1]));

	Mat structKernel = getStructuringElement(MORPH_RECT, Size(3, 3));//卷积大小
	Mat imageROI = src(Range(y1, y1 + h1), Range(x1, x1 + w1));

	Mat ycrcb_image;
	cvtColor(imageROI, ycrcb_image, CV_BGR2YCrCb); //首先转换到YCrCb空间
	Mat detect;
	vector<Mat> channels;
	split(ycrcb_image, channels);
	Mat output_mask = channels[1];
	threshold(output_mask, output_mask, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	imageROI.copyTo(detect, output_mask);

	//Mat result;
	//Mat Y, Cr, Cb;

	//Y = channels.at(0);
	//Cr = channels.at(1);
	//Cb = channels.at(2);


	//result.create(imageROI.rows, imageROI.cols, CV_8UC1);

	/*遍历图像，将符合阈值范围的像素设置为255，其余为0*/
	/*for (int j = 1; j < Y.rows - 1; j++)
	{
		uchar* currentCr = Cr.ptr< uchar>(j);
		uchar* currentCb = Cb.ptr< uchar>(j);
		uchar* current = result.ptr< uchar>(j);
		for (int i = 1; i < Y.cols - 1; i++)
		{
			if ((currentCr[i] > 137) && (currentCr[i] < 175) && (currentCb[i] > 100) && (currentCb[i] < 118))
				current[i] = 255;
			else
				current[i] = 0;
		}
	}
	imshow("output_mask", output_mask);
	imshow("output_mask2", result);
	waitKey(0);*/

	Mat img0;
	erode(detect, img0, structKernel, Point(0, 0), 2);//开运算，消除噪点

	//added at 2021-03-21
	vector<Point> faceContours;
	Mat faceMask(src.rows,src.cols, CV_8UC3,Scalar(0,0,0));

	CvPoint** point1 = new CvPoint * [1];
	point1[0] = new CvPoint[19];

	for (size_t j = 0; j < 17; j++)
	{
		Point edgeP;
		edgeP.x = p[6 + 2 * j];
		edgeP.y = p[6 + 2 * j + 1];
		point1[0][j] = edgeP;
	}

	Point edgeP;
	edgeP.x = p[6 + 2 * 16]- w1/15;
	edgeP.y = p[6 + 2 * 24+1]*0.85;

	point1[0][17] = edgeP;

	edgeP.x = p[6 + 2 * 0]+w1/15;
	edgeP.y = p[6 + 2 * 19 + 1] * 0.85;

	point1[0][18] = edgeP;

	/*int k = 0;
	for (size_t j = 26; j >=17; j--)
	{
		Point edgeP;
		edgeP.x = p[6 + 2 * j];
		edgeP.y = p[6 + 2 * j + 1];
		point1[0][17+k] = edgeP;
		k++;
	}*/
	
	IplImage face1 = cvIplImage(faceMask);
	int npts = { 19 };

	cvFillPoly(&face1, point1, &npts, 1,Scalar(255, 255, 255));
	faceContours.clear();
	delete[] point1[0];
	point1[0] = NULL;
	point1 = NULL;

	Mat dstImage = cvarrToMat(&face1); 

	int linePos = 0;  //眉毛所在的行
	int iFlag = 0;
	for (int row = 0; row < dstImage.rows; row++)
	{
		for (int col = 0; col < dstImage.cols; col++)
		{
			if (dstImage.at<Vec3b>(row, col)[0] ==255 && dstImage.at<Vec3b>(row, col)[1] ==255 &&dstImage.at<Vec3b>(row, col)[2] == 255)
			{
				linePos = row;
				iFlag = 1;
				break;
			}
		}
		if (iFlag == 1)
		{
			break;
		}
	}

	for (int i = 0; i < linePos*1.05; i++)
	{
		if (i >= y1 && i < y1 + h1)
		{
			for (int j = 0; j < dstImage.cols; j++)
			{
				if (j >= x1 && j < x1 + w1)
				{
					if (img0.at<Vec3b>(i - y1, j - x1)[0] * img0.at<Vec3b>(i - y1, j - x1)[1] * img0.at<Vec3b>(i - y1, j - x1)[2] != 0)
					{
						dstImage.at<Vec3b>(i, j)[0] = 255;
						dstImage.at<Vec3b>(i, j)[1] = 255;
						dstImage.at<Vec3b>(i, j)[2] = 255;
					}
				}				
			}
		}
	}

	/*imshow("srcImage", dstImage);
	waitKey(0);*/

	//寻找外接矩形

	int rectUp = 0;
	int rectDown = 0;
	int rectLeft = 0;
	int rectRight = 0;
	iFlag = 0;
	for (int i = 0; i < dstImage.rows; i+=2)   //找矩形的上边界
	{
		for (int j = 0; j < dstImage.cols; j+=2)
		{
			if (dstImage.at<Vec3b>(i, j)[0] == 255 && dstImage.at<Vec3b>(i, j)[1] == 255 && dstImage.at<Vec3b>(i, j)[2] == 255)
			{
				rectUp = i*1.1;
				iFlag = 1;
				break;
			}
		}
		if (iFlag == 1)
		{
			break;
		}
	}

	iFlag = 0;
	for (int i = dstImage.rows -1; i>0; i -= 2)   //找矩形的下边界
	{
		for (int j = 0; j < dstImage.cols; j += 2)
		{
			if (dstImage.at<Vec3b>(i, j)[0] == 255 && dstImage.at<Vec3b>(i, j)[1] == 255 && dstImage.at<Vec3b>(i, j)[2] == 255)
			{
				rectDown = i;
				iFlag = 1;
				break;
			}
		}
		if (iFlag == 1)
		{
			break;
		}
	}



	iFlag = 0;
	for (int j = 0; j < dstImage.cols; j += 2)   //找矩形的左边界
	{
		for (int i = 0; i < dstImage.rows; i += 2)
		{
			if (dstImage.at<Vec3b>(i, j)[0] == 255 && dstImage.at<Vec3b>(i, j)[1] == 255 && dstImage.at<Vec3b>(i, j)[2] == 255)
			{
				rectLeft = j*1.12;
				iFlag = 1;
				break;
			}
		}
		if (iFlag == 1)
		{
			break;
		}
	}

	iFlag = 0;
	for (int j = dstImage.cols-1; j >0; j -= 2)   //找矩形的右边界
	{
		for (int i = 0; i < dstImage.rows; i += 2)
		{
			if (dstImage.at<Vec3b>(i, j)[0] == 255 && dstImage.at<Vec3b>(i, j)[1] == 255 && dstImage.at<Vec3b>(i, j)[2] == 255)
			{
				rectRight = j*0.95;
				iFlag = 1;
				break;
			}
		}
		if (iFlag == 1)
		{
			break;
		}
	}

	Rect rect;
	rect.x = rectLeft;
	rect.y = rectUp;
	rect.width = rectRight - rectLeft;
	rect.height = rectDown - rectUp;

	int cX = rect.x + rect.width / 2;
	int cY = rect.y + rect.height / 2;

	//rectangle(dstImage, rect, Scalar(255, 255, 255), 2, 8, 0);
	//imshow("rectImage", dstImage);
	//waitKey(0); 

	/*vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(dstImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(Mat(contours[i]));
		rectangle(dstImage, boundRect[i].tl(), boundRect[i].br(), (0, 0, 255), 2, 8, 0);
	}
	imshow("srcImage", dstImage);
	waitKey(0);*/

	//return 0;

	//Mat result;
	//bitwise_and(src, faceMask, result);

	double rate = 0.01 * ratio;
	for (int row = 0; row < dstImage.rows; row++)
	{
		for (int col = 0; col < dstImage.cols; col++)
		{
			if (dstImage.at<Vec3b>(row, col)[0] == 0 && dstImage.at<Vec3b>(row, col)[1] == 0 && dstImage.at<Vec3b>(row, col)[2] == 0)
			{
				continue;
			}
			temp1.at<Vec3b>(row, col)[0] += (255 - temp1.at<Vec3b>(row, col)[0]) * rate*0.15;
			temp1.at<Vec3b>(row, col)[1] += (255 - temp1.at<Vec3b>(row, col)[1]) * rate*0.15;
			temp1.at<Vec3b>(row, col)[2] += (255 - temp1.at<Vec3b>(row, col)[2]) * rate*0.15;
		}
	}
	//temp1.copyTo(dst);

	//泊松融合

	Mat faceROI = temp1(Range(rect.y, rect.y+rect.height), Range(rect.x, rect.x + rect.width));
	Mat src_mask = 255 * Mat::ones(faceROI.rows, faceROI.cols, faceROI.depth());
	// The location of the center of the src in the dst
	Point center(cX, cY);
	// Seamlessly clone src into dst and put the results in output
	//Mat normal_clone;
	Mat mixed_clone;
	Mat monochrome_clone;
	//seamlessClone(faceROI, src, src_mask, center, normal_clone, NORMAL_CLONE);
	seamlessClone(faceROI, src, src_mask, center, mixed_clone, MIXED_CLONE);
	//seamlessClone(faceROI, src, src_mask, center, monochrome_clone, MONOCHROME_TRANSFER);

	//imshow("normal_clone", normal_clone);
	//imshow("minxed_clone", mixed_clone);
	//imshow("monochrome_clone", monochrome_clone);
	//waitKey(0);
	mixed_clone.copyTo(dst);
	return 0;

	////融合图像
	//float tmp = 0;
	//Mat dstH(src.size(), CV_8UC3);//RGB3通道就用CV_8UC3 高反差结果 H=F-I+128
	//int width = dst.cols;
	//int height = dst.rows;

	//for (int y = 0; y < height; y++)
	//{
	//	uchar* srcP = src.ptr<uchar>(y);
	//	uchar* lvboP = temp1.ptr<uchar>(y);
	//	uchar* dstHP = dstH.ptr<uchar>(y);

	//	for (int x = 0; x < width; x++)
	//	{
	//		float r0 = abs((float)lvboP[3 * x] - (float)srcP[3 * x]);
	//		tmp = abs(r0 + 128);
	//		tmp = tmp > 255 ? 255 : tmp;
	//		tmp = tmp < 0 ? 0 : tmp;
	//		dstHP[3 * x] = (uchar)(tmp);

	//		float r1 = abs((float)lvboP[3 * x + 1] - (float)srcP[3 * x + 1]);
	//		tmp = abs(r1 + 128);
	//		tmp = tmp > 255 ? 255 : tmp;
	//		tmp = tmp < 0 ? 0 : tmp;
	//		dstHP[3 * x + 1] = (uchar)(tmp);

	//		float r2 = abs((float)lvboP[3 * x + 2] - (float)srcP[3 * x + 2]);
	//		tmp = abs(r2 + 128);
	//		tmp = tmp > 255 ? 255 : tmp;
	//		tmp = tmp < 0 ? 0 : tmp;
	//		dstHP[3 * x + 2] = (uchar)(tmp);
	//	}
	//}
	//Mat dstY(dstH.size(), CV_8UC3);
	//int ksize = 3;
	//GaussianBlur(dstH, dstY, Size(ksize, ksize), 0, 0, 0); //高斯滤波得到Y 
	//Mat dstZ(src.size(), CV_8UC3);//Z =  X * Op + (X + 2 * Y - 256)* Op= X  + (2*Y-256) *Op  OP不透明度 X原图 Y是高斯滤波后图像
	//float OP = 0.05;//不透明度
	//for (int y = 0; y < height; y++) //图层混合
	//{
	//	uchar* XP = src.ptr<uchar>(y);
	//	uchar* dstYP = dstY.ptr<uchar>(y);
	//	uchar* dstZP = dstZ.ptr<uchar>(y);

	//	for (int x = 0; x < width; x++)
	//	{
	//		float r3 = ((float)dstYP[3 * x] + (float)dstYP[3 * x] - 256) * OP;
	//		tmp = r3 + (float)XP[3 * x];
	//		tmp = tmp > 255 ? 255 : tmp;
	//		tmp = tmp < 0 ? 0 : tmp;
	//		dstZP[3 * x] = (uchar)(tmp);

	//		float r4 = ((float)dstYP[3 * x + 1] + (float)dstYP[3 * x + 1] - 256) * OP;
	//		tmp = r4 + (float)XP[3 * x + 1];
	//		tmp = tmp > 255 ? 255 : tmp;
	//		tmp = tmp < 0 ? 0 : tmp;
	//		dstZP[3 * x + 1] = (uchar)(tmp);

	//		float r5 = ((float)dstYP[3 * x + 2] + (float)dstYP[3 * x + 2] - 256) * OP;
	//		tmp = r5 + (float)XP[3 * x + 2];
	//		tmp = tmp > 255 ? 255 : tmp;
	//		tmp = tmp < 0 ? 0 : tmp;
	//		dstZP[3 * x + 2] = (uchar)(tmp);
	//	}
	//}
	//dstZ.copyTo(dst);
	//return 0;

	//free(pBuffer);
	//pBuffer = NULL;
	//Mat dstFaceImg;
	//faceImage.copyTo(dstFaceImg);
	//for (int row = 0; row < faceImage.rows; row++)
	//{
	//	for (int col = 0; col < faceImage.cols; col++)
	//	{
	//		faceImage.at<Vec3b>(row, col)[0] = faceImage.at<Vec3b>(row, col)[0] + (255 - faceImage.at<Vec3b>(row, col)[0]) * rate * 0.05;
	//		faceImage.at<Vec3b>(row, col)[1] = faceImage.at<Vec3b>(row, col)[1] + (255 - faceImage.at<Vec3b>(row, col)[1]) * rate * 0.05;
	//		faceImage.at<Vec3b>(row, col)[2] = faceImage.at<Vec3b>(row, col)[2] + (255 - faceImage.at<Vec3b>(row, col)[2]) * rate * 0.05;
	//	}
	//}

	//Mat temp1(src.rows,src.cols,src.type());
	////src.copyTo(temp1);
	//faceImage.copyTo(temp1(Rect(x, y, w, h)));
	//temp1.copyTo(dst);
	//free(pBuffer);
	//pBuffer = NULL;
	//return 0;

	//FILE* fp = fopen("test2.txt", "w");
	//fprintf(fp, "srcH:%d\n", gray.rows);
	//fprintf(fp, "srcW:%d\n", gray.cols);
	//fprintf(fp, "x:%d\n", x);
	//fprintf(fp, "y:%d\n", y);
	//fprintf(fp, "w:%d\n", w);
	//fprintf(fp, "h:%d\n", h);
	//fprintf(fp, "new_srcH:%d\n", faceImage.rows);
	//fprintf(fp, "new_srcW:%d\n", faceImage.cols);
	//fclose(fp);
	//fp = NULL;
	//Mat faceImage = src(Range(y, y + h), Range(x, x + w)); //取出人脸矩形框

	//double rate = 1-  0.01*ratio;  //美白系数，系数越小，美白效果越明显(1-10)
	//double tempRate = rate*0.5 + 2.65;
	//Mat dstImage;
	//vector<Mat> Channels;
	//split(faceImage, Channels);
	//Mat B = Channels[0];
	//Mat G = Channels[1];
	//Mat R = Channels[2];
	//double Baver = mean(B)[0];	
	//double Gaver = mean(G)[0];	
	//double Raver = mean(R)[0];
	////beta = 2;
	//double K = (Baver + Gaver + Raver) / tempRate; //计算增益系数	
	//double Kb, Kg, Kr;
	//Kb = K / Baver;	Kg = K / Gaver;	Kr = K / Raver;
	////白平衡处理后的通道	
	//Mat dstB, dstG, dstR;
	//dstB = B * Kb;	
	//dstG = G * Kg;	
	//dstR = R * Kr;
	//vector<Mat> dstChanges;
	//dstChanges.push_back(dstB);	
	//dstChanges.push_back(dstG);	
	//dstChanges.push_back(dstR);
	//merge(dstChanges, dstImage);//合并通道
	////dstImage.copyTo(dst);

	//Mat temp1;
	//src.copyTo(temp1);
	//dstImage.copyTo(temp1(Rect(x, y, w, h)));

	//memcpy(dst.data, temp1.data, sizeof(unsigned char) * temp1.rows * temp1.cols * 3);
	//temp1.copyTo(dst);
	//free(pBuffer);
	//pBuffer = NULL;
	//return 0;
}

double  __stdcall AdjustLip(Mat src, double ratio, Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	double rate = ratio * 0.015;

	Mat data = BilinearInterpolation(src, p[6 + 2 * 48], p[6 + 2 * 48 + 1], p[6 + 2 * 48] - 15, p[6 + 2 * 48 + 1], (p[6 + 2 * 59 + 1] - p[6 + 2 * 48 + 1]) * rate);
	Mat data1 = BilinearInterpolation(data, p[6 + 2 * 54], p[6 + 2 * 54 + 1], p[6 + 2 * 54] + 15, p[6 + 2 * 54 + 1], (p[6 + 2 * 55 + 1] - p[6 + 2 * 54 + 1]) * rate);

	data1.copyTo(dst);
	return 0;
}

double  __stdcall AdjustForeHead(Mat src, double ratio, Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	double rate = ratio * 0.01;

	//int a = rate * (4 * (p[6 + 2 * 38 + 1] - p[6 + 2 * 20 + 1]) / 5);
	//int b = (p[6 + 2 * 38 + 1] - p[6 + 2 * 20 + 1])*0.6;

	//Mat data2 = BilinearInterpolation(src, p[6 + 2 * 20], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 20], p[6 + 2 * 20 + 1], a);
	//Mat data3 = BilinearInterpolation(data2, p[6 + 2 * 23], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 23], p[6 + 2 * 23 + 1], a);
	//Mat data4 = BilinearInterpolation(data3, p[6 + 2 * 21], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 21], p[6 + 2 * 21 + 1], a);
	//Mat data5 = BilinearInterpolation(data4, p[6 + 2 * 22], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 22], p[6 + 2 * 22 + 1], a);
	//Mat data6 = BilinearInterpolation(data5, p[6 + 2 * 28], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 28], p[6 + 2 * 28 + 1], a);


	int a1 = rate * (p[6 + 2 * 37 + 1] - p[6 + 2 * 19 + 1]) * 0.6;
	int b1 = (p[6 + 2 * 37 + 1] - p[6 + 2 * 19 + 1])*0.65;

	int a2 = rate * (p[6 + 2 * 44 + 1] - p[6 + 2 * 24 + 1]) * 0.6;
	int b2 = (p[6 + 2 * 44 + 1] - p[6 + 2 * 24 + 1])*0.65;

	Mat data1 = BilinearInterpolation(src, p[6 + 2 * 18], p[6 + 2 * 19 + 1] - b1, p[6 + 2 * 18], p[6 + 2 * 19 + 1], a1);
	Mat data2 = BilinearInterpolation(data1, p[6 + 2 * 19], p[6 + 2 * 19 + 1] - b1, p[6 + 2 * 19], p[6 + 2 * 19 + 1], a1);
	Mat data3 = BilinearInterpolation(data2, p[6 + 2 * 20], p[6 + 2 * 19 + 1] - b1, p[6 + 2 * 20], p[6 + 2 * 19 + 1], a1);
	Mat datat = BilinearInterpolation(data3, p[6 + 2 * 21], p[6 + 2 * 19 + 1] - b1, p[6 + 2 * 21], p[6 + 2 * 19 + 1], a1);


	Mat data4 = BilinearInterpolation(datat, p[6 + 2 * 25], p[6 + 2 * 24 + 1] - b2, p[6 + 2 * 25], p[6 + 2 * 24 + 1], a2);
	Mat data5 = BilinearInterpolation(data4, p[6 + 2 * 24], p[6 + 2 * 24 + 1] - b2, p[6 + 2 * 24], p[6 + 2 * 24 + 1], a2);
	Mat data6 = BilinearInterpolation(data5, p[6 + 2 * 23], p[6 + 2 * 24 + 1] - b2, p[6 + 2 * 23], p[6 + 2 * 24 + 1], a2);
	Mat data7 = BilinearInterpolation(data6, p[6 + 2 * 22], p[6 + 2 * 24 + 1] - b2, p[6 + 2 * 22], p[6 + 2 * 24 + 1], a2);


	//Mat data = BilinearInterpolation(src, p[6 + 2 * 18], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 18], p[6 + 2 * 18 + 1], a);
	//Mat data1 = BilinearInterpolation(data, p[6 + 2 * 20], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 20], p[6 + 2 * 20 + 1], a);
	//Mat data2 = BilinearInterpolation(data1, p[6 + 2 * 22], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 22], p[6 + 2 * 22 + 1], a);
	//Mat data3 = BilinearInterpolation(data2, p[6 + 2 * 23], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 23], p[6 + 2 * 23 + 1], a);
	//Mat data5 = BilinearInterpolation(data3, p[6 + 2 * 25], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 25], p[6 + 2 * 25 + 1], a);
	//Mat data6 = BilinearInterpolation(data5, p[6 + 2 * 28], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 28], p[6 + 2 * 28 + 1], a);
	//Mat data7 = BilinearInterpolation(data6, p[6 + 2 * 27], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 27], p[6 + 2 * 27 + 1], a);
	//int a = rate * (p[6 + 2 * 38 + 1] - p[6 + 2 * 20 + 1])*0.75;
	//int b = (p[6 + 2 * 38 + 1] - p[6 + 2 * 20 + 1]);
	//Mat data1 = BilinearInterpolation(src,p[6 + 2 * 20], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 20], p[6 + 2 * 20 + 1], a);
	//Mat data2 = BilinearInterpolation(data1, p[6 + 2 * 25], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 25], p[6 + 2 * 25 + 1], a);
	//Mat data3 = BilinearInterpolation(data2, p[6 + 2 * 28], p[6 + 2 * 20 + 1] - b, p[6 + 2 * 28], p[6 + 2 * 28 + 1], a);


	//float radius_left_eye = sqrt((p[6 + 2 * 19] - p[6 + 2 * 37]) * (p[6 + 2 * 19] - p[6 + 2 * 37]) + (p[6 + 2 * 19 + 1] - p[6 + 2 * 37 + 1]) * (p[6 + 2 * 19 + 1] - p[6 + 2 * 37 + 1]));
	//float radius_right_eye = sqrt((p[6 + 2 * 24] - p[6 + 2 * 44]) * (p[6 + 2 * 24] - p[6 + 2 * 44]) + (p[6 + 2 * 24 + 1] - p[6 + 2 * 44 + 1]) * (p[6 + 2 * 24 + 1] - p[6 + 2 * 44 + 1]));

	/*int radius_left_eye = abs(p[6 + 2 * 19 + 1] - p[6 + 2 * 37 + 1]);*/

	//Mat data = BilinearInterpolation(src, p[6 + 2 * 19], (int)(p[6 + 2 * 19 + 1] * 0.85), p[6 + 2 * 19], (int)(p[6 + 2 * 19 + 1] + radius_left_eye * 0.25), rate * radius_left_eye * 0.8);
	//Mat data1 = BilinearInterpolation(data, p[6 + 2 * 24], (int)(p[6 + 2 * 24 + 1] * 0.85), p[6 + 2 * 24], (int)(p[6 + 2 * 24 + 1] + radius_right_eye * 0.25), rate * radius_right_eye * 0.8);

	//add at 2021-3-20
	//int leftRegionUpRow = (int)(p[6 + 2 * 19 + 1] - radius_left_eye) >0 ? (int)(p[6 + 2 * 19 + 1] - radius_left_eye) : (int)(p[6 + 2 * 19 + 1]/3); //眉毛至睫毛额距离为H,再向上2H表示额头区域
	//int rightRegionUpRow = (int)(p[6 + 2 * 24 + 1] - radius_right_eye) > 0 ? (int)(p[6 + 2 * 24 + 1] - radius_right_eye) : (int)(p[6 + 2 * 24 + 1] / 3);; //眉毛至睫毛额距离为H,再向上2H表示额头区域

	//int regionTopRow = (int)(p[6 + 2 * 19 + 1] - radius_left_eye*1.5) > 0 ? (int)(p[6 + 2 * 19 + 1] - radius_left_eye*1.5) : (int)(p[6 + 2 * 19 + 1] / 3);
	//int regionBottomRow = p[6 + 2 * 19 + 1] -5;

	//int regionLeftCol = p[6 + 2 * 17];
	//int regionRightCol = p[6 + 2 * 26];


	////int moveLines = (int)rate;
	////Mat data = BilinearInterpolation(src, p[6 + 2 * 19], (int)(leftRegionUpRow*1.5), p[6 + 2 * 19], p[6 + 2 * 19 + 1], rate * radius_left_eye*0.75);
	////Mat data1 = BilinearInterpolation(data, p[6 + 2 * 24], (int)(rightRegionUpRow*1.5), p[6 + 2 * 24], p[6 + 2 * 24 + 1], rate * radius_right_eye*0.75);

	//Mat data1 = ForeHeadHandle(src, regionTopRow, regionBottomRow, regionLeftCol, regionRightCol, rate);


	data7.copyTo(dst);
	return 0;

}

double __stdcall ColourCorrect(Mat src, double ratio, Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}

	Mat srcLab;
	cvtColor(src, srcLab, COLOR_BGR2Lab);  //颜色空间转换

	//int x = p[0];
	//int y = p[1];
	//int w = p[2];
	//int h = p[3];
	//Mat faceImage = src(Range(y, y + h), Range(x, x + w)); //取出人脸矩形框

	//added at 2021-03-21 	//定位人脸区域
	int x1 = p[6 + 2 * 1];
	int y1 = p[6 + 2 * 1 + 1] - 1.2 * (p[6 + 2 * 9 + 1] - p[6 + 2 * 1 + 1]);
	int w1 = p[6 + 2 * 16] - p[6 + 2 * 1];
	int h1 = p[6 + 2 * 9 + 1] - (p[6 + 2 * 1 + 1] - 1.2 * (p[6 + 2 * 9 + 1] - p[6 + 2 * 1 + 1]));

	Mat structKernel = getStructuringElement(MORPH_RECT, Size(3, 3));//卷积大小
	Mat imageROI = src(Range(y1, y1 + h1), Range(x1, x1 + w1));

	Mat ycrcb_image;
	cvtColor(imageROI, ycrcb_image, CV_BGR2YCrCb); //首先转换到YCrCb空间
	Mat detect;
	vector<Mat> channels;
	split(ycrcb_image, channels);
	Mat output_mask = channels[1];
	threshold(output_mask, output_mask, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	imageROI.copyTo(detect, output_mask);
	Mat img0;
	erode(detect, img0, structKernel, Point(0, 0), 2);//开运算，消除噪点

	vector<Point> faceContours;
	Mat faceMask(src.rows, src.cols, CV_8UC3, Scalar(0, 0, 0));

	CvPoint** point1 = new CvPoint * [1];
	point1[0] = new CvPoint[19];

	for (size_t j = 0; j < 17; j++)
	{
		Point edgeP;
		edgeP.x = p[6 + 2 * j];
		edgeP.y = p[6 + 2 * j + 1];
		point1[0][j] = edgeP;
	}

	int temp_dist = p[6 + 2 * 8 + 1] - p[6 + 2 * 27 + 1];

	Point edgeP;
	edgeP.x = p[6 + 2 * 16]-w1/10;
	edgeP.y = p[6 + 2 * 24 + 1] -temp_dist/5;
	point1[0][17] = edgeP;

	edgeP.x = p[6 + 2 * 0]+w1/10;
	edgeP.y = p[6 + 2 * 19 + 1] - temp_dist / 5;
	point1[0][18] = edgeP;

	/*int k = 0;
	for (size_t j = 26; j >=17; j--)
	{
		Point edgeP;
		edgeP.x = p[6 + 2 * j];
		edgeP.y = p[6 + 2 * j + 1];
		point1[0][17+k] = edgeP;
		k++;
	}*/

	IplImage face1 = cvIplImage(faceMask);
	int npts = { 19 };

	cvFillPoly(&face1, point1, &npts, 1, Scalar(255, 255, 255));
	faceContours.clear();
	delete[] point1[0];
	point1[0] = NULL;
	point1 = NULL;

	int linePos = 0;  //眉毛所在的行
	int iFlag = 0;
	for (int row = 0; row < faceMask.rows; row++)
	{
		for (int col = 0; col < faceMask.cols; col++)
		{
			if (faceMask.at<Vec3b>(row, col)[0] == 255 && faceMask.at<Vec3b>(row, col)[1] == 255 && faceMask.at<Vec3b>(row, col)[2] == 255)
			{
				linePos = row;
				iFlag = 1;
				break;
			}
		}
		if (iFlag == 1)
		{
			break;
		}
	}

	for (int i = 0; i < linePos*1.05; i++)
	{
		if (i >= y1 && i < y1 + h1)
		{
			for (int j = 0; j < faceMask.cols; j++)
			{
				if (j >= x1 && j < x1 + w1)
				{
					if (img0.at<Vec3b>(i - y1, j - x1)[0] * img0.at<Vec3b>(i - y1, j - x1)[1] * img0.at<Vec3b>(i - y1, j - x1)[2] != 0)
					{
						faceMask.at<Vec3b>(i, j)[0] = 255;
						faceMask.at<Vec3b>(i, j)[1] = 255;
						faceMask.at<Vec3b>(i, j)[2] = 255;
					}
				}
			}
		}
	}

	Mat result;
	bitwise_and(src, faceMask, result);

	double rate = 0.35+ 0.005 * ratio; 

	Mat dstImage;
	vector<Mat> Channels;
	split(result, Channels);
	Mat B = Channels[0];
	Mat G = Channels[1];
	Mat R = Channels[2];
	double Baver = mean(B)[0];
	double Gaver = mean(G)[0];
	double Raver = mean(R)[0];

	double K = (Baver + Gaver + Raver) / 3;

	double Kb = K / Baver;	
	double Kg = K / Gaver;	
	double Kr = K / Raver;

	/*R * 0.299 + G * 0.587 + B * 0.114*/

	//白平衡处理后的通道	
	Mat dstB = B * Kb* rate;
	Mat dstG = G * Kg* rate;
	Mat dstR = R * Kr* rate;
	vector<Mat> dstChanges;

	dstChanges.push_back(dstB);
	dstChanges.push_back(dstG);
	dstChanges.push_back(dstR);
	merge(dstChanges, dstImage); //合并通道


	uchar* pImg = srcLab.data;
	// 计算颜色转换值
	for (int i = 0; i < srcLab.rows; i++)
	{
		for (int j = 0; j < srcLab.cols; j++)
		{

			if (faceMask.at<Vec3b>(i, j)[0] == 0 && faceMask.at<Vec3b>(i, j)[1] == 0 && faceMask.at<Vec3b>(i, j)[2] == 0)
			{
				continue;
			}

			//pImg[3 * j + 0] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 0]+ rate * 4));
			//pImg[3 * j + 1] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 1] + rate * 5));
			//pImg[3 * j + 2] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 2]+ rate * 3));


			pImg[3 * j + 0] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 0]));
			pImg[3 * j + 1] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 1] + rate * 5));
			pImg[3 * j + 2] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 2]));


		}
		pImg += srcLab.step;
	}

	Mat temp1;
	cvtColor(srcLab, temp1, COLOR_Lab2BGR);

	for (int row = 0; row < dstImage.rows; row++)
	{
		for (int col = 0; col < dstImage.cols; col++)
		{
			if (faceMask.at<Vec3b>(row, col)[0] == 0 && faceMask.at<Vec3b>(row, col)[1] == 0 && faceMask.at<Vec3b>(row, col)[2] == 0)
			{
				continue;
			}
			//temp1.at<Vec3b>(row, col)[0] = dstImage.at<Vec3b>(row, col)[0];
			//temp1.at<Vec3b>(row, col)[1] = dstImage.at<Vec3b>(row, col)[1];
			//temp1.at<Vec3b>(row, col)[2] = dstImage.at<Vec3b>(row, col)[2];

			temp1.at<Vec3b>(row, col)[0] += (255 - temp1.at<Vec3b>(row, col)[0]) * rate*0.75 ;
			temp1.at<Vec3b>(row, col)[1] += (255 - temp1.at<Vec3b>(row, col)[1]) * rate*0.75 ;
			temp1.at<Vec3b>(row, col)[2] += (255 - temp1.at<Vec3b>(row, col)[2]) * rate*0.75;

		}
	}
	//temp1.copyTo(dst);

	//融合图像
	float tmp = 0;
	Mat dstH(src.size(), CV_8UC3);//RGB3通道就用CV_8UC3 高反差结果 H=F-I+128
	int width = dst.cols;
	int height = dst.rows;

	for (int y = 0; y < height; y++)
	{
		uchar* srcP = src.ptr<uchar>(y);
		uchar* lvboP = temp1.ptr<uchar>(y);
		uchar* dstHP = dstH.ptr<uchar>(y);

		for (int x = 0; x < width; x++)
		{
			float r0 = abs((float)lvboP[3 * x] - (float)srcP[3 * x]);
			tmp = abs(r0 + 128);
			tmp = tmp > 255 ? 255 : tmp;
			tmp = tmp < 0 ? 0 : tmp;
			dstHP[3 * x] = (uchar)(tmp);

			float r1 = abs((float)lvboP[3 * x + 1] - (float)srcP[3 * x + 1]);
			tmp = abs(r1 + 128);
			tmp = tmp > 255 ? 255 : tmp;
			tmp = tmp < 0 ? 0 : tmp;
			dstHP[3 * x + 1] = (uchar)(tmp);

			float r2 = abs((float)lvboP[3 * x + 2] - (float)srcP[3 * x + 2]);
			tmp = abs(r2 + 128);
			tmp = tmp > 255 ? 255 : tmp;
			tmp = tmp < 0 ? 0 : tmp;
			dstHP[3 * x + 2] = (uchar)(tmp);
		}
	}
	Mat dstY(dstH.size(), CV_8UC3);
	int ksize = 3;
	GaussianBlur(dstH, dstY, Size(ksize, ksize), 0, 0, 0); //高斯滤波得到Y 
	Mat dstZ(src.size(), CV_8UC3);//Z =  X * Op + (X + 2 * Y - 256)* Op= X  + (2*Y-256) *Op  OP不透明度 X原图 Y是高斯滤波后图像
	float OP = 0.03;//不透明度
	for (int y = 0; y < height; y++) //图层混合
	{
		uchar* XP = src.ptr<uchar>(y);
		uchar* dstYP = dstY.ptr<uchar>(y);
		uchar* dstZP = dstZ.ptr<uchar>(y);

		for (int x = 0; x < width; x++)
		{
			float r3 = ((float)dstYP[3 * x] + (float)dstYP[3 * x] - 256) * OP;
			tmp = r3 + (float)XP[3 * x];
			tmp = tmp > 255 ? 255 : tmp;
			tmp = tmp < 0 ? 0 : tmp;
			dstZP[3 * x] = (uchar)(tmp);

			float r4 = ((float)dstYP[3 * x + 1] + (float)dstYP[3 * x + 1] - 256) * OP;
			tmp = r4 + (float)XP[3 * x + 1];
			tmp = tmp > 255 ? 255 : tmp;
			tmp = tmp < 0 ? 0 : tmp;
			dstZP[3 * x + 1] = (uchar)(tmp);

			float r5 = ((float)dstYP[3 * x + 2] + (float)dstYP[3 * x + 2] - 256) * OP;
			tmp = r5 + (float)XP[3 * x + 2];
			tmp = tmp > 255 ? 255 : tmp;
			tmp = tmp < 0 ? 0 : tmp;
			dstZP[3 * x + 2] = (uchar)(tmp);
		}
	}
	dstZ.copyTo(dst);

	return 0;






	//double tempRate = 0.1*ratio;  //美白系数，系数越小，美白效果越明显(1-10)
	////double tempRate =4;
	//Mat dstImage;
	//vector<Mat> Channels;
	//split(src, Channels);
	//Mat B = Channels[0];
	//Mat G = Channels[1];
	//Mat R = Channels[2];
	//double Baver = mean(B)[0];	
	//double Gaver = mean(G)[0];	
	//double Raver = mean(R)[0];
	////double K = (Baver + Gaver + Raver) / tempRate; //计算增益系数	
	//double Kb, Kg, Kr;
	////Kb = K / Baver;	Kg = K / Gaver;	Kr = K / Raver;
	//Kb = 1+ tempRate/ Baver;	Kg = 1+ tempRate/ Gaver;	Kr = 1+ tempRate/ Raver;

	////白平衡处理后的通道	
	//Mat dstB, dstG, dstR;
	//dstB = B * Kb*0.5;	
	//dstG = G * Kg*0.5;	
	//dstR = R * Kr*0.5;
	//vector<Mat> dstChanges;
	//dstChanges.push_back(dstB);	
	//dstChanges.push_back(dstG);	
	//dstChanges.push_back(dstR);
	//merge(dstChanges, dstImage); //合并通道
	//
	////dstImage.copyTo(dst(Rect(x, y, w, h)));
	//dstImage.copyTo(dst);
}

double __stdcall Sharpen(Mat src, double ratio, Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}

	src.copyTo(dst);
	Mat imgBlur, imgLow;
	imgLow.create(src.size(), CV_8UC1);

	int thre = 0;       //改变 1.阀值 2.Amount 值的变化对图像显示的效果存在很大影响  敏感数字
	double radius = 0.5*ratio;    //3. 半径  半径的变化会对应对比度的变化
	double amount = 3;

	GaussianBlur(src, imgBlur, Size(3, 3), radius, radius);  //高斯低通滤波  注意半径改变的是标准差的值SIGMMA  Size 设置3,3 太小的话效果不好
	imgLow = abs(src - imgBlur) < thre;     //掩膜计算  原图像-低通==高通滤波的值   和阀值比较 小于的则对应的值设为1 其他为0
	dst = src * (1 + amount) + imgBlur * (-amount);  //原图形+高通的值*amount
	src.copyTo(dst, imgLow);                //拷贝小于阀值的像素的值

	return 0;
}

double  __stdcall EnhanceRed(Mat src, double ratio, Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	double rate =  0.25+ratio * 0.005;
	
	Vec3f srcMean3f, tarMean3f;// 源/目标图像均值
	Vec3f srcVariance3f, tarVariance3f;// 源/目标图像标准差
	Vec3f ratioVariance3f;// 标准差比例


	//uchar* pImg1 = src.data;
	//FILE* pFile1 = fopen("src.txt", "w");
	//pImg1 += src.step * 50;
	//for (int i = 50; i < 60; i++)
	//{
	//	for (int j = 50; j < 60; j++)
	//	{
	//		fprintf(pFile1, "%d\n", (uchar)(pImg1[3 * j + 0]));
	//		fprintf(pFile1, "%d\n", (uchar)(pImg1[3 * j + 1]));
	//		fprintf(pFile1, "%d\n", (uchar)(pImg1[3 * j + 2]));
	//	}
	//	pImg1+= src.step;
	//}
	//fclose(pFile1);
	//pFile1 = NULL;

	//cvtColor(src, src, COLOR_RGB2BGR);  //通道问题，取消注释即可解决

	// BGR空间转Lab空间
	Mat srcLab;
	cvtColor(src, srcLab, COLOR_BGR2Lab);

	// 计算当前图像与目标图像均值及标准差
	CompMeanAndVariance(srcLab, srcMean3f, srcVariance3f);

	tarVariance3f[0] = 34.285606;
	tarVariance3f[1] = 6.891623;
	tarVariance3f[2] = 4.424289;
	tarMean3f[0] = 207.746002;
	tarMean3f[1] = 155.629791;
	//tarMean3f[1] = 169.77;
	tarMean3f[2] = 135.385712;

	// 标准差比
	ratioVariance3f[0] = rate*tarVariance3f[0] / srcVariance3f[0];
	ratioVariance3f[1] = rate*tarVariance3f[1] / srcVariance3f[1];
	ratioVariance3f[2] = rate*tarVariance3f[2] / srcVariance3f[2];

	//added at 2021-03-21 	//定位人脸区域
	int x1 = p[6 + 2 * 1];
	int y1 = p[6 + 2 * 1 + 1] - 1.2 * (p[6 + 2 * 9 + 1] - p[6 + 2 * 1 + 1]);
	int w1 = p[6 + 2 * 16] - p[6 + 2 * 1];
	int h1 = p[6 + 2 * 9 + 1] - (p[6 + 2 * 1 + 1] - 1.2 * (p[6 + 2 * 9 + 1] - p[6 + 2 * 1 + 1]));

	Mat structKernel = getStructuringElement(MORPH_RECT, Size(3, 3));//卷积大小
	Mat imageROI = src(Range(y1, y1 + h1), Range(x1, x1 + w1));
	Mat ycrcb_image;
	cvtColor(imageROI, ycrcb_image, CV_BGR2YCrCb); //首先转换到YCrCb空间
	Mat detect;
	vector<Mat> channels;
	split(ycrcb_image, channels);
	Mat output_mask = channels[1];
	threshold(output_mask, output_mask, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	imageROI.copyTo(detect, output_mask);
	Mat img0;
	erode(detect, img0, structKernel, Point(0, 0), 2);//开运算，消除噪点

	vector<Point> faceContours;
	Mat faceMask(src.rows, src.cols, CV_8UC3, Scalar(0, 0, 0));

	CvPoint** point1 = new CvPoint * [1];
	point1[0] = new CvPoint[19];

	for (size_t j = 0; j < 17; j++)
	{
		Point edgeP;
		edgeP.x = p[6 + 2 * j];
		edgeP.y = p[6 + 2 * j + 1];
		point1[0][j] = edgeP;
	}

	Point edgeP;
	edgeP.x = p[6 + 2 * 16] - w1 / 15;
	edgeP.y = p[6 + 2 * 24 + 1] * 0.85;

	point1[0][17] = edgeP;

	edgeP.x = p[6 + 2 * 0] + w1 / 15;
	edgeP.y = p[6 + 2 * 19 + 1] * 0.85;

	point1[0][18] = edgeP;

	/*int k = 0;
	for (size_t j = 26; j >=17; j--)
	{
		Point edgeP;
		edgeP.x = p[6 + 2 * j];
		edgeP.y = p[6 + 2 * j + 1];
		point1[0][17+k] = edgeP;
		k++;
	}*/

	IplImage face1 = cvIplImage(faceMask);
	int npts = { 19 };

	cvFillPoly(&face1, point1, &npts, 1, Scalar(255, 255, 255));
	faceContours.clear();

	delete[] point1[0];
	point1[0] = NULL;
	point1 = NULL;

	//Mat dstImage = cvarrToMat(&face1);

	int linePos = 0;  //眉毛所在的行
	int iFlag = 0;
	for (int row = 0; row < faceMask.rows; row++)
	{
		for (int col = 0; col < faceMask.cols; col++)
		{
			if (faceMask.at<Vec3b>(row, col)[0] == 255 && faceMask.at<Vec3b>(row, col)[1] == 255 && faceMask.at<Vec3b>(row, col)[2] == 255)
			{
				linePos = row;
				iFlag = 1;
				break;
			}
		}
		if (iFlag == 1)
		{
			break;
		}
	}

	for (int i = 0; i < linePos*1.05; i++)
	{
		if (i >= y1 && i < y1 + h1)
		{
			for (int j = 0; j < faceMask.cols; j++)
			{
				if (j >= x1 && j < x1 + w1)
				{
					if (img0.at<Vec3b>(i - y1, j - x1)[0] * img0.at<Vec3b>(i - y1, j - x1)[1] * img0.at<Vec3b>(i - y1, j - x1)[2] != 0)
					{
						faceMask.at<Vec3b>(i, j)[0] = 255;
						faceMask.at<Vec3b>(i, j)[1] = 255;
						faceMask.at<Vec3b>(i, j)[2] = 255;
					}
				}
			}
		}
	}
	uchar* pImg = srcLab.data;
	// 计算颜色转换值
	for (int i = 0; i < srcLab.rows; i++)
	{
		for (int j = 0; j < srcLab.cols; j++)
		{

			if (faceMask.at<Vec3b>(i, j)[0] == 0 && faceMask.at<Vec3b>(i, j)[1] == 0 && faceMask.at<Vec3b>(i, j)[2] == 0)
			{
				continue;
			}

			//pImg[3 *j + 0] = (uchar)min_uchar(255, max_uchar(0, ratioVariance3f[0] * (pImg[3 * j + 0] - srcMean3f[0]) + tarMean3f[0]));
			//pImg[3 *j + 1] = (uchar)min_uchar(255, max_uchar(0, 2*ratioVariance3f[1] * (pImg[3 * j + 1] - srcMean3f[1]) + tarMean3f[1]));
			//pImg[3 *j + 2] = (uchar)min_uchar(255, max_uchar(0, 2*ratioVariance3f[2] * (pImg[3 * j + 2] - srcMean3f[2]) + tarMean3f[2]));

		    pImg[3 *j + 0] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 0]));
			pImg[3 *j + 1] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 1]+ rate*5));
			pImg[3 *j + 2] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 2]));

		}
		pImg += srcLab.step;
	}
	
	cvtColor(srcLab, dst, COLOR_Lab2BGR);

	
	return 0;
}

double  __stdcall FrozenFilter(Mat src, Mat& dst, short* p) //冰冻滤镜
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}

	src.copyTo(dst);

	for (size_t i = 0; i < src.rows; i++)
	{
		for (size_t j = 0; j < src.cols; j++)
		{

			dst.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(std::abs(src.at<cv::Vec3b>(i, j)[0] - src.at<cv::Vec3b>(i, j)[1] - src.at<cv::Vec3b>(i, j)[2]) * 3 >> 2);// blue
			dst.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(std::abs(src.at<cv::Vec3b>(i, j)[1] - src.at<cv::Vec3b>(i, j)[0] - src.at<cv::Vec3b>(i, j)[2]) * 3 >> 2);// green
			dst.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(std::abs(src.at<cv::Vec3b>(i, j)[2] - src.at<cv::Vec3b>(i, j)[0] - src.at<cv::Vec3b>(i, j)[1]) * 3 >> 2);// red
		}
	}
}

int calmp(int value, int minValue = 0, int maxValue = 255) 
{
	if (value < minValue)

		return minValue;

	else if (value > maxValue)

		return maxValue;

	return value;
}

double  __stdcall AnaglyphFilter(Mat src, Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	src.copyTo(dst);

	int rowNumber = dst.rows;

	int colNumber = dst.cols;

	for (int i = 1; i < rowNumber - 1; ++i) 
	{

		for (int j = 1; j < colNumber - 1; ++j) 
		{

			dst.at<Vec3b>(i, j)[0] = calmp(src.at<Vec3b>(i + 1, j + 1)[0] - src.at<Vec3b>(i - 1, j - 1)[0] + 128);

			dst.at<Vec3b>(i, j)[1] = calmp(src.at<Vec3b>(i + 1, j + 1)[1] - src.at<Vec3b>(i - 1, j - 1)[1] + 128);

			dst.at<Vec3b>(i, j)[2] = calmp(src.at<Vec3b>(i + 1, j + 1)[2] - src.at<Vec3b>(i - 1, j - 1)[2] + 128);

		}

	}
}

double  __stdcall CastingFilter(Mat src, Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	src.copyTo(dst);

	for (size_t i = 0; i < src.rows; i++)
	{
		for (size_t j = 0; j < src.cols; j++)
		{
			dst.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(128 * src.at<cv::Vec3b>(i, j)[0] / (src.at<cv::Vec3b>(i, j)[1] + src.at<cv::Vec3b>(i, j)[2] + 1));// blue
			dst.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(128 * src.at<cv::Vec3b>(i, j)[1] / (src.at<cv::Vec3b>(i, j)[0] + src.at<cv::Vec3b>(i, j)[2] + 1));// green
			dst.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(128 * src.at<cv::Vec3b>(i, j)[2] / (src.at<cv::Vec3b>(i, j)[0] + src.at<cv::Vec3b>(i, j)[1] + 1));// red
		}
	}
}

double  __stdcall FreehandFilter(Mat src, Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	src.copyTo(dst);

	Mat Image_out(src.size(), CV_32FC3);
	src.convertTo(Image_out, CV_32FC3);
	Mat I(src.size(), CV_32FC1);
	cv::cvtColor(Image_out, I, COLOR_BGR2GRAY);
	I = I / 255.0;
	Mat I_invert;
	I_invert = -I + 1.0;
	Mat I_gau;
	GaussianBlur(I_invert, I_gau, Size(25, 25), 0, 0);
	float delta = 0.01;
	I_gau = -I_gau + 1.0 + delta;
	Mat I_dst;
	cv::divide(I, I_gau, I_dst);
	I_dst = I_dst;
	Mat b(src.size(), CV_32FC1);
	Mat g(src.size(), CV_32FC1);
	Mat r(src.size(), CV_32FC1);
	Mat rgb[] = { b,g,r };
	float alpha = 0.75;
	r = alpha * I_dst + (1 - alpha) * 200.0 / 255.0;
	g = alpha * I_dst + (1 - alpha) * 205.0 / 255.0;
	b = alpha * I_dst + (1 - alpha) * 105.0 / 255.0;
	cv::merge(rgb, 3, Image_out);

	Image_out = Image_out * 255;

	Image_out.convertTo(dst, CV_8UC3);

}

double  __stdcall SketchFilter(Mat src, Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	src.copyTo(dst);

	//1、去色
	cv::Mat gray(src.size(), CV_8UC3);
	for (size_t i = 0; i < src.rows; i++)
	{
		for (size_t j = 0; j < src.cols; j++)
		{
			int max = std::max(std::max(src.at<cv::Vec3b>(i, j)[0], src.at<cv::Vec3b>(i, j)[1]),src.at<cv::Vec3b>(i, j)[2]);
			int min = std::min(std::min(src.at<cv::Vec3b>(i, j)[0], src.at<cv::Vec3b>(i, j)[1]),src.at<cv::Vec3b>(i, j)[2]);

			for (size_t k = 0; k < 3; k++)
			{
				gray.at<cv::Vec3b>(i, j)[k] = (max + min) / 2;
			}
		}
	}

	//2、复制去色图层，并且反色
	cv::Mat gray_revesal(src.size(), CV_8UC3);
	for (size_t i = 0; i < gray.rows; i++)
	{
		for (size_t j = 0; j < gray.cols; j++)
		{
			for (size_t k = 0; k < 3; k++)
			{
				gray_revesal.at<cv::Vec3b>(i, j)[k] = 255 - gray.at<cv::Vec3b>(i, j)[k];
			}
		}
	}


	//3、对反色图像进行高斯模糊；
	cv::GaussianBlur(gray_revesal, gray_revesal, cv::Size(7, 7), 0);

	//4、模糊后的图像叠加模式选择颜色减淡效果。
	// 减淡公式：C =MIN( A +（A×B）/（255-B）,255)，其中C为混合结果，A为去色后的像素点，B为高斯模糊后的像素点。

	cv::Mat result(gray.size(), CV_8UC3);
	for (size_t i = 0; i < gray.rows; i++)
	{
		for (size_t j = 0; j < gray.cols; j++)
		{
			for (size_t k = 0; k < 3; k++)
			{
				int a = gray.at<cv::Vec3b>(i, j)[k];
				int b = gray_revesal.at<cv::Vec3b>(i, j)[k];
				int c = std::min(a + (a * b) / (255 - b), 255);
				result.at<cv::Vec3b>(i, j)[k] = c;
			}
		}
	}

	result.copyTo(dst);

}


void GetTexTransMatrix(float x1, float y1, float x2, float y2, float x3, float y3, float tx1, float ty1, float tx2, float ty2, float tx3, float ty3, float* texMatrix)
{
	float detA;
	detA = tx1 * ty2 + tx2 * ty3 + tx3 * ty1 - tx3 * ty2 - tx1 * ty3 - tx2 * ty1;
	float A11, A12, A13, A21, A22, A23, A31, A32, A33;
	A11 = ty2 - ty3;
	A21 = -(ty1 - ty3);
	A31 = ty1 - ty2;
	A12 = -(tx2 - tx3);
	A22 = tx1 - tx3;
	A32 = -(tx1 - tx2);
	A13 = tx2 * ty3 - tx3 * ty2;
	A23 = -(tx1 * ty3 - tx3 * ty1);
	A33 = tx1 * ty2 - tx2 * ty1;
	texMatrix[0] = (x1 * A11 + x2 * A21 + x3 * A31) / detA;
	texMatrix[1] = (x1 * A12 + x2 * A22 + x3 * A32) / detA;
	texMatrix[2] = (x1 * A13 + x2 * A23 + x3 * A33) / detA;
	texMatrix[3] = (y1 * A11 + y2 * A21 + y3 * A31) / detA;
	texMatrix[4] = (y1 * A12 + y2 * A22 + y3 * A32) / detA;
	texMatrix[5] = (y1 * A13 + y2 * A23 + y3 * A33) / detA;
}

int Trent_Sticker(unsigned char* srcData, int width, int height, int stride, unsigned char* mask, int maskWidth, int maskHeight, int maskStride, int srcFacePoints[6], int maskFacePoints[6], int ratio)
{
	int ret = 0;
	float H[6] = { 0 };
	GetTexTransMatrix(maskFacePoints[0], maskFacePoints[1], maskFacePoints[2], maskFacePoints[3], maskFacePoints[4], maskFacePoints[5], srcFacePoints[0], srcFacePoints[1], srcFacePoints[2], srcFacePoints[3], srcFacePoints[4], srcFacePoints[5], H);
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			float x = (float)i;
			float y = (float)j;
			float tx = 0;
			float ty = 0;
			tx = (int)((H[0] * (x)+H[1] * (y)+H[2]) + 0.5);
			ty = (int)((H[3] * (x)+H[4] * (y)+H[5]) + 0.5);
			tx = CLIP3(tx, 0, maskWidth - 1);
			ty = CLIP3(ty, 0, maskHeight - 1);
			int mb = mask[(int)tx * 4 + (int)ty * maskStride];
			int mg = mask[(int)tx * 4 + (int)ty * maskStride + 1];
			int mr = mask[(int)tx * 4 + (int)ty * maskStride + 2];
			int alpha = mask[(int)tx * 4 + (int)ty * maskStride + 3];
			int b = srcData[i * 3 + j * stride];
			int g = srcData[i * 3 + j * stride + 1];
			int r = srcData[i * 3 + j * stride + 2];
			srcData[(int)i * 3 + (int)j * stride] = CLIP3((b * (255 - alpha) + mb * alpha) / 255, 0, 255);
			srcData[(int)i * 3 + (int)j * stride + 1] = CLIP3((g * (255 - alpha) + mg * alpha) / 255, 0, 255);
			srcData[(int)i * 3 + (int)j * stride + 2] = CLIP3((r * (255 - alpha) + mr * alpha) / 255, 0, 255);
		}
	}
	return ret;
};

double  __stdcall Mask(Mat src, Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	src.copyTo(dst);

	Mat maskImg = imread("mask_b1.png", -1);

	/*FILE* pFile = fopen("pixel.txt", "w");
	fprintf(pFile, "%d\n", maskImg.channels());
	for (int i = 290; i <300; i++)
	{
		for (int j = 235; j < 255; j++)
		{
			fprintf(pFile, "%d\n", maskImg.at<Vec4b>(i, j)[0]);
			fprintf(pFile, "%d\n", maskImg.at<Vec4b>(i, j)[1]);
			fprintf(pFile, "%d\n", maskImg.at<Vec4b>(i, j)[2]);
			fprintf(pFile, "%d\n", maskImg.at<Vec4b>(i, j)[3]);
		}
	}

	fclose(pFile);
	pFile = NULL;*/
	//return 0;


	int a1 = p[6 + 2 * 37];
	int a11 = p[6 + 2 * 37 + 1];
	int a2 = p[6 + 2 * 44];
	int a22 = p[6 + 2 * 44 + 1];
	int a3 = p[6 + 2 * 62];
	int a33 = p[6 + 2 * 62 + 1];

	int pSrcPoint[6] = {a1,a11,a2,a22,a3,a33};
	int MaskPoint[6] = {307, 364, 423, 364, 365, 490};

	Trent_Sticker(dst.data, dst.cols, dst.rows, dst.cols*3, maskImg.data, maskImg.cols, maskImg.rows, maskImg.cols*4, pSrcPoint, MaskPoint, 100);

	return 0;
}