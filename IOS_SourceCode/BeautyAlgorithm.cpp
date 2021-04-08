
#include "BeautyAlgorithm.h" 
using namespace cv;
#define PI 3.1415926
#define DETECT_BUFFER_SIZE 0x20000

#define max_uchar(a, b)  (((a) > (b)) ? (a) : (b))
#define min_uchar(a, b)  (((a) < (b)) ? (a) : (b))

#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define CLIP3(x, a, b) MIN2(MAX2(a,x), b)

//#define DEBUG
//#define AUTHOR
#define   STARTTIME "20201209"
#define   ENDTIME   "20210230"

int auth(const char* startTime,const char* endTime)
{
   time_t sysTime;
   sysTime = time(NULL);
   struct tm *nowTime = localtime(&sysTime);
 
   //timespec time;
   char currentTime[16] = { 0 };  //sysTime->tm_year
   sprintf(currentTime, "%04d%02d%02d", nowTime->tm_year + 1900, nowTime->tm_mon + 1, nowTime->tm_mday);
  printf("current time=%s", currentTime);
   if (strncmp(currentTime, startTime, 8) < 0 ||strncmp(currentTime, endTime, 8) > 0)
   {
      printf("author out of time ");
      return -1;
  }
   return 0;
}

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
	Mat copyImg;
	copyMakeBorder(img, copyImg, 0, 0, 0, 0, BORDER_REPLICATE);
	double ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY);
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



Mat white(Mat& img, int startX, int startY, int x, int y ,int w, int h,float strength, float radius) //美白专用
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
	Mat copyImg;
	copyMakeBorder(img, copyImg,0,0,0,0,BORDER_REPLICATE);

	for (int i = x; i < x+w; i+=2)
	{
		for (int j = y; j < y+h; j+=2)
		{
			if (fabs(i - startX) >= radius && fabs(j - startY) >= radius)
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
				copyImg.ptr<Vec3b>(j)[i][0] = min(255, max(0, B));
				copyImg.ptr<Vec3b>(j)[i][1] = min(255, max(0, G));
				copyImg.ptr<Vec3b>(j)[i][2] = min(255, max(0, R));
				
				B = float(img.ptr<Vec3b>(j)[i+1][0]) + a;
				G = float(img.ptr<Vec3b>(j)[i+1][1]) + a;
				R = float(img.ptr<Vec3b>(j)[i+1][2]) + a;
				copyImg.ptr<Vec3b>(j)[i+1][0] = min(255, max(0, B));
				copyImg.ptr<Vec3b>(j)[i+1][1] = min(255, max(0, G));
				copyImg.ptr<Vec3b>(j)[i+1][2] = min(255, max(0, R));
				
				
				
				B = float(img.ptr<Vec3b>(j+1)[i][0]) + a;
				G = float(img.ptr<Vec3b>(j+1)[i][1]) + a;
				R = float(img.ptr<Vec3b>(j+1)[i][2]) + a;				
				copyImg.ptr<Vec3b>(j+1)[i][0] = min(255, max(0, B));
				copyImg.ptr<Vec3b>(j+1)[i][1] = min(255, max(0, G));
				copyImg.ptr<Vec3b>(j+1)[i][2] = min(255, max(0, R));
				
				
				
				B = float(img.ptr<Vec3b>(j+1)[i+1][0]) + a;
				G = float(img.ptr<Vec3b>(j+1)[i+1][1]) + a;
				R = float(img.ptr<Vec3b>(j+1)[i+1][2]) + a;
				copyImg.ptr<Vec3b>(j+1)[i+1][0] = min(255, max(0, B));
				copyImg.ptr<Vec3b>(j+1)[i+1][1] = min(255, max(0, G));
				copyImg.ptr<Vec3b>(j+1)[i+1][2] = min(255, max(0, R));
			
			}
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
double  SlimFace(cv::Mat src, double ratio, cv::Mat& dst,short *p)
{
#ifdef DEBUG
 printf("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
#endif
#ifdef AUTHOR
 int authorTime =  auth(STARTTIME,ENDTIME);
 if(authorTime < 0)
 {
    printf("start to %s author failure\n", __FUNCTION__);
return -1;
 }
#endif
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		printf("the mat src is null or p is null\n");
		return -1;
	}
	
	printf("SlimFace 2021-4-7===p0:%d,p1:%d,p2:%d,p3:%d,p4:%d,p5:%d,p6:%d,p7:%d\n",p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);

    
	double rate = ratio * 0.01;

	float r_left1 = sqrt((p[4 + 2 * 1] - p[4 + 2 * 30]) * (p[4 + 2 * 1] - p[4 + 2 * 30]) + (p[4 + 2 * 1 + 1] - p[4 + 2 * 30 + 1]) * (p[4 + 2 * 1 + 1] - p[4 + 2 * 30 + 1]));
	float r_right1 = sqrt((p[4 + 2 * 15] - p[4 + 2 * 30]) * (p[4 + 2 * 15] - p[4 + 2 * 30]) + (p[4 + 2 * 15 + 1] - p[4 + 2 * 30 + 1]) * (p[4 + 2 * 15 + 1] - p[4 + 2 * 30 + 1]));
	Mat data = BilinearInterpolation(src, p[4 + 2 * 1], p[4 + 2 *1 + 1], p[4 + 2 * 30], p[4 + 2 * 30 + 1], rate * r_left1 * 0.35);
	Mat data1 = BilinearInterpolation(data, p[4 + 2 * 15], p[4 + 2 * 15 + 1], p[4 + 2 * 30], p[4 + 2 * 30 + 1], rate * r_right1 * 0.35);

	float r_left2 = sqrt((p[4 + 2 * 4] - p[4 + 2 * 51]) * (p[4 + 2 * 4] - p[4 + 2 * 51]) + (p[4 + 2 * 4 + 1] - p[4 + 2 * 51 + 1]) * (p[4 + 2 * 4 + 1] - p[4 + 2 * 51 + 1]));
	float r_right2 = sqrt((p[4 + 2 * 12] - p[4 + 2 * 51]) * (p[4 + 2 * 12] - p[4 + 2 * 51]) + (p[4 + 2 * 12 + 1] - p[4 + 2 * 51 + 1]) * (p[4 + 2 * 12 + 1] - p[4 + 2 * 51+ 1]));
	Mat data2 = BilinearInterpolation(data1, p[4 + 2 * 4], p[4 + 2 * 4 + 1], p[4 + 2 * 51], p[4 + 2 * 51 + 1], rate * r_left1 * 0.35);
	Mat data3 = BilinearInterpolation(data2, p[4 + 2 * 12], p[4 + 2 * 12 + 1], p[4 + 2 * 51], p[4 + 2 * 51 + 1], rate * r_right1 * 0.35);

	//额头缩放
	//Mat data4 = BilinearInterpolation(data3, p[4 + 2 * 20], 2 * p[4 + 2 * 20 + 1] - p[4 + 2 * 29 + 1], p[4 + 2 * 20], p[4 + 2 * 20 + 1], p[4 + 2 * 29 + 1] - p[4 + 2 * 20 + 1]);
	//Mat data5 = BilinearInterpolation(data4, p[4 + 2 * 25], 2 * p[4 + 2 * 20 + 1] - p[4 + 2 * 29 + 1], p[4 + 2 * 20], p[4 + 2 * 20 + 1], p[4 + 2 * 29 + 1] - p[4 + 2 * 20 + 1]);
	//Mat data6 = BilinearInterpolation(data5, p[4 + 2 * 29], 2 * p[4 + 2 * 20 + 1] - p[4 + 2 * 29 + 1], p[4 + 2 * 20], p[4 + 2 * 20 + 1], p[4 + 2 * 29 + 1] - p[4 + 2 * 20 + 1]);

	//float radius_left_eye = sqrt((p[4 + 2 * 19] - p[4 + 2 * 37]) * (p[4 + 2 * 19] - p[4 + 2 * 37]) + (p[4 + 2 * 19 + 1] - p[4 + 2 * 37 + 1]) * (p[4 + 2 *19 + 1] - p[4 + 2 * 37 + 1]));
	//float radius_right_eye = sqrt((p[4 + 2 * 24] - p[4 + 2 * 44]) * (p[4 + 2 * 24] - p[4 + 2 * 44]) + (p[4 + 2 * 24 + 1] - p[4 + 2 * 44 + 1]) * (p[4 + 2 * 24 + 1] - p[4 + 2 * 44 + 1]));

	//Mat data4 = BilinearInterpolation(data3, p[4 + 2 * 19], (int)(p[4 + 2 * 19 + 1]*0.75), p[4 + 2 * 19], (int)(p[4 + 2 * 19 + 1] + radius_left_eye*1.1), rate * radius_left_eye*1.25);
	//Mat data5 = BilinearInterpolation(data4, p[4 + 2 * 24], (int)(p[4 + 2 * 24 + 1]*0.75), p[4 + 2 * 24], (int)(p[4 + 2 * 24 + 1]+  radius_right_eye*1.1), rate * radius_right_eye*1.25);

	//Mat data6 = BilinearInterpolation(data5, p[4 + 2 * 27], (int)((p[4 + 2 * 21 + 1] + p[4 + 2 * 22 + 1])*0.5*0.8), p[4 + 2 * 27], p[4 + 2 * 27 + 1] * 0.75, rate * (r_right1+r_right2) *0.5* 0.8);
	//Mat data6 = BilinearInterpolation(data5, p[4 + 2 * 29], 2 * p[4 + 2 * 20 + 1] - p[4 + 2 * 29 + 1], p[4 + 2 * 20], p[4 + 2 * 20 + 1], rate * (r_right1+ r_left1) * 0.25);

	data3.copyTo(dst);
#ifdef DEBUG
    
	printf("p[0]=%d, p[1]=%d, p[2]=%d,p[3]=%d\n",p[0],p[1],p[2],p[3]);
    printf("leave %s\n",__FUNCTION__);
#endif

	return 0;
}

//大眼
double  ZoomEyes(cv::Mat src, double ratio, cv::Mat& dst, short* p)
{
#ifdef DEBUG
 printf("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
#endif
#ifdef AUTHOR
 int authorTime =  auth(STARTTIME,ENDTIME);
 if(authorTime < 0)
 {
    printf("start to %s author failure\n", __FUNCTION__);
return -1;
 }
#endif
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	printf("ZoomEyes 2021-4-7===p0:%d,p1:%d,p2:%d,p3:%d,p4:%d,p5:%d,p6:%d,p7:%d\n",p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
	
	double rate = ratio * 0.01;

	int left_landmark = 37;  //左眼左边界点
	int left_landmark_down = 40;  //左眼右边界点
	int right_landmark = 43;  //右眼左边界点
	int right_landmark_down = 46; //右眼右边界点

	float r_left = sqrt((p[4 + 2 * left_landmark] - p[4 + 2 * left_landmark_down]) * (p[4 + 2 * left_landmark] - p[4 + 2 * left_landmark_down]) + (p[4 + 2 * left_landmark + 1] - p[4 + 2 * left_landmark_down + 1]) * (p[4 + 2 * left_landmark + 1] - p[4 + 2 * left_landmark_down + 1]));
	float r_right = sqrt((p[4 + 2 * right_landmark] - p[4 + 2 * right_landmark_down]) * (p[4 + 2 * right_landmark] - p[4 + 2 * right_landmark_down]) + (p[4 + 2 * right_landmark + 1] - p[4 + 2 * right_landmark_down + 1]) * (p[4 + 2 * right_landmark + 1] - p[4 + 2 * right_landmark_down + 1]));
	
	Mat data = BilinearInterpolation1(src, (p[4 + 2 * left_landmark] + p[4 + 2 * left_landmark_down])/2, (p[4 + 2 * left_landmark+1] + p[4 + 2 * left_landmark_down+1]) / 2, 0.8*rate,  r_left );
	Mat data1 = BilinearInterpolation1(data, (p[4 + 2 * right_landmark] + p[4 + 2 * right_landmark_down]) / 2, (p[4 + 2 * right_landmark + 1] + p[4 + 2 * right_landmark_down + 1]) / 2, 0.8*rate, r_right);
	data1.copyTo(dst);
#ifdef DEBUG
    printf("leave %s\n",__FUNCTION__);
#endif
	return 0;
}

//瘦鼻
double  SlimNose(cv::Mat src, double ratio, cv::Mat& dst,short * p)
{
#ifdef DEBUG
 printf("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
#endif
#ifdef AUTHOR
 int authorTime =  auth(STARTTIME,ENDTIME);
 if(authorTime < 0)
 {
    printf("start to %s author failure\n", __FUNCTION__);
    return -1;
 }
    
 printf("SlimNose 2021-4-7===p0:%d,p1:%d,p2:%d,p3:%d,p4:%d,p5:%d,p6:%d,p7:%d\n",p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
#endif
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

	float r_left = sqrt((p[4 + 2 * left_landmark] - p[4 + 2 * left_landmark_down]) * (p[4 + 2 * left_landmark] - p[4 + 2 * left_landmark_down]) + (p[4 + 2 * left_landmark + 1] - p[4 + 2 * left_landmark_down + 1]) * (p[4 + 2 * left_landmark + 1] - p[4 + 2 * left_landmark_down + 1]));
	float r_right = sqrt((p[4 + 2 * right_landmark] - p[4 + 2 * right_landmark_down]) * (p[4 + 2 * right_landmark] - p[4 + 2 * right_landmark_down]) + (p[4 + 2 * right_landmark + 1] - p[4 + 2 * right_landmark_down + 1]) * (p[4 + 2 * right_landmark + 1] - p[4 + 2 * right_landmark_down + 1]));
	Mat data = BilinearInterpolation(src, p[4 + 2 * left_landmark], p[4 + 2 * left_landmark + 1], p[4 + 2 * endPt], p[4 + 2 * endPt + 1], rate * r_left*0.75);
	Mat data1 = BilinearInterpolation(data, p[4 + 2 * right_landmark], p[4 + 2 * right_landmark + 1], p[4 + 2 * endPt], p[4 + 2 * endPt + 1], rate * r_right*0.75);
	data1.copyTo(dst);
#ifdef DEBUG
    printf("leave %s\n",__FUNCTION__);
#endif
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
//	//float r_left1 = sqrt((p[4 + 2 * 3] - p[4 + 2 * 5]) * (p[4 + 2 * 3] - p[4 + 2 * 5]) + (p[4 + 2 * 3 + 1] - p[4 + 2 * 5 + 1]) * (p[4 + 2 * 3 + 1] - p[4 + 2 * 5 + 1]));
//	//float r_right1 = sqrt((p[4 + 2 * 13] - p[4 + 2 * 15]) * (p[4 + 2 * 13] - p[4 + 2 * 15]) + (p[4 + 2 * 13 + 1] - p[4 + 2 * 15 + 1]) * (p[4 + 2 * 13 + 1] - p[4 + 2 * 15 + 1]));
//	//Mat data_slim1 = BilinearInterpolation(src, p[4 + 2 * 3], p[4 + 2 * 3 + 1], p[4 + 2 * 30], p[4 + 2 * 30 + 1],  rate*r_left1*0.9);
//	//Mat data1_slim2 = BilinearInterpolation(data_slim1, p[4 + 2 * 13], p[4 + 2 * 13 + 1], p[4 + 2 * 30], p[4 + 2 * 30 + 1],  rate*r_right1*0.9);
//
//	int left_landmark = 6;
//	int left_landmark_down = 8;
//	int right_landmark = 10;
//	int right_landmark_down = 8;
//
//	float r_left = sqrt((p[4 + 2 * left_landmark] - p[4 + 2 * left_landmark_down]) * (p[4 + 2 * left_landmark] - p[4 + 2 * left_landmark_down]) + (p[4 + 2 * left_landmark + 1] - p[4 + 2 * left_landmark_down + 1]) * (p[4 + 2 * left_landmark + 1] - p[4 + 2 * left_landmark_down + 1]));
//	float r_right = sqrt((p[4 + 2 * right_landmark] - p[4 + 2 * right_landmark_down]) * (p[4 + 2 * right_landmark] - p[4 + 2 * right_landmark_down]) + (p[4 + 2 * right_landmark + 1] - p[4 + 2 * right_landmark_down + 1]) * (p[4 + 2 * right_landmark + 1] - p[4 + 2 * right_landmark_down + 1]));
//
//	Mat data1 = BilinearInterpolation(dst, p[4 + 2 * right_landmark_down], p[4 + 2 * right_landmark_down + 1], p[4 + 2 * right_landmark_down], p[4 + 2 * right_landmark_down + 1] + 10, rate * (r_right+ r_left)*0.35);
//	Mat data2 = BilinearInterpolation(data1, p[4 + 2 * left_landmark], p[4 + 2 * left_landmark + 1], p[4 + 2 * right_landmark_down], p[4 + 2 * right_landmark_down + 1] + 10, rate * (r_right + r_left) * 0.35);
//	Mat data3 = BilinearInterpolation(data2, p[4 + 2 * right_landmark], p[4 + 2 * right_landmark + 1], p[4 + 2 * right_landmark_down], p[4 + 2 * right_landmark_down + 1] + 10, rate * (r_right + r_left)* 0.35);
//
//	Mat data4 = BilinearInterpolation(data3, p[4 + 2 * (left_landmark-1)], p[4 + 2 * (left_landmark-1) + 1], p[4 + 2 * right_landmark_down], p[4 + 2 * right_landmark_down + 1] + 10, rate * (r_right + r_left) * 0.35);
//	Mat data5 = BilinearInterpolation(data4, p[4 + 2 * (right_landmark+1)], p[4 + 2 * (right_landmark+1) + 1], p[4 + 2 * right_landmark_down], p[4 + 2 * right_landmark_down + 1] + 10, rate * (r_right + r_left) * 0.35);
//
//	Mat data6 = BilinearInterpolation(data5, p[4 + 2 * (left_landmark - 2)], p[4 + 2 * (left_landmark - 2) + 1], p[4 + 2 * right_landmark_down], p[4 + 2 * right_landmark_down + 1] + 10, rate * (r_right + r_left) * 0.35);
//	Mat data7 = BilinearInterpolation(data6, p[4 + 2 * (right_landmark + 2)], p[4 + 2 * (right_landmark + 2) + 1], p[4 + 2 * right_landmark_down], p[4 + 2 * right_landmark_down + 1] + 10, rate * (r_right + r_left) * 0.35);
//
//	data7.copyTo(dst);
//
//	free(pBuffer);
//	pBuffer = NULL;
//
//	return 0;
//}

//added at 2020-5-10
double  GlobalBuffing(cv::Mat src, double ratio, cv::Mat& dst)
{
#ifdef DEBUG
 printf("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
#endif
#ifdef AUTHOR
 int authorTime =  auth(STARTTIME,ENDTIME);
 if(authorTime < 0)
 {
    printf("start to %s author failure\n", __FUNCTION__);
return -1;
 }
#endif
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

	//add by hyg
	//IplImage* img1 = &IplImage(img11);
	//IplImage hygTemp1 = (IplImage)img11;
    IplImage hygTemp1 = cvIplImage(img11);
    IplImage* img1 = &hygTemp1;
	////////定义矩形区域  

	Mat temp1;
	src.copyTo(temp1);
	//add by hyg
	IplImage hygTemp2 = cvIplImage(temp1);
	IplImage* img2 = &hygTemp2;
	//IplImage* img2 = &IplImage(temp1);

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
#ifdef DEBUG
    printf("leave %s\n",__FUNCTION__);
#endif
	return 0;
}

double  LocalBuffing(cv::Mat src, double ratio, cv::Mat& dst, short* p)
{
#ifdef DEBUG
 printf("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
#endif
#ifdef AUTHOR
 int authorTime =  auth(STARTTIME,ENDTIME);
 if(authorTime < 0)
 {
    printf("start to %s author failure\n", __FUNCTION__);
return -1;
 }
#endif
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
#ifdef DEBUG
    printf("leave %s\n", __FUNCTION__);
#endif
	return 0;
}

double  GlobalWhitening(cv::Mat src, double ratio, cv::Mat& dst, short* p)
{
#ifdef DEBUG
 printf("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
#endif
#ifdef AUTHOR
 int authorTime =  auth(STARTTIME,ENDTIME);
 if(authorTime < 0)
 {
    printf("start to %s author failure\n", __FUNCTION__);
return -1;
 }
#endif
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
#ifdef DEBUG
    printf("leave %s\n",__FUNCTION__);
#endif
	return 0;

}

double  Whitening(cv::Mat src, double ratio, cv::Mat& dst, short* p)
{
#ifdef DEBUG
 printf("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
#endif
#ifdef AUTHOR
 int authorTime =  auth(STARTTIME,ENDTIME);
 if(authorTime < 0)
 {
    printf("start to %s author failure\n", __FUNCTION__);
return -1;
 }
#endif
/*#if 0
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	double rate = 0.01 * ratio;

	Mat temp1;
	src.copyTo(temp1);

	int x = p[0];
	int y = p[1];
	int w = p[2];
	int h = p[3];

	Mat faceImage = src(Range(y, y + h), Range(x, x + w)); //取出人脸矩形框
	float radius = sqrt(faceImage.rows * faceImage.rows / 4 + faceImage.cols * faceImage.cols / 4) ;
	
	Mat data1 = BilinearInterpolation2(faceImage, faceImage.cols/2, faceImage.rows/2, rate * 15, radius);
	
	data1.copyTo(temp1(Rect(x, y, w, h)));

	memcpy(dst.data, temp1.data, sizeof(unsigned char) * temp1.rows * temp1.cols * 3);
#endif*/
 if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	
	printf("Whitening 2021-4-7===p0:%d,p1:%d,p2:%d,p3:%d,p4:%d,p5:%d,p6:%d,p7:%d\n",p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
	
	Mat temp1;
	src.copyTo(temp1);

	//add at 2021-3-23
	int x1 = p[4 + 2 * 1];
	int y1 = p[4 + 2 * 1 + 1] - 1.2 * (p[4 + 2 * 9 + 1] - p[4 + 2 * 1 + 1]);
	if(y1 <0)
	{
		y1 = 1;
	}
	int w1 = p[4 + 2 * 16] -x1;
	int h1 = p[4 + 2 * 9 + 1] - y1;
	
	printf("x1:%d,y1:%d,w1:%d,h1:%d\n",x1,y1,w1,h1);

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


	Mat faceMask(src.rows,src.cols, CV_8UC3,Scalar(0,0,0));
	vector<Point> rootPoints;
	for (size_t j = 0; j < 17; j++)
	{
		Point pt;
		pt.x = p[4 + 2 * j];
		pt.y = p[4 + 2 * j+1];
		rootPoints.push_back(pt);
	}

	Point edgeP;
	edgeP.x = p[4 + 2 * 16]- w1/15;
	edgeP.y = p[4 + 2 * 24+1]*0.85;
	rootPoints.push_back(edgeP);
		
	
	edgeP.x = p[4 + 2 * 0]+w1/15;
	edgeP.y = p[4 + 2 * 19 + 1] * 0.85;
	
	rootPoints.push_back(edgeP);

	vector<vector<Point>> rp;
	
	rp.push_back(rootPoints);
	fillPoly(faceMask,rp,Scalar(255,255,255));
	
	
	
	int linePos = 0;  //眉毛所在的行
	int iFlag = 0;
	for (int row = 0; row < faceMask.rows; row++)
	{
		for (int col = 0; col < faceMask.cols; col++)
		{
			if (faceMask.at<Vec3b>(row, col)[0] ==255 && faceMask.at<Vec3b>(row, col)[1] ==255 &&faceMask.at<Vec3b>(row, col)[2] == 255)
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
	
	/*double rate = 0.01 * ratio;
	for (int row = 0; row < faceMask.rows; row++)
	{
		for (int col = 0; col < faceMask.cols; col++)
		{
			if (faceMask.at<Vec3b>(row, col)[0] == 0 && faceMask.at<Vec3b>(row, col)[1] == 0 && faceMask.at<Vec3b>(row, col)[2] == 0)
			{
				continue;
			}

			temp1.at<Vec3b>(row, col)[0] += (255 - temp1.at<Vec3b>(row, col)[0]) * rate;
			temp1.at<Vec3b>(row, col)[1] += (255 - temp1.at<Vec3b>(row, col)[1]) * rate;
			temp1.at<Vec3b>(row, col)[2] += (255 - temp1.at<Vec3b>(row, col)[2]) * rate;

		}
	}*/
	//temp1.copyTo(dst);
	
	
	//add at 2021-04-28
	//寻找外接矩形

	int rectUp = 0;
	int rectDown = 0;
	int rectLeft = 0;
	int rectRight = 0;
	iFlag = 0;
	for (int i = 0; i < faceMask.rows; i+=2)   //找矩形的上边界
	{
		for (int j = 0; j < faceMask.cols; j+=2)
		{
			if (faceMask.at<Vec3b>(i, j)[0] == 255 && faceMask.at<Vec3b>(i, j)[1] == 255 && faceMask.at<Vec3b>(i, j)[2] == 255)
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
	for (int i = faceMask.rows -1; i>0; i -= 2)   //找矩形的下边界
	{
		for (int j = 0; j < faceMask.cols; j += 2)
		{
			if (faceMask.at<Vec3b>(i, j)[0] == 255 && faceMask.at<Vec3b>(i, j)[1] == 255 && faceMask.at<Vec3b>(i, j)[2] == 255)
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
	for (int j = 0; j < faceMask.cols; j += 2)   //找矩形的左边界
	{
		for (int i = 0; i < faceMask.rows; i += 2)
		{
			if (faceMask.at<Vec3b>(i, j)[0] == 255 && faceMask.at<Vec3b>(i, j)[1] == 255 && faceMask.at<Vec3b>(i, j)[2] == 255)
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
	for (int j = faceMask.cols-1; j >0; j -= 2)   //找矩形的右边界
	{
		for (int i = 0; i < faceMask.rows; i += 2)
		{
			if (faceMask.at<Vec3b>(i, j)[0] == 255 && faceMask.at<Vec3b>(i, j)[1] == 255 && faceMask.at<Vec3b>(i, j)[2] == 255)
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


	double rate = 0.01 * ratio;
	for (int row = 0; row < faceMask.rows; row++)
	{
		for (int col = 0; col < faceMask.cols; col++)
		{
			if (faceMask.at<Vec3b>(row, col)[0] == 0 && faceMask.at<Vec3b>(row, col)[1] == 0 && faceMask.at<Vec3b>(row, col)[2] == 0)
			{
				continue;
			}
			temp1.at<Vec3b>(row, col)[0] += (255 - temp1.at<Vec3b>(row, col)[0]) * rate*0.125;
			temp1.at<Vec3b>(row, col)[1] += (255 - temp1.at<Vec3b>(row, col)[1]) * rate*0.125;
			temp1.at<Vec3b>(row, col)[2] += (255 - temp1.at<Vec3b>(row, col)[2]) * rate*0.125;
		}
	}

	//泊松融合
	Mat faceROI = temp1(Range(rect.y, rect.y+rect.height), Range(rect.x, rect.x + rect.width));
	Mat src_mask = 255 * Mat::ones(faceROI.rows, faceROI.cols, faceROI.depth());
	Point center(cX, cY);
	Mat mixed_clone;
	seamlessClone(faceROI, src, src_mask, center, mixed_clone, MIXED_CLONE);	
	mixed_clone.copyTo(dst);
	
	//mix image
	/*float tmp = 0;
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
	Mat dstZ(src.size(), CV_8UC3);//Z =??X * Op + (X + 2 * Y - 256)* Op= X  + (2*Y-256) *Op  OP不透明度 X原图 Y是高斯滤波后图像
	float OP = 0.035;//不透明度
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
	dstZ.copyTo(dst);*/

	printf("2021-04-08 local whiten finished!");
	return 0;
}

double   AdjustLip(cv::Mat src, double ratio, cv::Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	double rate = ratio * 0.015;

	Mat data = BilinearInterpolation(src, p[4 + 2 * 48], p[4 + 2 * 48 + 1], p[4 + 2 * 48] - 15, p[4 + 2 * 48 + 1], (p[4 + 2 * 59 + 1] - p[4 + 2 * 48 + 1]) * rate);
	Mat data1 = BilinearInterpolation(data, p[4 + 2 * 54], p[4 + 2 * 54 + 1], p[4 + 2 * 54] + 15, p[4 + 2 * 54 + 1], (p[4 + 2 * 55 + 1] - p[4 + 2 * 54 + 1]) * rate);

	data1.copyTo(dst);
	return 0;
}

double  AdjustForeHead(cv::Mat src, double ratio, cv::Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	double rate = ratio * 0.01;
	
	printf("AdjustForeHead 2021-4-7===p0:%d,p1:%d,p2:%d,p3:%d,p4:%d,p5:%d,p6:%d,p7:%d\n",p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
	
	//int a = rate*(p[4 + 2 * 37 + 1] - p[4 + 2 * 19 + 1])*0.8;
	//int b = (p[4 + 2 * 37 + 1] - p[4 + 2 * 19 + 1])*0.6;
	
	int a1 = rate * (p[4 + 2 * 37 + 1] - p[4 + 2 * 19 + 1]) * 0.6;
	int b1 = (p[4 + 2 * 37 + 1] - p[4 + 2 * 19 + 1])*0.65;

	int a2 = rate * (p[4 + 2 * 44 + 1] - p[4 + 2 * 24 + 1]) * 0.6;
	int b2 = (p[4 + 2 * 44 + 1] - p[4 + 2 * 24 + 1])*0.65;
	
	
	Mat data1 = BilinearInterpolation(src, p[4 + 2 * 18], p[4 + 2 * 19 + 1] - b1, p[4 + 2 * 18], p[4 + 2 * 19 + 1], a1);
	Mat data2 = BilinearInterpolation(data1, p[4 + 2 * 19], p[4 + 2 * 19 + 1] - b1, p[4 + 2 * 19], p[4 + 2 * 19 + 1], a1);
	Mat data3 = BilinearInterpolation(data2, p[4 + 2 * 20], p[4 + 2 * 19 + 1] - b1, p[4 + 2 * 20], p[4 + 2 * 19 + 1], a1);
	Mat datat = BilinearInterpolation(data3, p[4 + 2 * 21], p[4 + 2 * 19 + 1] - b1, p[4 + 2 * 21], p[4 + 2 * 19 + 1], a1);
	
	Mat data4 = BilinearInterpolation(datat, p[4 + 2 * 25], p[4 + 2 * 24 + 1] - b2, p[4 + 2 * 25], p[4 + 2 * 24 + 1],a2);
	Mat data5 = BilinearInterpolation(data4, p[4 + 2 * 24], p[4 + 2 * 24 + 1] - b2, p[4 + 2 * 24], p[4 + 2 * 24 + 1], a2);
	Mat data6 = BilinearInterpolation(data5, p[4 + 2 * 23], p[4 + 2 * 24 + 1] - b2, p[4 + 2 * 23], p[4 + 2 * 24 + 1], a2);
	Mat data7 = BilinearInterpolation(data6, p[4 + 2 * 22], p[4 + 2 * 24 + 1] - b2, p[4 + 2 * 22], p[4 + 2 * 24 + 1], a2);
   
	//Mat data4 = BilinearInterpolation(data3, p[4 + 2 * 21], p[4 + 2 * 19 + 1] - b, p[4 + 2 * 21], p[4 + 2 * 19 + 1],a);
	//Mat data5 = BilinearInterpolation(data4, p[4 + 2 * 22], p[4 + 2 * 19 + 1] - b, p[4 + 2 * 22], p[4 + 2 * 19 + 1], a);
	//Mat data6 = BilinearInterpolation(data5, p[4 + 2 * 28], p[4 + 2 * 20 + 1] - b, p[4 + 2 * 28], p[4 + 2 * 28 + 1], a);
	
	printf("adjust head finished only left side\n");
	//printf("a:%d,b:%d\n",a,b);
	printf("p[44]=%d,p[45]=%d,p[50]=%d,p[51]=%d,p[46]=%d,p[47]=%d,p[48]=%d,p[49]=%d,p[60]=%d,p[61]=%d,p[81]=%d\n",p[44],p[45],p[50],p[51],p[46],p[47],p[48],p[49],p[60],p[61],p[81]);
	
	
	data7.copyTo(dst);
	return 0;

}

double  ColourCorrect(cv::Mat src, double ratio, cv::Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	
	printf("ColourCorrect 2021-4-8===p0:%d,p1:%d,p2:%d,p3:%d,p4:%d,p5:%d,p6:%d,p7:%d\n",p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
	
	printf("Correct:rows:%d,cols:%d\n",src.rows,src.cols);
	
	Mat srcLab;
	cvtColor(src, srcLab, COLOR_BGR2Lab);  //颜色空间转换
	
	printf("src convertTo lab finished!\n");
	
	//added at 2021-03-21 	//定位人脸区域
	int x1 = p[4 + 2 * 1];
	int y1 = p[4 + 2 * 1 + 1] - 1.2 * (p[4 + 2 * 9 + 1] - p[4 + 2 * 1 + 1]);
	if(y1 <0)
	{
		y1 = 1;
	}
	int w1 = p[4 + 2 * 16] -x1;
	int h1 = p[4 + 2 * 9 + 1] - y1;
	
	printf("Correct x1:%d,y1:%d,w1:%d,h1:%d\n",x1,y1,w1,h1);
	Mat structKernel = getStructuringElement(MORPH_RECT, Size(3, 3));//卷积大小
	Mat imageROI = src(Range(y1, y1 + h1), Range(x1, x1 + w1));

	Mat ycrcb_image;
	cvtColor(imageROI, ycrcb_image, CV_BGR2YCrCb); //首先转换到YCrCb空间
	printf("bgr convertTo Ycrcb finished\n");
	
	Mat detect;
	vector<Mat> channels;
	split(ycrcb_image, channels);
	Mat output_mask = channels[1];
	threshold(output_mask, output_mask, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	imageROI.copyTo(detect, output_mask);
	Mat img0;
	erode(detect, img0, structKernel, Point(0, 0), 2);//开运算，消除噪点

	Mat faceMask(src.rows, src.cols, CV_8UC3, Scalar(0, 0, 0));
	printf("faceMask finished\n");
	
	vector<Point> rootPoints;
	for (size_t j = 0; j < 17; j++)
	{
		Point pt;
		pt.x = p[4 + 2 * j];
		pt.y = p[4 + 2 * j+1];
		rootPoints.push_back(pt);
	}

	Point edgeP;
	edgeP.x = p[4 + 2 * 16]- w1/15;
	edgeP.y = p[4 + 2 * 24+1]*0.85;
	rootPoints.push_back(edgeP);
		
	
	edgeP.x = p[4 + 2 * 0]+w1/15;
	edgeP.y = p[4 + 2 * 19 + 1] * 0.85;
	
	rootPoints.push_back(edgeP);

	vector<vector<Point>> rp;
	
	rp.push_back(rootPoints);
	fillPoly(faceMask,rp,Scalar(255,255,255));
	printf("fillPoly finished\n");
	
	
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

	double rate = 0.35+ 0.005 * ratio; 

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

			pImg[3 * j + 0] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 0]));
			pImg[3 * j + 1] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 1] + rate * 5));
			pImg[3 * j + 2] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 2]));

		}
		pImg += srcLab.step;
	}

	Mat temp1;
	cvtColor(srcLab, temp1, COLOR_Lab2BGR);

	
	int rectUp = 0;
	int rectDown = 0;
	int rectLeft = 0;
	int rectRight = 0;
	iFlag = 0;
	for (int i = 0; i < faceMask.rows; i+=2)   //找矩形的上边界
	{
		for (int j = 0; j < faceMask.cols; j+=2)
		{
			if (faceMask.at<Vec3b>(i, j)[0] == 255 && faceMask.at<Vec3b>(i, j)[1] == 255 && faceMask.at<Vec3b>(i, j)[2] == 255)
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
	for (int i = faceMask.rows -1; i>0; i -= 2)   //找矩形的下边界
	{
		for (int j = 0; j < faceMask.cols; j += 2)
		{
			if (faceMask.at<Vec3b>(i, j)[0] == 255 && faceMask.at<Vec3b>(i, j)[1] == 255 && faceMask.at<Vec3b>(i, j)[2] == 255)
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
	for (int j = 0; j < faceMask.cols; j += 2)   //找矩形的左边界
	{
		for (int i = 0; i < faceMask.rows; i += 2)
		{
			if (faceMask.at<Vec3b>(i, j)[0] == 255 && faceMask.at<Vec3b>(i, j)[1] == 255 && faceMask.at<Vec3b>(i, j)[2] == 255)
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
	for (int j = faceMask.cols-1; j >0; j -= 2)   //找矩形的右边界
	{
		for (int i = 0; i < faceMask.rows; i += 2)
		{
			if (faceMask.at<Vec3b>(i, j)[0] == 255 && faceMask.at<Vec3b>(i, j)[1] == 255 && faceMask.at<Vec3b>(i, j)[2] == 255)
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

	for (int row = 0; row < faceMask.rows; row++)
	{
		for (int col = 0; col < faceMask.cols; col++)
		{
			if (faceMask.at<Vec3b>(row, col)[0] == 0 && faceMask.at<Vec3b>(row, col)[1] == 0 && faceMask.at<Vec3b>(row, col)[2] == 0)
			{
				continue;
			}
			temp1.at<Vec3b>(row, col)[0] += (255 - temp1.at<Vec3b>(row, col)[0]) * rate*0.15;
			temp1.at<Vec3b>(row, col)[1] += (255 - temp1.at<Vec3b>(row, col)[1]) * rate*0.15;
			temp1.at<Vec3b>(row, col)[2] += (255 - temp1.at<Vec3b>(row, col)[2]) * rate*0.15;
		}
	}

	//泊松融合
	Mat faceROI = temp1(Range(rect.y, rect.y+rect.height), Range(rect.x, rect.x + rect.width));
	Mat src_mask = 255 * Mat::ones(faceROI.rows, faceROI.cols, faceROI.depth());
	Point center(cX, cY);
	Mat mixed_clone;
	seamlessClone(faceROI, src, src_mask, center, mixed_clone, MIXED_CLONE);	
	mixed_clone.copyTo(dst);
	
	printf("20210408 Poisson Fusion!\n");
	
	
    return 0;
}

double  Sharpen(cv::Mat src, double ratio, cv::Mat& dst, short* p)
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

double   EnhanceRed(cv::Mat src, double ratio, cv::Mat& dst, short* p)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	
	printf("EnhanceRed 2021-4-7=== p0:%d,p1:%d,p2:%d,p3:%d,p4:%d,p5:%d,p6:%d,p7:%d\n",p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
	
	double rate =  0.25+ratio * 0.005;
	
	cvtColor(src, src, COLOR_RGB2BGR);  //通道问题，取消注释即可解决

	// BGR空间转Lab空间
	Mat srcLab;
	cvtColor(src, srcLab, COLOR_BGR2Lab);

	//added at 2021-03-21 	//定位人脸区域
	int x1 = p[4 + 2 * 1];
	int y1 = p[4 + 2 * 1 + 1] - 1.2 * (p[4 + 2 * 9 + 1] - p[4 + 2 * 1 + 1]);
	if(y1 <0)
	{
		y1 = 1;
	}
	int w1 = p[4 + 2 * 16] -x1;
	int h1 = p[4 + 2 * 9 + 1] - y1;

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

	Mat faceMask(src.rows, src.cols, CV_8UC3, Scalar(0, 0, 0));

	vector<Point> rootPoints;
	for (size_t j = 0; j < 17; j++)
	{
		Point pt;
		pt.x = p[4 + 2 * j];
		pt.y = p[4 + 2 * j+1];
		rootPoints.push_back(pt);
	}

	Point edgeP;
	edgeP.x = p[4 + 2 * 16]- w1/15;
	edgeP.y = p[4 + 2 * 24+1]*0.85;
	rootPoints.push_back(edgeP);
		
	
	edgeP.x = p[4 + 2 * 0]+w1/15;
	edgeP.y = p[4 + 2 * 19 + 1] * 0.85;
	
	rootPoints.push_back(edgeP);

	vector<vector<Point>> rp;
	
	rp.push_back(rootPoints);
	fillPoly(faceMask,rp,Scalar(255,255,255));

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
		    pImg[3 *j + 0] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 0]));
			pImg[3 *j + 1] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 1]+ rate*5));
			pImg[3 *j + 2] = (uchar)min_uchar(255, max_uchar(0, pImg[3 * j + 2]));

		}
		pImg += srcLab.step;
	}
	cvtColor(srcLab, dst, COLOR_Lab2BGR);
	cvtColor(dst, dst, COLOR_BGR2RGB);
    return 0;
}

double   FrozenFilter(cv::Mat src, cv::Mat& dst, short* p) //冰冻滤镜
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
        return 0;
}

int calmp(int value, int minValue = 0, int maxValue = 255) 
{
	if (value < minValue)

		return minValue;

	else if (value > maxValue)

		return maxValue;

	return value;
}

double   AnaglyphFilter(cv::Mat src, cv::Mat& dst, short* p)
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
       return 0;
}

double   CastingFilter(cv::Mat src, cv::Mat& dst, short* p)
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
      return 0;
}

double   FreehandFilter(cv::Mat src, cv::Mat& dst, short* p)
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

        return 0;
}

double   SketchFilter(cv::Mat src, cv::Mat& dst, short* p)
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
        return 0;
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

double   Mask(cv::Mat src, cv::Mat& dst, short* p,char *maskPicture)
{
	if (src.empty() || !p) //判断输入图像及人脸关键点指针是否为空
	{
		return -1;
	}
	src.copyTo(dst);
    if(maskPicture == NULL)
	{
	    return -1;
	}
//	printf("Mask config =%s\n", sConfig);
	//Mat maskImg = imread("mask_b1.png", -1);
	Mat maskImg = imread(maskPicture, -1);

	int a1 = p[4 + 2 * 37];
	int a11 = p[4 + 2 * 37 + 1];
	int a2 = p[4 + 2 * 44];
	int a22 = p[4 + 2 * 44 + 1];
	int a3 = p[4 + 2 * 62];
	int a33 = p[4 + 2 * 62 + 1];

	int pSrcPoint[6] = {a1,a11,a2,a22,a3,a33};
	int MaskPoint[6] = {307, 364, 423, 364, 365, 490};
#if 0
	printf("mask.channels=%d\n", maskImg.channels());
        for(int i = 290; i < 300; i++)
	{
		for (int j = 235; j < 255; j++)
                {
                  printf("data[%d][%d]=%d\n",i,j, maskImg.at<Vec4b>(i, j)[0]);
                  printf("data[%d][%d]=%d\n",i,j, maskImg.at<Vec4b>(i, j)[1]);
                  printf("data[%d][%d]=%d\n",i,j, maskImg.at<Vec4b>(i, j)[2]);
                  printf("data[%d][%d]=%d\n",i,j, maskImg.at<Vec4b>(i, j)[3]);
                }
	}
#endif
	Trent_Sticker(dst.data, dst.cols, dst.rows, dst.cols*3, maskImg.data, maskImg.cols, maskImg.rows, maskImg.cols*4, pSrcPoint, MaskPoint, 100);

	return 0;
}
