#ifndef __DETECT_FACE__
#define __DETECT_FACE__
#endif // !__DETECT_2FACE__#pragma once

#include<stdlib.h>
#include<iostream>
#include<string.h> 
#include <fstream>  
#include <cstdlib>
#include <math.h>
#include <sstream>
#include <opencv2\opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>

using namespace std;
using namespace cv;

#pragma once

extern "C" _declspec(dllexport) double __stdcall SlimFace(Mat src, double ratio, Mat & dst, short* p);//瘦脸

extern "C" _declspec(dllexport) double __stdcall SlimNose(Mat src, double ratio, Mat & dst, short* p); //瘦鼻

//extern "C" _declspec(dllexport) double __stdcall AwlFace(Mat src, double ratio, Mat& dst, short* p); //锥子脸

extern "C" _declspec(dllexport) double __stdcall ZoomEyes(Mat src, double ratio, Mat & dst, short* p); //大眼

extern "C" _declspec(dllexport) double  __stdcall GlobalBuffing(Mat src, double ratio, Mat & dst);   //整体磨皮

extern "C" _declspec(dllexport) double  __stdcall GlobalWhitening(Mat src, double ratio, Mat & dst, short* p); //全局美白

extern "C" _declspec(dllexport) double __stdcall LocalBuffing(Mat src, double ratio, Mat & dst, short* p); //局部磨皮（人脸部位磨皮）

extern "C" _declspec(dllexport) double  __stdcall Whitening(Mat src, double ratio, Mat & dst, short* p); //局部美白

extern "C" _declspec(dllexport) double  __stdcall AdjustLip(Mat src, double ratio, Mat & dst, short* p);//调整嘴唇

extern "C" _declspec(dllexport) double  __stdcall AdjustForeHead(Mat src, double ratio, Mat & dst, short* p); //调整额头

extern "C" _declspec(dllexport) double  __stdcall ColourCorrect(Mat src, double ratio, Mat & dst, short* p); //校色

extern "C" _declspec(dllexport) double  __stdcall Sharpen(Mat src, double ratio, Mat & dst, short* p); //锐化

extern "C" _declspec(dllexport) double  __stdcall EnhanceRed(Mat src, double ratio, Mat & dst, short* p); //红润

extern "C" _declspec(dllexport) double  __stdcall FrozenFilter(Mat src, Mat & dst, short* p); //冰冻滤镜

extern "C" _declspec(dllexport) double  __stdcall AnaglyphFilter(Mat src, Mat & dst, short* p); //浮雕滤镜

extern "C" _declspec(dllexport) double  __stdcall CastingFilter(Mat src, Mat & dst, short* p); //熔铸滤镜

extern "C" _declspec(dllexport) double  __stdcall FreehandFilter(Mat src, Mat & dst, short* p); //手绘滤镜

extern "C" _declspec(dllexport) double  __stdcall SketchFilter(Mat src, Mat & dst, short* p); //素描滤镜

extern "C" _declspec(dllexport) double  __stdcall Mask(Mat src, Mat & dst, short* p); //面具