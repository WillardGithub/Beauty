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

extern "C" _declspec(dllexport) double __stdcall SlimFace(Mat src, double ratio, Mat & dst, short* p);//����

extern "C" _declspec(dllexport) double __stdcall SlimNose(Mat src, double ratio, Mat & dst, short* p); //�ݱ�

//extern "C" _declspec(dllexport) double __stdcall AwlFace(Mat src, double ratio, Mat& dst, short* p); //׶����

extern "C" _declspec(dllexport) double __stdcall ZoomEyes(Mat src, double ratio, Mat & dst, short* p); //����

extern "C" _declspec(dllexport) double  __stdcall GlobalBuffing(Mat src, double ratio, Mat & dst);   //����ĥƤ

extern "C" _declspec(dllexport) double  __stdcall GlobalWhitening(Mat src, double ratio, Mat & dst, short* p); //ȫ������

extern "C" _declspec(dllexport) double __stdcall LocalBuffing(Mat src, double ratio, Mat & dst, short* p); //�ֲ�ĥƤ��������λĥƤ��

extern "C" _declspec(dllexport) double  __stdcall Whitening(Mat src, double ratio, Mat & dst, short* p); //�ֲ�����

extern "C" _declspec(dllexport) double  __stdcall AdjustLip(Mat src, double ratio, Mat & dst, short* p);//�����촽

extern "C" _declspec(dllexport) double  __stdcall AdjustForeHead(Mat src, double ratio, Mat & dst, short* p); //������ͷ

extern "C" _declspec(dllexport) double  __stdcall ColourCorrect(Mat src, double ratio, Mat & dst, short* p); //Уɫ

extern "C" _declspec(dllexport) double  __stdcall Sharpen(Mat src, double ratio, Mat & dst, short* p); //��

extern "C" _declspec(dllexport) double  __stdcall EnhanceRed(Mat src, double ratio, Mat & dst, short* p); //����

extern "C" _declspec(dllexport) double  __stdcall FrozenFilter(Mat src, Mat & dst, short* p); //�����˾�

extern "C" _declspec(dllexport) double  __stdcall AnaglyphFilter(Mat src, Mat & dst, short* p); //�����˾�

extern "C" _declspec(dllexport) double  __stdcall CastingFilter(Mat src, Mat & dst, short* p); //�����˾�

extern "C" _declspec(dllexport) double  __stdcall FreehandFilter(Mat src, Mat & dst, short* p); //�ֻ��˾�

extern "C" _declspec(dllexport) double  __stdcall SketchFilter(Mat src, Mat & dst, short* p); //�����˾�

extern "C" _declspec(dllexport) double  __stdcall Mask(Mat src, Mat & dst, short* p); //���