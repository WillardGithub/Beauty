/*
 *
 * Copyright 2001-2011 Texas Instruments, Inc. - http://www.ti.com/
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <poll.h>
#include <string>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <time.h>

#include "BeautyAlgorithm.h"
//#include "facedetectcnn.h"
//#include "face_test.h"
using namespace std;
using namespace cv;


#define JNI_LOG             "JNI_LOG"
//#define JNIREG_CLASS      "com/cosmetology/interfaces"
#define JNIREG_CLASS        "com/grg/finger/FingerVeinAndroid"
//#define DETECT_BUFFER_SIZE  0x20000

//#define DEBUG 
//#define AUTHOR
#define   STARTTIME "20200101"
#define   ENDTIME   "20211230"



#define LOG_DEBUG
#ifdef  LOG_DEBUG
   #define  JNI_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "========= JNI Info =========   ", __VA_ARGS__)
 
   #define  JNI_LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, "========= JNI Error =========   ", __VA_ARGS__)
 
   #define  JNI_LOGD(...)  __android_log_print(ANDROID_LOG_INFO, "========= JNI Debug =========   ", __VA_ARGS__)
 
   #define  JNI_LOGW(...)  __android_log_print(ANDROID_LOG_WARN, "========= JNI Warn =========   ", __VA_ARGS__)
#else
   #define  JNI_LOGI(...)
   #define  JNI_LOGE(...)
   #define  JNI_LOGD(...)
   #define  JNI_LOGW(...)
#endif


int auth(const char* startTime,const char* endTime)
{
   time_t sysTime;
   sysTime = time(NULL);
   struct tm *nowTime = localtime(&sysTime);
 
   //timespec time;
   char currentTime[16] = { 0 };  //sysTime->tm_year
   sprintf(currentTime, "%04d%02d%02d", nowTime->tm_year + 1900, nowTime->tm_mon + 1, nowTime->tm_mday);
#ifdef DEBUG
   LOGD("current time=%s", currentTime); 
#endif
   if (strncmp(currentTime, startTime, 8) < 0 ||strncmp(currentTime, endTime, 8) > 0)
   {
      LOGD("error is exist");
      return -1;
  }
   return 0;
}


static jdouble Cosmetology_Api_SlimFace(JNIEnv* env, jobject obj,jlong src,jdouble ratio,jlong dst,jshortArray p)
{
    #ifdef DEBUG
     JNI_LOGD("1start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
    #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif
     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
	if(i < length)
	JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
    #ifdef DEBUG
     JNI_LOGD("\n");
    #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = SlimFace(recvSrc,ratio,retDst,nativeP);
     if(nativeP !=NULL)
     {
	     free(nativeP);
	     nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif

     return ret;
}

static jdouble Cosmetology_Api_SlimNose(JNIEnv* env, jobject obj,jlong src,jdouble ratio,jlong dst,jshortArray p)
{
     #ifdef DEBUG
    JNI_LOGD("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = SlimNose(recvSrc,ratio,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;
}


static jdouble Cosmetology_Api_ZoomEyes(JNIEnv* env, jobject obj,jlong src, jdouble ratio, jlong dst, jshortArray p)
{
    #ifdef DEBUG
      JNI_LOGD("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = ZoomEyes(recvSrc,ratio,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
      JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}

static jdouble Cosmetology_Api_GlobalBuffing(JNIEnv *env, jobject obj, jlong src, jdouble ratio, jlong dst)
{
    #ifdef DEBUG
    JNI_LOGD("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
    #endif

    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif
    
     double ret = 0;
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = GlobalBuffing(recvSrc,ratio,retDst);
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}
static jdouble Cosmetology_Api_GlobalWhitening(JNIEnv *env, jobject obj, jlong src, jdouble ratio, jlong dst,jshortArray p)
{
	     #ifdef DEBUG
    JNI_LOGD("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
     #endif
 
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif
     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;

#ifdef  DEBUG
             /*for(int m = 0 ;m< 10;m++)
        {
                for(int n = 0;n<10;n++)
                {
                        JNI_LOGD("SRC:%d,%d,%d",recvSrc.at<Vec3b>(m,n)[0],recvSrc.at<Vec3b>(m,n)[1],recvSrc.at<Vec3b>(m,n)[2]);
                }
        }*/

#endif
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = GlobalWhitening(recvSrc,ratio,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}
static jdouble Cosmetology_Api_LocalBuffing(JNIEnv *env, jclass clazz, jlong src, jdouble ratio, jlong dst,jshortArray p)
{
	     #ifdef DEBUG
    JNI_LOGD("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = LocalBuffing(recvSrc,ratio,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}

static jdouble Cosmetology_Api_Whitening(JNIEnv *env, jclass clazz, jlong src, jdouble ratio, jlong dst,jshortArray p)
{
	     #ifdef DEBUG
    JNI_LOGD("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = Whitening(recvSrc,ratio,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}
static jdouble Cosmetology_Api_AdjustLip(JNIEnv *env, jclass clazz, jlong src, jdouble ratio, jlong dst,jshortArray p)
{
	     #ifdef DEBUG
    JNI_LOGD("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = AdjustLip(recvSrc,ratio,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}
static jdouble Cosmetology_Api_AdjustForeHead(JNIEnv *env, jclass clazz, jlong src, jdouble ratio, jlong dst,jshortArray p)
{
	     #ifdef DEBUG
    JNI_LOGD("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = AdjustForeHead(recvSrc,ratio,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}
static jdouble Cosmetology_Api_ColourCorrect(JNIEnv *env, jclass clazz, jlong src, jdouble ratio, jlong dst,jshortArray p)
{
	     #ifdef DEBUG
    JNI_LOGD("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = ColourCorrect(recvSrc,ratio,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}
static jdouble Cosmetology_Api_Sharpen(JNIEnv *env, jclass clazz, jlong src, jdouble ratio, jlong dst,jshortArray p)
{
	     #ifdef DEBUG
    JNI_LOGD("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = Sharpen(recvSrc,ratio,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}
static jdouble Cosmetology_Api_EnhanceRed(JNIEnv *env, jclass clazz, jlong src, jdouble ratio, jlong dst,jshortArray p)
{
	     #ifdef DEBUG
    JNI_LOGD("start to %s, ratio=%d\n", __FUNCTION__,(int)ratio);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = EnhanceRed(recvSrc,ratio,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}
static jdouble Cosmetology_Api_FrozenFilter(JNIEnv *env, jclass clazz, jlong src, jlong dst,jshortArray p)
{
     #ifdef DEBUG
    JNI_LOGD("start to %s\n", __FUNCTION__);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = FrozenFilter(recvSrc,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}
static jdouble Cosmetology_Api_AnaglyphFilter(JNIEnv *env, jclass clazz, jlong src, jlong dst,jshortArray p)
{
	   #ifdef DEBUG
    JNI_LOGD("start to %s\n", __FUNCTION__);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = AnaglyphFilter(recvSrc,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}
static jdouble Cosmetology_Api_CastingFilter(JNIEnv *env, jclass clazz, jlong src, jlong dst,jshortArray p)
{
	   #ifdef DEBUG
    JNI_LOGD("start to %s\n", __FUNCTION__);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = CastingFilter(recvSrc,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}
static jdouble Cosmetology_Api_FreehandFilter(JNIEnv *env, jclass clazz, jlong src, jlong dst,jshortArray p)
{
	   #ifdef DEBUG
    JNI_LOGD("start to %s\n", __FUNCTION__);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = FreehandFilter(recvSrc,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}
static jdouble Cosmetology_Api_SketchFilter(JNIEnv *env, jclass clazz, jlong src, jlong dst,jshortArray p)
{
	   #ifdef DEBUG
    JNI_LOGD("start to %s\n", __FUNCTION__);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }
     ret = SketchFilter(recvSrc,retDst,nativeP);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;

}
static jdouble Cosmetology_Api_Mask(JNIEnv *env, jclass clazz, jlong src, jlong dst,jshortArray p,jstring sConfig)
{
	   #ifdef DEBUG
    JNI_LOGD("start to %s\n", __FUNCTION__);
     #endif
    #ifdef AUTHOR
     int authorTime =  auth(STARTTIME,ENDTIME);
     if(authorTime < 0)
     {
        JNI_LOGD("start to %s au failure\n", __FUNCTION__);
	return -1;
     }
    #endif

     double ret = 0;
     jshort *tmpP = env->GetShortArrayElements(p,NULL);
     int length = env->GetArrayLength(p);
     short *nativeP = (short*)malloc(length);

     for(int i = 0; i < length; i++)
     {
        *(nativeP+i) = *(tmpP+i);
     #ifdef DEBUG
        if(i < length)
        JNI_LOGD("nativep[%d]=%d", i,*(nativeP+i));
     #endif
     }
     #ifdef  DEBUG
     JNI_LOGD("\n");
     #endif
     Mat& recvSrc =*(Mat*)src;
     Mat& retDst = *(Mat*)dst;
     if(recvSrc.empty())
       {
          JNI_LOGE("the src mat is NULL\n");
          return -1;
       }

     const char *c_sConfig = env->GetStringUTFChars(sConfig,NULL);
     if(c_sConfig == NULL)
     {
          JNI_LOGE("the c_sConfig is NULL\n");
          return -1;
     }
     ret = Mask(recvSrc,retDst,nativeP, c_sConfig);
     if(nativeP !=NULL)
     {
             free(nativeP);
             nativeP = NULL;
     }
     env->ReleaseShortArrayElements(p, tmpP, 0);//释放资源
     env->ReleaseStringUTFChars(sConfig, c_sConfig);
     #ifdef DEBUG
     JNI_LOGD("leave %s\n", __FUNCTION__);
     #endif
     return ret;
}
//all methods in jni
static JNINativeMethod gMethods[] ={

       //algorithm functions
	{"native_Cosmetology_Api_SlimFace", "(JDJ[S)D", (void*)Cosmetology_Api_SlimFace},
 	{"native_Cosmetology_Api_SlimNose", "(JDJ[S)D", (void*)Cosmetology_Api_SlimNose},
	{"native_Cosmetology_Api_ZoomEyes", "(JDJ[S)D", (void*)Cosmetology_Api_ZoomEyes},
	
        {"native_Cosmetology_Api_GlobalBuffing", "(JDJ)D", (void*)Cosmetology_Api_GlobalBuffing},
	
        {"native_Cosmetology_Api_GlobalWhitening", "(JDJ[S)D", (void*)Cosmetology_Api_GlobalWhitening},
	{"native_Cosmetology_Api_LocalBuffing", "(JDJ[S)D", (void*)Cosmetology_Api_LocalBuffing},
	{"native_Cosmetology_Api_Whitening", "(JDJ[S)D", (void*)Cosmetology_Api_Whitening},

	{"native_Cosmetology_Api_AdjustLip", "(JDJ[S)D", (void*)Cosmetology_Api_AdjustLip},
	{"native_Cosmetology_Api_AdjustForeHead", "(JDJ[S)D", (void*)Cosmetology_Api_AdjustForeHead},
	{"native_Cosmetology_Api_ColourCorrect", "(JDJ[S)D", (void*)Cosmetology_Api_ColourCorrect},
	{"native_Cosmetology_Api_Sharpen", "(JDJ[S)D", (void*)Cosmetology_Api_Sharpen},
	{"native_Cosmetology_Api_EnhanceRed", "(JDJ[S)D", (void*)Cosmetology_Api_EnhanceRed},
	
        {"native_Cosmetology_Api_FrozenFilter", "(JJ[S)D", (void*)Cosmetology_Api_FrozenFilter},
	{"native_Cosmetology_Api_AnaglyphFilter", "(JJ[S)D", (void*)Cosmetology_Api_AnaglyphFilter},
	{"native_Cosmetology_Api_CastingFilter", "(JJ[S)D", (void*)Cosmetology_Api_CastingFilter},
	{"native_Cosmetology_Api_FreehandFilter", "(JJ[S)D", (void*)Cosmetology_Api_FreehandFilter},
	{"native_Cosmetology_Api_SketchFilter", "(JJ[S)D", (void*)Cosmetology_Api_SketchFilter},
	{"native_Cosmetology_Api_Mask", "(JJ[SLjava/lang/String;)D", (void*)Cosmetology_Api_Mask},
};


/*
 * Register several native methods for one class.
 */
static int registerNatives(JNIEnv* env, const char* className,
               JNINativeMethod* gMethods, int numMethods)
{
    jclass clazz;
    clazz = env->FindClass(className);
    if (clazz == NULL) {
         LOGD("Can not find class %s\n", className);
        return JNI_FALSE;
    }

    if (env->RegisterNatives(clazz, gMethods, numMethods) < 0) {
        LOGD("Can not RegisterNatives\n");
        return JNI_FALSE;
    }

    return JNI_TRUE;
}

/*Commented the onload functionality end*/

jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    JNIEnv* env = NULL;
    jint result = -1;

    if (vm->GetEnv((void**)&env, JNI_VERSION_1_4) != JNI_OK) {
        goto bail;
    }

    if (!registerNatives(env,JNIREG_CLASS,gMethods,sizeof(gMethods)/sizeof(gMethods[0]))) {
        goto bail;
    }
    /* success -- return valid version number */
    result = JNI_VERSION_1_4;

bail:
    return result;
}


