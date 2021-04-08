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
#include <dlfcn.h>
#include <opencv2/opencv.hpp>

#include <FaceLandmarker.h>
#include <seeta/FaceDetector.h>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sys/time.h>
#include <unistd.h>
//#include "BeautyAlgorithm.h"




using namespace std;
using namespace cv;



#define HYG_TEST_FACEDETECT_CNN_SIZE  500
short pFace[HYG_TEST_FACEDETECT_CNN_SIZE] = {0};




#define JNIREG_CLASS      "com/grg/finger/FingerVeinAndroid"


#define JNI_LOG           "JNI_LOG"
#define  LOGI(...) __android_log_print(ANDROID_LOG_INFO, "========= Info =========   ", __VA_ARGS__)
 
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, "========= Error =========   ", __VA_ARGS__)
 
#define  LOGD(...)  __android_log_print(ANDROID_LOG_INFO, "========= Debug =========   ", __VA_ARGS__)
 
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN, "========= Warn =========   ", __VA_ARGS__)

JavaVM *g_jVM = NULL;
jclass _sJClass;
jobject g_obj = NULL;



seeta::FaceDetector *new_fd(char*path) 
{
    seeta::ModelSetting FD_model;
	FD_model.append(path);
    LOGD("load fd_model success!\n"); 
    return new seeta::FaceDetector(FD_model);
}
			
seeta::FaceLandmarker *new_FL(char *path) 
{
    seeta::ModelSetting PD_model;
	PD_model.append(path);
    LOGD("load pd_model success!\n"); 
    return new seeta::FaceLandmarker(PD_model);
}


static seeta::FaceDetector *fd = NULL ; //人脸检测的初始化
static seeta::FaceLandmarker *FL = NULL; //关键点检测模型初始化
//Mat recvSrc;


char  pPath[100]  = {0};
jshort tmpShortArray[HYG_TEST_FACEDETECT_CNN_SIZE]={0};
static jshortArray Face_Detect_Point(JNIEnv* env, jobject obj,jint stride,jint width,jint height,jint tmpflip,jint tmprotateAngle,jbyteArray srcYUV,jlong dst)
{
       #if 1
       struct timeval tpstart,tpend;
       float timeuse;
       gettimeofday(&tpstart,NULL);
        LOGD("kkkk hyg for test 1 stride=%d, widthh=%d,height=%d,flip=%d,rotateAngle=%d",stride,width,height,tmpflip,tmprotateAngle);
        gettimeofday(&tpend,NULL);
        timeuse=1000000*(tpend.tv_sec-tpstart.tv_sec)+tpend.tv_usec-tpstart.tv_usec;
        timeuse/=1000;
        LOGD("data entry time:%f\n",timeuse);

        //Mat& recvSrc =*(Mat*)src;
        jbyte *pYUVData = env->GetByteArrayElements(srcYUV,0);
        Mat g_BGRImage;
        g_BGRImage.create(height, width, CV_8UC3);
        Mat YUVSrc(height + height / 2, stride, CV_8UC1, (unsigned char*)pYUVData);
	Mat YUVImage = YUVSrc(Range(0, YUVSrc.rows), Range(0, width));
		
        cvtColor(YUVImage, g_BGRImage, COLOR_YUV2RGB_NV12);

        //Mat srcImg(height, width, CV_8UC3);
        //cvtColor(g_BGRImage, srcImg, COLOR_RGB2BGR);

        Mat ZoomSrc1;
	ZoomSrc1.create(height/4, width/4, CV_8UC3);
        LOGD("cols=%d, rows=%d,width=%d,height=%d", g_BGRImage.cols, g_BGRImage.rows,width,height);
        resize(g_BGRImage,ZoomSrc1,Size(g_BGRImage.cols/4,g_BGRImage.rows/4)); //缩放为原图的1/4
		
		
	//旋转及翻转
        Mat timg,ZoomSrc;
	    int rotateflag = 0;
		if(tmprotateAngle == -90)  //逆时针旋转90度
		{
			transpose(ZoomSrc1, timg);
			flip(timg, ZoomSrc, 0);
			rotateflag =1;
			LOGE("逆时针旋转90度\n");
		}
		
		if(tmprotateAngle == 90)  //顺时针旋转90度
		{
			transpose(ZoomSrc1, timg);
			flip(timg, ZoomSrc, 1);
			rotateflag =1;
			LOGE("顺时针旋转90度\n");
		}
		
		if(abs(tmprotateAngle) == 180)  //顺时针/逆时针旋转180度
		{
			flip(ZoomSrc1, timg,0);
			flip(timg, ZoomSrc, 1);
			rotateflag =1;
			LOGE("旋转180度\n");
		}
		
		if(tmpflip ==1) //左右翻转
		{
			if(rotateflag ==0)
			{
				ZoomSrc1.copyTo(ZoomSrc);
			}	
			flip(ZoomSrc, ZoomSrc, 1);
			LOGE("水平左右翻转\n");
		}
		
		if(tmpflip ==0 && tmprotateAngle ==0)  //不作任何处理直接拷贝
		{
			ZoomSrc1.copyTo(ZoomSrc);
			LOGE("直接拷贝\n");
		}
    
		    
        env->ReleaseByteArrayElements(srcYUV, pYUVData, 0);
        if(ZoomSrc.empty())
        {
           LOGE("the Zoomsrc is empty!\n");
           return NULL;
        }
		
        SeetaImageData image;
        image.height = ZoomSrc.rows;
        image.width = ZoomSrc.cols;
        image.channels = ZoomSrc.channels();
        image.data = ZoomSrc.data;
         
        LOGD("test height=%d,width=%d,channels=%d",image.height,image.width,image.channels);
        gettimeofday(&tpend,NULL);
        timeuse=1000000*(tpend.tv_sec-tpstart.tv_sec)+tpend.tv_usec-tpstart.tv_usec;
        timeuse/=1000;
        LOGD("transte time:%f\n",timeuse);
        auto faces = fd->detect((const SeetaImageData)image);


        gettimeofday(&tpend,NULL);
        timeuse=1000000*(tpend.tv_sec-tpstart.tv_sec)+tpend.tv_usec-tpstart.tv_usec;
        timeuse/=1000;
        LOGD("detect time:%f\n",timeuse);
        
        //return
        jshortArray retShortArray =env->NewShortArray(HYG_TEST_FACEDETECT_CNN_SIZE);
        if(retShortArray == NULL)
        {
           LOGE("%s:retShortArray new failure!\n",__FUNCTION__);
           return NULL;
        }
        LOGD("size=%d",faces.size);
		
	//返回的Mat(无论检测成功与否，都需要返回)
		Mat dstImg(ZoomSrc.rows,ZoomSrc.cols, CV_8UC3);
        cvtColor(ZoomSrc, dstImg, COLOR_RGB2BGR);
        *(Mat*)dst = dstImg;
		 
        if (faces.size == 1)
        {
                auto &face = faces.data[0].pos;
                std::vector<SeetaPointF> points(FL->number());
                FL->mark(image, face, points.data());	
				
				
				if(face.x  <0 )
				{
					face.x = abs(face.x);
				}
				
				if(face.x > image.width -1)
				{
					return NULL;
				}					
				if(face.y <0)
				{
					face.y = abs(face.y);
				}
				if(face.y>image.height -1)
				{
					return NULL;
				}
								
                pFace[0] = face.x;			
                pFace[1] = face.y;
                pFace[2] = face.width;
                pFace[3] = face.height;
				
                for(int i=0;i<68;i++)
                {

                  pFace[4+i*2] = abs((int)points[i].x);
                  pFace[4+i*2+1] = abs((int)points[i].y);
				  
				  if(pFace[4+i*2] > image.width -1)   //防止越界
				  {
					  pFace[4+i*2] = image.width -1;
				  }
				  
				  if(pFace[4+i*2+1] > image.height -1)  //防止越界
				  {
					  pFace[4+i*2+1] = image.height -1;
				  }
				  			  
                }

                memcpy(tmpShortArray,pFace,(68*2+4)*sizeof(short));

                env->SetShortArrayRegion(retShortArray,0,HYG_TEST_FACEDETECT_CNN_SIZE,tmpShortArray);
                
				gettimeofday(&tpend,NULL);
                timeuse=1000000*(tpend.tv_sec-tpstart.tv_sec)+tpend.tv_usec-tpstart.tv_usec;
                timeuse/=1000;
                LOGD("return p time:%f\n",timeuse);
		
        return retShortArray;
        
		}
       return NULL;
      #endif
}

static int FISA_Interfaces_Init(JNIEnv *env, jobject obj,jstring sConfigPath)
{
    LOGD("entry FISA_Interfaces_Init!\n");

    struct timeval tpstart,tpend;
    float timeuse;
    gettimeofday(&tpstart,NULL);

     const char *c_sConfig = env->GetStringUTFChars(sConfigPath,NULL);
     if(c_sConfig == NULL)
     {
          LOGE("the c_sConfig is NULL\n");
          return -1;
     }

     char tempPath1[512] = {0};
     char tempPath2[512] = {0};
     sprintf(tempPath1,"%s/face_detector.csta",c_sConfig);
     sprintf(tempPath2,"%s/face_landmarker_pts68.csta",c_sConfig);
     LOGD("model1=%s,model2=%s", tempPath1,tempPath2); 
     
     //for test
     memcpy(pPath,c_sConfig,strlen(c_sConfig));
     //use src
    fd = new_fd(tempPath1);; //人脸检测模型初始化
    FL = new_FL(tempPath2); //关键点检测模型初始化

    env->ReleaseStringUTFChars(sConfigPath, c_sConfig);
    gettimeofday(&tpend,NULL);
    timeuse=1000000*(tpend.tv_sec-tpstart.tv_sec)+tpend.tv_usec-tpstart.tv_usec;
    timeuse/=1000;
    LOGD("Used Time:%f\n",timeuse);
    LOGD("END FISA_Interfaces_Init!\n"); 
    return 0;
}





//all methods in jni
static JNINativeMethod gMethods[] ={
{"native_face_detect_init", "(Ljava/lang/String;)I",(void*)FISA_Interfaces_Init},
{"native_face_detect_point", "(IIIII[BJ)[S",(void*)Face_Detect_Point},

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


