
// BeautyDemoDlgDlg.h: 头文件
//

#pragma once
#include"SelectFolderDlg.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;


// CBeautyDemoDlgDlg 对话框
class CBeautyDemoDlgDlg : public CDialogEx
{
// 构造
public:
	CBeautyDemoDlgDlg(CWnd* pParent = nullptr);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_BEAUTYDEMODLG_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

public:
	afx_msg void OnBnClickedButtonLoadPic();
	afx_msg void OnCustomdrawSlider1(NMHDR* pNMHDR, LRESULT* pResult);
	// 滑动条控件变量
	CSliderCtrl m_Slider;
	BOOL m_slider_flag; //标记滑动条是否被选中
	CString path;
	CSelectFolderDlg dlg; //文件夹选项

	Mat src_Mat;   //加载原始图像
	Mat dst_Mat;   //处理后的图像
	short* pFace;  //人脸关键点检测指针

	int FaceNumbers; //检测出来的人脸个数
	BOOL IsVideo; //是否处理视频流



	int m_Pos1; //获得滑块的当前位置
	int m_Pos2;
	int m_Pos3;
	int m_Pos4;
	int m_Pos5;
	int m_Pos6;
	int m_Pos7; 
	int m_Pos8;
	int m_Pos9;
	int m_Pos10;
	int m_Pos11;



	afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
	// 大眼滑动条
	CSliderCtrl m_slider2;
	BOOL m_slider2_flag; //标记滑动条是否被选中
	// 瘦鼻滑动条变量
	CSliderCtrl m_slider3;
	BOOL m_slider3_flag; //标记滑动条是否被选中
	// 锥子脸滑动条
	CSliderCtrl m_slider4;
	BOOL m_slider4_flag;
	afx_msg void OnBnClickedButtonOpenCam();
	static DWORD WINAPI ThreadCaptureImg(LPVOID lparam);


	BOOL Is_Pause;
	BOOL Is_Stop;
	afx_msg void OnBnClickedButtonGrabimg();
	afx_msg void OnBnClickedButtonBack();
	// 磨皮滑动条
	CSliderCtrl m_slider5;
	BOOL m_slider5_flag;
	// 美白滚动条
	CSliderCtrl m_slider6;
	BOOL m_slider6_flag;
	afx_msg void OnBnClickedButtonSave();

	// 嘴唇调整变量
	CSliderCtrl m_slider7;
	// 额头调整变量
	CSliderCtrl m_slider8;
	CSliderCtrl m_slider9;
	CSliderCtrl m_slider10;
	CSliderCtrl m_slider11;
	afx_msg void OnBnClickedRadio5();
	// 单选按钮关联变量
	int m_fliter;
	afx_msg void OnClickedRadio1();
	afx_msg void OnBnClickedButtonMask();
};
