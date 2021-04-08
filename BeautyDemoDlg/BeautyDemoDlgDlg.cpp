
// BeautyDemoDlgDlg.cpp: 实现文件
//

#include "pch.h"
#include "framework.h"
#include "BeautyDemoDlg.h"
#include "BeautyDemoDlgDlg.h"
#include "BeautyAlgorithm.h"
#include "afxdialogex.h"
#include <atlimage.h>
#include <string>

#include "facedetect-dll.h"
#include "facedetectcnn.h"


#define DETECT_BUFFER_SIZE 0x20000



using namespace std;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	
} 

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CBeautyDemoDlgDlg 对话框



CBeautyDemoDlgDlg::CBeautyDemoDlgDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_BEAUTYDEMODLG_DIALOG, pParent)
	, m_fliter(0)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CBeautyDemoDlgDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_SLIDER1, m_Slider);
	DDX_Control(pDX, IDC_SLIDER2, m_slider2);
	DDX_Control(pDX, IDC_SLIDER3, m_slider3);
	DDX_Control(pDX, IDC_SLIDER4, m_slider4);
	DDX_Control(pDX, IDC_SLIDER5, m_slider5);
	DDX_Control(pDX, IDC_SLIDER6, m_slider6);
	DDX_Control(pDX, IDC_SLIDER7, m_slider7);
	DDX_Control(pDX, IDC_SLIDER8, m_slider8);
	DDX_Control(pDX, IDC_SLIDER9, m_slider9);
	DDX_Control(pDX, IDC_SLIDER10, m_slider10);
	DDX_Control(pDX, IDC_SLIDER11, m_slider11);
	DDX_Radio(pDX, IDC_RADIO1, m_fliter);
}

BEGIN_MESSAGE_MAP(CBeautyDemoDlgDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_LOAD_PIC, &CBeautyDemoDlgDlg::OnBnClickedButtonLoadPic)
	//ON_NOTIFY(NM_CUSTOMDRAW,IDC_SLIDER1,&CBeautyDemoDlgDlg::OnCustomdrawSlider1)
	ON_WM_HSCROLL()
	ON_BN_CLICKED(IDC_BUTTON_OPEN_CAM, &CBeautyDemoDlgDlg::OnBnClickedButtonOpenCam)
	ON_BN_CLICKED(IDC_BUTTON_GRABIMG, &CBeautyDemoDlgDlg::OnBnClickedButtonGrabimg)
	ON_BN_CLICKED(IDC_BUTTON_BACK, &CBeautyDemoDlgDlg::OnBnClickedButtonBack)
	ON_BN_CLICKED(IDC_BUTTON_SAVE, &CBeautyDemoDlgDlg::OnBnClickedButtonSave)
	ON_BN_CLICKED(IDC_RADIO1, &CBeautyDemoDlgDlg::OnClickedRadio1)
	ON_BN_CLICKED(IDC_RADIO2, &CBeautyDemoDlgDlg::OnClickedRadio1)
	ON_BN_CLICKED(IDC_RADIO3, &CBeautyDemoDlgDlg::OnClickedRadio1)
	ON_BN_CLICKED(IDC_RADIO4, &CBeautyDemoDlgDlg::OnClickedRadio1)
	ON_BN_CLICKED(IDC_RADIO5, &CBeautyDemoDlgDlg::OnClickedRadio1)
	ON_BN_CLICKED(IDC_BUTTON_MASK, &CBeautyDemoDlgDlg::OnBnClickedButtonMask)
END_MESSAGE_MAP()


// CBeautyDemoDlgDlg 消息处理程序

BOOL CBeautyDemoDlgDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	pFace = NULL;
	FaceNumbers = 0;


	m_slider_flag = FALSE;
	m_slider2_flag = FALSE;
	m_slider3_flag = FALSE;
	m_slider4_flag = FALSE;
	m_slider5_flag = FALSE;
	m_slider6_flag = FALSE;

	m_Pos1 = 0;
	m_Pos2 = 0;
	m_Pos3 = 0;
	m_Pos4 = 0;
	m_Pos5 = 0;
	m_Pos6 = 0;
	m_Pos7 = 0;
	m_Pos8 = 0;
	m_Pos9 = 0;
	m_Pos10 = 0;
	m_Pos11 = 0;


	Is_Pause = FALSE;
	Is_Stop = FALSE;

	IsVideo = FALSE;

	//瘦脸滑动条
	m_Slider.SetRange(0, 100); //设置滑块位置的最大值和最小值
	m_Slider.SetTicFreq(1);
	for (int jj = 0; jj <= 100; jj += 1)
	{
		m_Slider.SetTic(jj);
	}

	m_Slider.SetPos(0);  //设置滑块的默认当前位置

	//大眼滑动条
	m_slider2.SetRange(0, 100); //设置滑块位置的最大值和最小值
	m_slider2.SetTicFreq(1);
	for (int jj = 0; jj <= 100; jj += 1)
	{
		m_slider2.SetTic(jj);
	}

	m_slider2.SetPos(0);

	m_slider3.SetRange(0, 100); //设置滑块位置的最大值和最小值
	m_slider3.SetTicFreq(1);
	for (int jj = 0; jj <= 100; jj += 1)
	{
		m_slider3.SetTic(jj);
	}
	m_slider3.SetPos(0);

	m_slider4.SetRange(0, 100); //设置滑块位置的最大值和最小值
	m_slider4.SetTicFreq(1);
	for (int jj = 0; jj <= 100; jj += 1)
	{
		m_slider4.SetTic(jj);	
	}
	m_slider4.SetPos(0);

	m_slider5.SetRange(0, 100); //设置滑块位置的最大值和最小值
	m_slider5.SetTicFreq(1);
	for (int jj = 0; jj <= 100; jj += 1)
	{
		m_slider5.SetTic(jj);

	}
	m_slider5.SetPos(0);

	m_slider6.SetRange(0, 100); //设置滑块位置的最大值和最小值
	m_slider6.SetTicFreq(1);
	for (int jj = 0; jj <= 100; jj += 1)
	{
		m_slider6.SetTic(jj);
	}
	m_slider6.SetPos(0);


	m_slider7.SetRange(0, 100); //设置滑块位置的最大值和最小值
	m_slider7.SetTicFreq(1);
	for (int jj = 0; jj <= 100; jj += 1)
	{
		m_slider7.SetTic(jj);
	}
	m_slider7.SetPos(0);

	m_slider8.SetRange(0, 100); //设置滑块位置的最大值和最小值
	m_slider8.SetTicFreq(1);
	for (int jj = 0; jj <= 100; jj += 1)
	{
		m_slider8.SetTic(jj);
	}
	m_slider8.SetPos(0);


	m_slider9.SetRange(0, 100); //设置滑块位置的最大值和最小值
	m_slider9.SetTicFreq(1);
	for (int jj = 0; jj <= 100; jj += 1)
	{
		m_slider9.SetTic(jj);
	}
	m_slider9.SetPos(0);


	m_slider10.SetRange(0, 100); //设置滑块位置的最大值和最小值
	m_slider10.SetTicFreq(1);
	for (int jj = 0; jj <= 100; jj += 1)
	{
		m_slider10.SetTic(jj);
	}
	m_slider10.SetPos(0);


	m_slider11.SetRange(0, 100); //设置滑块位置的最大值和最小值
	m_slider11.SetTicFreq(1);
	for (int jj = 0; jj <= 100; jj += 1)
	{
		m_slider11.SetTic(jj);
	}
	m_slider11.SetPos(0);

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CBeautyDemoDlgDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CBeautyDemoDlgDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CBeautyDemoDlgDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void MatToCImage(Mat& mat, CImage& cImage)
{ 
	int width = mat.cols;
	int height = mat.rows;
	int channels = mat.channels();

	cImage.Destroy(); //clear  
	cImage.Create(width, height, 8 * channels); //默认图像像素单通道占用1个字节  

	uchar* ps;
	uchar* pimg = (uchar*)cImage.GetBits(); //A pointer to the bitmap buffer  
	int step = cImage.GetPitch();

	for (int i = 0; i < height; ++i)
	{
		ps = (mat.ptr<uchar>(i));
		for (int j = 0; j < width; ++j)
		{
			if (channels == 1) //gray  
			{
				*(pimg + i * step + j) = ps[j];
			}
			else if (channels == 3) //color  
			{
				for (int k = 0; k < 3; ++k)
				{
					*(pimg + i * step + j * 3 + k) = ps[j * 3 + k];
				}
			}
		}
	}
}

void CBeautyDemoDlgDlg::OnBnClickedButtonLoadPic()
{
	GetDlgItem(IDC_BUTTON_GRABIMG)->EnableWindow(FALSE);
	GetDlgItem(IDC_BUTTON_BACK)->EnableWindow(FALSE);
	Is_Stop = TRUE;
	IsVideo = FALSE;
	Sleep(500);

	// TODO: 在此添加控件通知处理程序代码
	CFileDialog dlg(TRUE, _T("*.*"), NULL, OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY,
		_T(" All Files (*.*) |*.*||image files (*.jpg; *.bmp) |*.jpg; *.bmp |"), NULL
	);
	dlg.m_ofn.lpstrTitle = _T("加载单张图片");
	if (dlg.DoModal() != IDOK)
	{
		return;
	}

	path = dlg.GetPathName();
	char Data[500] = { 0 };
	WideCharToMultiByte(CP_ACP, 0, path, path.GetLength(), Data, sizeof(Data), NULL, NULL);
	string s(Data);
	src_Mat = imread(s, 1); //加载原图

	//Mat srcZoom(src_Mat.rows/4, src_Mat.cols/4, CV_8UC3);
	//resize(src_Mat, srcZoom, Size(srcZoom.cols, srcZoom.rows));
	//srcZoom.copyTo(src_Mat);

	unsigned char* pBuffer = (unsigned char*)malloc(DETECT_BUFFER_SIZE);
	Mat gray;
	cvtColor(src_Mat, gray, COLOR_BGR2GRAY);

	int* pFaceResults = facedetect_multiview_reinforce(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step, 1.2f, 2, 48, 0, 1);

     //pFace = facedetect_cnn(pBuffer, src_Mat.data, src_Mat.cols, src_Mat.rows, src_Mat.channels() * src_Mat.cols);

	FaceNumbers = *(pFaceResults);
    pFace = (short*)(pFaceResults + 1);



	//保存关键点信息

	FILE* pFile = fopen("keyPoint.txt", "w");
	fprintf(pFile, "%d,%d,%d,%d\n", pFace[0], pFace[1], pFace[2], pFace[3]);
	for (int j = 0; j < 68; j++) 
	{ 
	  //circle(src_Mat, Point((int)pFace[6 + 2 * j], (int)pFace[6 + 2 * j + 1]), 2, CV_RGB(255, 0, 0));
	  //cout << "pFace" << "[" <<6 + 2 * j<<"]" <<"="<< pFace[6 + 2 * j + 1] << endl;
	  fprintf(pFile, "p[%d] = %d\n", 4 + 2 * j, pFace[6 + 2 * j]);
	  fprintf(pFile, "p[%d] = %d\n", 4 + 2 * j+1, pFace[6 + 2 * j+1]);
	}
	fclose(pFile);
	pFile = NULL;

	//Rect rc;
	//rc.x = pFace[0];
	//rc.y = pFace[1];
	//rc.width = pFace[2];
	//rc.height = pFace[3];

	//rectangle(src_Mat, rc, CV_RGB(255, 0, 0, 0), 2);
	//imshow("src", src_Mat);
	//waitKey(0);

	//unsigned char* pBuffer = (unsigned char*)malloc(DETECT_BUFFER_SIZE);
	//Mat gray;
	//cvtColor(src_Mat, gray, COLOR_BGR2GRAY);
	//int* pRes = new int[10000];
	//pRes = facedetect_multiview_reinforce(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step, 1.2f, 2, 48, 0, 1);

	////int* pFaceRes = new int[10000];
	////memset(pFaceRes,0,sizeof(int)*10000);
	////detectFace(src_Mat, pFaceRes); //检测人脸关键点


	//FILE* fp3 = fopen("wwwwtste.txt", "a+");
	//fprintf(fp3, "FaceNumbers:%d\n", *(pFaceRes));
	//fclose(fp3);
	//fp3 = NULL;

	//pFace = (short*)pFaceRes;


	src_Mat.copyTo(dst_Mat); //拷贝原图至目标图

	Is_Pause = TRUE;
	Is_Stop = TRUE;

	int cxl, cyl;
	Mat rgbImageL1;//opencv图片
	CImage rgbImageL2;
	CRect rectl, pic_rectl;
	int widthl, heightl;
	//改变图片大小适应picture控件
	GetDlgItem(IDC_STATIC_IMGSHOW)->GetClientRect(&pic_rectl);
	widthl = pic_rectl.right;
	heightl = pic_rectl.bottom;
	resize(src_Mat, rgbImageL1, Size(widthl, heightl));
	MatToCImage(rgbImageL1, rgbImageL2);//OpenCV中Mat对象转MFC的CImage类的函数（见我另一篇博文）
	//获取图片的宽 高度  
	cxl = rgbImageL2.GetWidth();
	cyl = rgbImageL2.GetHeight();
	//获取Picture Control控件的大小  
	GetDlgItem(IDC_STATIC_IMGSHOW)->GetWindowRect(&rectl);
	//将客户区选中到控件表示的矩形区域内  
	ScreenToClient(&rectl);
	//窗口移动到控件表示的区域  
	GetDlgItem(IDC_STATIC_IMGSHOW)->MoveWindow(rectl.left, rectl.top, cxl, cyl, TRUE);
	CWnd* pWnd = NULL;
	pWnd = GetDlgItem(IDC_STATIC_IMGSHOW);//获取控件句柄  
	pWnd->GetClientRect(&rectl);//获取句柄指向控件区域的大小  
	CDC* pDc = NULL;
	pDc = pWnd->GetDC();//获取picture的DC  
	rgbImageL2.Draw(pDc->m_hDC, rectl);//将图片绘制到picture表示的区域内  
	ReleaseDC(pDc);
	pDc = NULL;
}


void CBeautyDemoDlgDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar) //美颜所有功能
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值
	//Is_Pause = TRUE;

	CDialogEx::OnHScroll(nSBCode, nPos, pScrollBar);
	UpdateData(TRUE);

	// TODO: 在此添加消息处理程序代码和/或调用默认值
	//CSliderCtrl* pSlidCtrl = (CSliderCtrl*)GetDlgItem(IDC_SLIDER1);
	////m_int 即为当前滑块的值。
	//int m_int = 0.1 * pSlidCtrl->GetPos();//取得当前位置值  

	m_Pos1 = m_Slider.GetPos(); //获得滑块的当前位置
	m_Pos2 = m_slider2.GetPos();
	m_Pos3 = m_slider3.GetPos();
	m_Pos4 = m_slider4.GetPos();
	m_Pos5 = m_slider5.GetPos();
	m_Pos6 = m_slider6.GetPos();
	m_Pos7 = m_slider7.GetPos();
	m_Pos8 = m_slider8.GetPos();
	m_Pos9 = m_slider9.GetPos();
	m_Pos10 = m_slider10.GetPos();
	m_Pos11 = m_slider11.GetPos();

	//FILE* fp2 = fopen("pos_src.txt", "a+");
	//fprintf(fp2, "pos1:%d\n", m_Pos1);
	//fprintf(fp2, "pos2:%d\n", m_Pos2);
	//fprintf(fp2, "pos3:%d\n", m_Pos3);
	//fprintf(fp2, "pos5:%d\n", m_Pos5);
	//fprintf(fp2, "pos5:%d\n", m_Pos6);
	//fclose(fp2);
	//fp2 = NULL;

	//CDialogEx::OnHScroll(nSBCode, nPos, pScrollBar);
	UpdateData(FALSE);


	if (IsVideo)
	{
		return;
	}
	

	//m_slider_flag = TRUE;
	//m_slider2_flag = FALSE;
	//m_slider3_flag = FALSE;
	//m_slider4_flag = FALSE;
	//m_slider5_flag = FALSE;
	//m_slider6_flag = FALSE;

	//if (FaceNumbers < 1)  //若没有人脸直接返回
	//{
	//	return;
	//}


	CSliderCtrl* pSlider = (CSliderCtrl*)pScrollBar;
	// 根据CSliderCtrl ID 来判断是哪一个CSliderCtrl
	//if (pSlider->GetDlgCtrlID() == IDC_SLIDER1)
	//{
	//	SlimFace(src_Mat, m_Pos1, dst_Mat, pFace);	
	//}


	//if (pSlider->GetDlgCtrlID() == IDC_SLIDER2)
	//{
	//	ZoomEyes(dst_Mat, m_Pos2, dst_Mat, pFace);
	//}

	//if (pSlider->GetDlgCtrlID() == IDC_SLIDER3)
	//{
	//	SlimNose(dst_Mat, m_Pos3, dst_Mat, pFace);
	//}

	//if (pSlider->GetDlgCtrlID() == IDC_SLIDER4)
	//{

	//}


	//if (pSlider->GetDlgCtrlID() == IDC_SLIDER5)
	//{
	//	LocalBuffing(dst_Mat, m_Pos5, dst_Mat, pFace);
	//}


	//if (pSlider->GetDlgCtrlID() == IDC_SLIDER6)
	//{
	//	Whitening(dst_Mat, m_Pos6, dst_Mat, pFace);  //全局美白
	//}

	//if (pSlider->GetDlgCtrlID() == IDC_SLIDER7)
	//{
	//	AdjustLip(dst_Mat, m_Pos7, dst_Mat, pFace);
	//}


	//if (pSlider->GetDlgCtrlID() == IDC_SLIDER8)
	//{
	//	AdjustForeHead(dst_Mat, m_Pos8, dst_Mat, pFace);
	//}

	//if (pSlider->GetDlgCtrlID() == IDC_SLIDER9)
	//{
	//	ColourCorrect(dst_Mat, m_Pos9, dst_Mat, pFace);
	//}

	//if (pSlider->GetDlgCtrlID() == IDC_SLIDER10)
	//{
	//	Sharpen(dst_Mat, m_Pos10, dst_Mat, pFace);
	//}

	//if (pSlider->GetDlgCtrlID() == IDC_SLIDER11)
	//{
	//	EnhanceRed(dst_Mat, m_Pos11, dst_Mat, pFace);
	//}
	src_Mat.copyTo(dst_Mat);

	if (m_Pos1 > 0)
	{
		SlimFace(src_Mat, m_Pos1, dst_Mat, pFace);
	}
	if (m_Pos2 > 0)
	{
		ZoomEyes(dst_Mat, m_Pos2, dst_Mat, pFace);
	}
	if (m_Pos3 > 0)
	{
		SlimNose(dst_Mat, m_Pos3, dst_Mat, pFace);
	}
	if (m_Pos5 > 0)
	{
		LocalBuffing(dst_Mat, m_Pos5, dst_Mat, pFace);
	}
	if (m_Pos6 > 0)
	{
		Whitening(dst_Mat, m_Pos6, dst_Mat, pFace);  //全局美白
		//GlobalWhitening(dst_Mat, m_Pos6, dst_Mat, pFace);
	}
	
	if (m_Pos7 > 0)
	{
		AdjustLip(dst_Mat, m_Pos7, dst_Mat, pFace);
	}
	
	if (m_Pos8 > 0)
	{
		AdjustForeHead(dst_Mat, m_Pos8, dst_Mat, pFace);
	}
	if (m_Pos9 > 0)
	{
		ColourCorrect(dst_Mat, m_Pos9, dst_Mat, pFace);
	}
	
	if (m_Pos10 > 0)
	{
		Sharpen(dst_Mat, m_Pos10, dst_Mat, pFace);
	}
	
	if (m_Pos11>0)
	{
		EnhanceRed(dst_Mat, m_Pos11, dst_Mat, pFace);
	}
	

	int cxl, cyl;
	Mat rgbImageL1;//opencv图片
	CImage rgbImageL2;
	CRect rectl, pic_rectl;
	int widthl, heightl;
	//改变图片大小适应picture控件
	GetDlgItem(IDC_STATIC_IMGSHOW)->GetClientRect(&pic_rectl);
	widthl = pic_rectl.right;
	heightl = pic_rectl.bottom;
	resize(dst_Mat, rgbImageL1, Size(widthl, heightl));
	MatToCImage(rgbImageL1, rgbImageL2);//OpenCV中Mat对象转MFC的CImage类的函数（见我另一篇博文）
	//获取图片的宽高度  
	cxl = rgbImageL2.GetWidth();
	cyl = rgbImageL2.GetHeight();
	//获取Picture Control控件的大小  
	GetDlgItem(IDC_STATIC_IMGSHOW)->GetWindowRect(&rectl);
	//将客户区选中到控件表示的矩形区域内  
	ScreenToClient(&rectl);
	//窗口移动到控件表示的区域  
	GetDlgItem(IDC_STATIC_IMGSHOW)->MoveWindow(rectl.left, rectl.top, cxl, cyl, TRUE);
	CWnd* pWnd = NULL;
	pWnd = GetDlgItem(IDC_STATIC_IMGSHOW);//获取控件句柄  
	pWnd->GetClientRect(&rectl);//获取句柄指向控件区域的大小  
	CDC* pDc = NULL;
	pDc = pWnd->GetDC();//获取picture的DC  
	rgbImageL2.Draw(pDc->m_hDC, rectl);//将图片绘制到picture表示的区域内  
	ReleaseDC(pDc);
	pDc = NULL;

	return;

	//Mat dst_Mat(src_Mat.rows, src_Mat.cols, src_Mat.type());
	//memcpy(dst_Mat.data, src_Mat.data, sizeof(unsigned char) * src_Mat.rows * src_Mat.cols);
	//src_Mat.copyTo(dst_Mat);
}

void CBeautyDemoDlgDlg::OnBnClickedButtonOpenCam()
{
	// TODO: 在此添加控件通知处理程序代码

	//GetDlgItem(IDC_BUTTON_LOAD_PIC)->EnableWindow(FALSE);

	IsVideo = TRUE;

	GetDlgItem(IDC_BUTTON_GRABIMG)->EnableWindow(TRUE);
	GetDlgItem(IDC_BUTTON_BACK)->EnableWindow(TRUE);
	Sleep(300);

	CreateThread(NULL, 0, ThreadCaptureImg, this, 0, NULL);
	Is_Stop = FALSE;
	Is_Pause = FALSE;
}

DWORD WINAPI CBeautyDemoDlgDlg::ThreadCaptureImg(LPVOID lparam)
{
	VideoCapture cap(0);
	//判断摄像头是否打开 
	if (!cap.isOpened())
	{
		return -1;
	}

	

	CBeautyDemoDlgDlg* p = (CBeautyDemoDlgDlg*)lparam;

	while (!p->Is_Stop)
	{
		if (p->Is_Pause)
		{
			Sleep(300);
			continue;
		}

		cap >> p->src_Mat;
		//imshow("src", p->src_Mat);
		//waitKey(5);
		//continue;

		p->src_Mat.copyTo(p->dst_Mat);
		unsigned char* pBuffer = (unsigned char*)malloc(DETECT_BUFFER_SIZE);
		Mat gray;
		cvtColor(p->src_Mat, gray, COLOR_BGR2GRAY);
		int* pFaceResults = facedetect_multiview_reinforce(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step, 1.2f, 2, 48, 0, 1);
		//int* pFaceResults = NULL;
		p->FaceNumbers = *(pFaceResults);  
		p->pFace = (short*)(pFaceResults +1);

	
		if (p->FaceNumbers >= 1)
		{
			p->m_Pos1 = p->m_Slider.GetPos(); //获得滑块的当前位置
			p->m_Pos2 = p->m_slider2.GetPos();
			p->m_Pos3 = p->m_slider3.GetPos();
			p->m_Pos5 = p->m_slider5.GetPos();
			p->m_Pos6 = p->m_slider6.GetPos();
			p->m_Pos7 = p->m_slider7.GetPos();
			p->m_Pos8 = p->m_slider8.GetPos();

			//p->pFace = (short*)(pFaceResults + 1);

			//SlimFace(p->src_Mat, p->m_Pos1, p->dst_Mat, p->pFace);
			//ZoomEyes(p->dst_Mat, p->m_Pos2, p->dst_Mat, p->pFace);
			//SlimNose(p->dst_Mat, p->m_Pos3, p->dst_Mat, p->pFace);
			//LocalBuffing(p->dst_Mat, p->m_Pos5, p->dst_Mat, p->pFace);
			GlobalWhitening(p->dst_Mat, p->m_Pos6, p->dst_Mat, p->pFace);  //全局美白
			//AdjustLip(p->dst_Mat, p->m_Pos7, p->dst_Mat, p->pFace);
			//AdjustForeHead(p->dst_Mat, p->m_Pos8, p->dst_Mat, p->pFace);

			//FILE* fp3 = fopen("pos202008021033.txt", "a+");
			//fprintf(fp3, "pos1:%d\n", p->m_Pos1);
			//fprintf(fp3, "pos2:%d\n", p->m_Pos2);
			//fprintf(fp3, "pos3:%d\n", p->m_Pos3);
			//fprintf(fp3, "pos5:%d\n", p->m_Pos5);
			//fprintf(fp3, "pos6:%d\n", p->m_Pos6);

			//fprintf(fp3, "x:%d\n", p->pFace[0]);
			//fprintf(fp3, "y:%d\n", p->pFace[1]);
			//fprintf(fp3, "w:%d\n", p->pFace[2]);
			//fprintf(fp3, "h:%d\n", p->pFace[3]);

			//fprintf(fp3, "===============\n");
			//fp3 = NULL;
		}


		//delete[] pFaceResults;

		int cxl = 0, cyl = 0;
		Mat rgbImageL1;//opencv图片
		CImage rgbImageL2;
		CRect rectl, pic_rectl;
		int widthl, heightl;
		//改变图片大小适应picture控件
		p->GetDlgItem(IDC_STATIC_IMGSHOW)->GetClientRect(&pic_rectl);
		widthl = pic_rectl.right;
		heightl = pic_rectl.bottom;
		resize(p->dst_Mat, rgbImageL1, Size(widthl, heightl));
		MatToCImage(rgbImageL1, rgbImageL2);//OpenCV中Mat对象转MFC的CImage类的函数
		//获取图片的宽 高度  
		cxl = rgbImageL2.GetWidth();
		cyl = rgbImageL2.GetHeight();
		//获取Picture Control控件的大小  
		p->GetDlgItem(IDC_STATIC_IMGSHOW)->GetWindowRect(&rectl);
		//将客户区选中到控件表示的矩形区域内  
		p->ScreenToClient(&rectl);
		//窗口移动到控件表示的区域  
		p->GetDlgItem(IDC_STATIC_IMGSHOW)->MoveWindow(rectl.left, rectl.top, cxl, cyl, TRUE);
		CWnd* pWnd = NULL;
		pWnd = p->GetDlgItem(IDC_STATIC_IMGSHOW);//获取控件句柄  
		pWnd->GetClientRect(&rectl);//获取句柄指向控件区域的大小  
		CDC* pDc = NULL;
		pDc = pWnd->GetDC();//获取picture的DC  
		rgbImageL2.Draw(pDc->m_hDC, rectl);//将图片绘制到picture表示的区域内  
		p->ReleaseDC(pDc);
		pDc = NULL;
	}

	cap.release();
	return 0;

}

void CBeautyDemoDlgDlg::OnBnClickedButtonGrabimg()
{
	// TODO: 在此添加控件通知处理程序代码
	Is_Pause = TRUE;
	IsVideo = FALSE;

}


void CBeautyDemoDlgDlg::OnBnClickedButtonBack()
{
	// TODO: 在此添加控件通知处理程序代码
	Is_Pause = FALSE;
	IsVideo = TRUE;
}


void CBeautyDemoDlgDlg::OnBnClickedButtonSave()
{
	if(dst_Mat.empty())
	{
		return;	
	}

	// TODO: 在此添加控件通知处理程序代码
	SYSTEMTIME sys;
	GetLocalTime(&sys);

	//printf("%4d/%02d/%02d%02d%02d%02d.%03d\n", sys.wYear, sys.wMonth, sys.wDay, sys.wHour, sys.wMinute, sys.wSecond, sys.wMilliseconds);

	int year = sys.wYear;
	int month = sys.wMonth;
	int day = sys.wDay;
	int hour = sys.wHour;
	int minute = sys.wMinute;
	int sec = sys.wSecond;

	//int mill_sec = sys.wMilliseconds;
	
	string savePath = "美颜_" + to_string(year) + to_string(month) + to_string(day) + to_string(hour) + to_string(minute) + to_string(sec) + ".jpg";
	imwrite(savePath, dst_Mat);
	

	//if (m_slider_flag == TRUE)
	//{
	//	string savePath = "瘦脸_"+ to_string(year) + to_string(month) + to_string(day) + to_string(hour) + to_string(minute) + to_string(sec)  + ".jpg";
	//	imwrite(savePath, dst_Mat);
	//	//MessageBox(_T("saved"));
	//}

	//else if (m_slider2_flag == TRUE)
	//{
	//	string savePath = "大眼_" + to_string(year) + to_string(month) + to_string(day) + to_string(hour) + to_string(minute) + to_string(sec)  + ".jpg";
	//	imwrite(savePath, dst_Mat);
	//}

	//else if (m_slider3_flag == TRUE)
	//{
	//	string savePath = "瘦鼻_" + to_string(year) + to_string(month) + to_string(day) + to_string(hour) + to_string(minute) + to_string(sec)  + ".jpg";
	//	imwrite(savePath, dst_Mat);
	//}

	//else if (m_slider4_flag == TRUE)
	//{
	//	string savePath = "锥子脸_" + to_string(year) + to_string(month) + to_string(day) + to_string(hour) + to_string(minute) + to_string(sec)  + ".jpg";
	//	imwrite(savePath, dst_Mat);
	//}
	//else if (m_slider5_flag == TRUE)
	//{
	//	string savePath = "磨皮_" + to_string(year) + to_string(month) + to_string(day) + to_string(hour) + to_string(minute) + to_string(sec)  + ".jpg";
	//	imwrite(savePath, dst_Mat);
	//}

	//else if (m_slider6_flag == TRUE)
	//{
	//	string savePath = "美白_" + to_string(year) + to_string(month) + to_string(day) + to_string(hour) + to_string(minute) + to_string(sec) + ".jpg";
	//	imwrite(savePath, dst_Mat);
	//}

	//else
	//{
	//	return;
	//}

}



void CBeautyDemoDlgDlg::OnClickedRadio1()
{
	// TODO: 在此添加控件通知处理程序代码
	UpdateData(TRUE);
	switch (m_fliter)
	{
	case 0:
		FrozenFilter(src_Mat, dst_Mat, pFace); //冰冻滤镜
		break;
	case 1:
		AnaglyphFilter(src_Mat, dst_Mat, pFace); //浮雕滤镜
		break;
	case 2:
		CastingFilter(src_Mat, dst_Mat, pFace); //熔铸滤镜
		break;
	case 3:
		FreehandFilter(src_Mat, dst_Mat, pFace); //手绘滤镜
		break;
	case 4:
		SketchFilter(src_Mat, dst_Mat, pFace); //素描滤镜
		break;
	}

	int cxl, cyl;
	Mat rgbImageL1;//opencv图片
	CImage rgbImageL2;
	CRect rectl, pic_rectl;
	int widthl, heightl;
	//改变图片大小适应picture控件
	GetDlgItem(IDC_STATIC_IMGSHOW)->GetClientRect(&pic_rectl);
	widthl = pic_rectl.right;
	heightl = pic_rectl.bottom;
	resize(dst_Mat, rgbImageL1, Size(widthl, heightl));
	MatToCImage(rgbImageL1, rgbImageL2);//OpenCV中Mat对象转MFC的CImage类的函数（见我另一篇博文）
	//获取图片的宽高度  
	cxl = rgbImageL2.GetWidth();
	cyl = rgbImageL2.GetHeight();
	//获取Picture Control控件的大小  
	GetDlgItem(IDC_STATIC_IMGSHOW)->GetWindowRect(&rectl);
	//将客户区选中到控件表示的矩形区域内  
	ScreenToClient(&rectl);
	//窗口移动到控件表示的区域  
	GetDlgItem(IDC_STATIC_IMGSHOW)->MoveWindow(rectl.left, rectl.top, cxl, cyl, TRUE);
	CWnd* pWnd = NULL;
	pWnd = GetDlgItem(IDC_STATIC_IMGSHOW);//获取控件句柄  
	pWnd->GetClientRect(&rectl);//获取句柄指向控件区域的大小  
	CDC* pDc = NULL;
	pDc = pWnd->GetDC();//获取picture的DC  
	rgbImageL2.Draw(pDc->m_hDC, rectl);//将图片绘制到picture表示的区域内  
	ReleaseDC(pDc);
	pDc = NULL;

}

void CBeautyDemoDlgDlg::OnBnClickedButtonMask()
{
	// TODO: 在此添加控件通知处理程序代码
	if (FaceNumbers < 1)  //若没有人脸直接返回
	{
		return;
	}
	Mask(src_Mat, dst_Mat, pFace);

	int cxl, cyl;
	Mat rgbImageL1;//opencv图片
	CImage rgbImageL2;
	CRect rectl, pic_rectl;
	int widthl, heightl;
	//改变图片大小适应picture控件
	GetDlgItem(IDC_STATIC_IMGSHOW)->GetClientRect(&pic_rectl);
	widthl = pic_rectl.right;
	heightl = pic_rectl.bottom;
	resize(dst_Mat, rgbImageL1, Size(widthl, heightl));
	MatToCImage(rgbImageL1, rgbImageL2);//OpenCV中Mat对象转MFC的CImage类的函数（见我另一篇博文）
	//获取图片的宽高度  
	cxl = rgbImageL2.GetWidth();
	cyl = rgbImageL2.GetHeight();
	//获取Picture Control控件的大小  
	GetDlgItem(IDC_STATIC_IMGSHOW)->GetWindowRect(&rectl);
	//将客户区选中到控件表示的矩形区域内  
	ScreenToClient(&rectl);
	//窗口移动到控件表示的区域  
	GetDlgItem(IDC_STATIC_IMGSHOW)->MoveWindow(rectl.left, rectl.top, cxl, cyl, TRUE);
	CWnd* pWnd = NULL;
	pWnd = GetDlgItem(IDC_STATIC_IMGSHOW);//获取控件句柄  
	pWnd->GetClientRect(&rectl);//获取句柄指向控件区域的大小  
	CDC* pDc = NULL;
	pDc = pWnd->GetDC();//获取picture的DC  
	rgbImageL2.Draw(pDc->m_hDC, rectl);//将图片绘制到picture表示的区域内  
	ReleaseDC(pDc);
	pDc = NULL;

}
