// bundle_adjustment.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <time.h>
#include <iostream>

#include "psba.h"
#include "readparams.h"
#include "misc.h"
#include "cl_psba.h"


using namespace std;

FILE *debug_file;
extern int itno;

clock_t start_time, end_time,cur_tm;
clock_t tm_diff[30];



char cams_file[] = "..\\data\\54cams.txt";
char pts_file[] = "..\\data\\54pts.txt";

PSBA_struct psbaStruct;		//LM算法的cl信息结构体

//原始的K值是fu,u0,v0,ar// aspect ratio=ax/ay
dtype K[5] = { 851.57945,851.57945 *1.00169,330.24755,262.19500,0 };	//相机的内参, a_x,a_y,x0,y0
dtype KK[5] = { 851.57945,330.24755,262.19500,1.00169,0 };

int main()
{
	int cnp_origin=6;					//原始的相机的参数维度(3+3,四元数向量和位移向量）
	int cnp = 6;				//转换后的相机参数维度(6+3,R and t)
	int pnp = 3;				//3D点的维度
	int mnp=2;					//2D投影点的维度

	int n3Dpts;				//3D点的数量
	int nCams;					//相机的数量
	int n2Dprojs;			//2D投影点的数量
	dtype *motion_data;			//指向存储相机参数(3+3,四元数向量和位移向量）+3D点（3）的空间  
	dtype *initrot;				//指向存储相机初始旋转四元数的空间

	dtype *impts_data;			//指向存储2D投影点的空间。   ncams*cnp + numpts3D*pnp, 
	//impts_data存储顺序为：3D点1在对应相机上的2D投影点，...3D点n在对应相机上的2D投影点
	dtype *impts_cov;			//2D投影点的协方差，可为NULL
	char *vmask;				//3D点针对各相机的掩码，vmask[i,j]=0表示3D点i不在相机j上有图像点。空间大小nCams*n3Dpts
	dtype *motion2_data;		//如果需要转换，则将motion_data中的参数转换后存到此处
	dtype finalErr;

	motion2_data = NULL;

	errno_t err;
	err = fopen_s(&debug_file,"e:\\psba_debug.txt", "wt");
	if (err != 0) {
		printf("can't open debug file.\n");
		exit(1);
	}


	//1,读取参数
	readInitialSBAEstimate(cams_file, pts_file, cnp_origin, pnp, mnp,
		quat2vec, cnp_origin + 1,		//cnp_origin=6,但文件中由于使用4个值的四元数，所以是cnp+1
		&nCams, &n3Dpts, &n2Dprojs,
		&motion_data,
		&initrot,	//会将相机的初始的旋转四元数存到这里
		&impts_data, &impts_cov, &vmask
		);
	cout << "相机数量：" << nCams << endl;
	cout << "3D点数量：" << n3Dpts << endl;
	cout << "2D投影点的数量:" << n2Dprojs << endl;
	if (impts_cov == NULL)
		cout << "没有图像点的协方差矩阵" << endl;
	cout << "初始相机四元数：" << endl;
	for (int j = 0; j < nCams; j++)
	{
		printf("[%d]", j);
		for (int k = 0; k < 4; k++)
			printf("%le  ", initrot[j * 4 + k]);
		printf("\n");
	}

	//##########将motion_data进行分解为相机参数cams_data和3D点坐标参数。并将四元数转换为旋转矩阵
	//if (cnp == 12 && cnp_origin == 6) {
	//	motion2_data = (dtype*)malloc((nCams * cnp +n3Dpts*pnp)* sizeof(dtype));
	//	camsFormatTrans(motion_data, nCams, 3, 3, motion2_data);		//3,3分别表示四元数向量v的维度 和位移的维度
	//	memcpy((void*)&motion2_data[nCams*cnp],(void*)&motion_data[nCams*cnp_origin], n3Dpts*pnp * sizeof(dtype));		//拷贝
	//}
	//清空motion_data的相机参数rot部分
	for (int j = 0;j < nCams; j++) {
		int base = j*cnp;
		motion_data[base++] = 0;
		motion_data[base++] = 0;
		motion_data[base] = 0;
	}
	//######### 配置openCL device, 以及创建所需要的buffer ##########//
	setup_cl(&psbaStruct,cnp,pnp,mnp,nCams,n3Dpts,n2Dprojs);
	//  填充初始数据
	fill_initBuffer(&psbaStruct, cnp,pnp,mnp,nCams,n3Dpts,n2Dprojs,
		K, impts_data, initrot, motion_data);

	//调用LM算法
	start_time = clock();
	if(cnp==6)
		levmar(&psbaStruct, cnp, pnp, mnp, n3Dpts, nCams, n2Dprojs, K, impts_data, vmask,
			initrot, motion_data,&finalErr);
	else
		levmar(&psbaStruct, cnp, pnp, mnp, n3Dpts, nCams, n2Dprojs,K,impts_data, vmask,
			initrot, motion2_data, &finalErr);
	end_time = clock();

	release_buffer(&psbaStruct);
	printf("time eclipse %lf s\n", double(end_time - start_time) / CLOCKS_PER_SEC);
	//计算最终误差
	printf("final error: %le \n",finalErr);
	printf("total iteration: %d\n", itno);

	for (int i = 0; i < 30; i++)
		printf("diff tm %d=%ld \n",i, tm_diff[i] );

	system("pause");
    return 0;
}

