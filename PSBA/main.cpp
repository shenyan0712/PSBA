// bundle_adjustment.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <time.h>
#include <iostream>

#include "psba.h"
#include "readparams.h"
#include "misc.h"
#include "cl_psba.h"
#include "levmar.h"
#include "trust_region.h"


using namespace std;

FILE *debug_file;
extern int itno;

clock_t start_time, end_time;

dtype initErr;		//初始误差
clock_t cur_tm;
clock_t total_tm=0;	//算法总时间
int itno=0;			//当前迭代次数
int jac_cnt=0;			//记录jac计算次数
int Sinv_cnt=0;			//记录Binv计算次数
clock_t alloc_tm = 0;	//分配空间的时间
clock_t genidx_tm = 0;	//生成索引的时间
clock_t jac_tm = 0;		//记录jac计算总时间
clock_t S_tm = 0;		//计算S_tm的总时间
clock_t Sinv_tm=0;		//记录Sinv计算总时间
clock_t ex_tm = 0;		//计算ex的时间
clock_t g_tm = 0;		//计算g的时间
clock_t pred_ex_tm = 0;	//预测误差的计算时间


//char cams_file[] = "..\\data\\Trafalgar-50-20431-cams.txt";
//char pts_file[] = "..\\data\\Trafalgar-50-20431-pts.txt";

//char cams_file[] = "..\\data\\Dubrovnik-88-64298-cams.txt";
//char pts_file[] = "..\\data\\Dubrovnik-88-64298-pts.txt";

//char cams_file[] = "..\\data\\Rome-93-61203-cams.txt";
//char pts_file[] = "..\\data\\Rome-93-61203-pts.txt";

//char cams_file[] = "..\\data\\Trafalgar-21-11315-cams.txt";
//char pts_file[] = "..\\data\\Trafalgar-21-11315-pts.txt";

//char cams_file[] = "..\\data\\Venice-52-64053-cams.txt";
//char pts_file[] = "..\\data\\Venice-52-64053-pts.txt";

//char cams_file[] = "..\\data\\Ladybug-138-19878-cams.txt";
//char pts_file[] = "..\\data\\Ladybug-138-19878-pts.txt";

//char cams_file[] = "..\\data\\Dubrovnik-16-22106-cams.txt";
//char pts_file[] = "..\\data\\Dubrovnik-16-22106-pts.txt";

char cams_file[] = "..\\data\\Trafalgar-21-11315-cams.txt";
char pts_file[] = "..\\data\\Trafalgar-21-11315-pts.txt";

//char cams_file[] = "..\\data\\54camsvarK.txt";
//char pts_file[] = "..\\data\\54pts.txt";


PSBA_struct psbaStruct;		//LM算法的cl信息结构体

int main()
{
	//int cnp_origin=6;					//原始的相机的参数维度(3+3,四元数向量和位移向量）
	int origin_cnp = 11;				//转换后的相机参数维度(K,Q and t)
	int pnp = 3;				//3D点的维度
	int mnp=2;					//2D投影点的维度

	int n3Dpts;				//3D点的数量
	int nCams;					//相机的数量
	int n2Dprojs;			//2D投影点的数量
	dtype *motion_data;			//前nCams*cnp元素存储相机参数(5+3+3,内参，四元数向量和位移向量），后n3Dpts*pnp元素存储3D点
	dtype *initrot;				//指向存储相机初始旋转四元数的空间
	dtype *impts_data;			//指向存储2D投影点的空间。 n2Dprojs*mnp, 
	//impts_data存储顺序为：3D点1在对应相机上的2D投影点，...3D点n在对应相机上的2D投影点
	dtype *impts_cov;			//2D投影点的协方差，可为NULL
	char *vmask;				//3D点针对各相机的掩码，vmask[i,j]=0表示3D点i不在相机j上有图像点。空间大小nCams*n3Dpts
	dtype finalErr;
	dtype *Kparas;				//各个相机外参
	dtype *camsExParas;			//相机外参，Q+t, Q和Q0结合组成最终的旋转

	errno_t err;
	err = fopen_s(&debug_file,"e:\\psba_debug.txt", "wt");
	if (err != 0) {
		printf("can't open debug file.\n");
		exit(1);
	}

	//1,读取参数
	//if (readNumParams(cams_file) - 1 == cnp + 5) {
	//	cnp = cnp + 5;
	//}

	readInitialSBAEstimate(cams_file, pts_file, origin_cnp, pnp, mnp,
		quat2vec, origin_cnp + 1,
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

	/*
	dtype tmp;
	printf("init rotation:\n");
	for (int j = 0; j < nCams; j++)
	{
		printf("[%d]", j);
		for (int k = 0; k < 4; k++) {
			tmp = initrot[j * 4 + k];
			printf("%f  ", tmp);
		}
		printf("\n");
	}
	*/

	//清空motion_data的相机参数rot部分
	for (int j = 0;j < nCams; j++) {
		int base = (j+1)*origin_cnp;
		motion_data[base-4] = 0;
		motion_data[base-5] = 0;
		motion_data[base-6] = 0;
	}

	//分离数据
	//将内参从motion_data中分离,剩余的Q,t转移到motstruct，作为待优化的参数。
	int final_cnp = origin_cnp - 5;
	Kparas = (dtype*)emalloc(sizeof(dtype)*nCams * 5);
	camsExParas = (dtype*)emalloc(sizeof(dtype)*nCams*final_cnp);
	for (int j = 0; j < nCams; j++)
	{
		for(int k=0; k<5; k++)
			Kparas[j*5+k] = motion_data[j*origin_cnp +k];
		for (int k = 0; k < final_cnp; k++)
			camsExParas[j*final_cnp +k] = motion_data[j*origin_cnp +k+5];
	}

	/***********************************************/
	/*
	printf("cams K:\n");
	for (int j = 0; j < nCams; j++)
	{
		printf("[%d]", j);
		for (int k = 0; k < 5; k++) {
			tmp = Kparas[j * 5 + k];
			printf("%f  ", tmp);
		}
		printf("\n");
	}
	printf("cams Ex paras:\n");
	for (int j = 0; j < nCams; j++)
	{
		printf("[%d]", j);
		for (int k = 0; k < final_cnp; k++) {
			tmp = camsExParas[j * final_cnp + k];
			printf("%f  ", tmp);
		}
		printf("\n");
	}
	*/
	/***********************************************/

	//######### 配置openCL device, 以及创建所需要的buffer ##########//
	setup_cl(&psbaStruct, final_cnp,pnp,mnp,nCams,n3Dpts,n2Dprojs);
	//  填充初始数据
	fill_initBuffer2(&psbaStruct, final_cnp,pnp,mnp,nCams,n3Dpts,n2Dprojs,
		Kparas, impts_data, initrot, camsExParas,&motion_data[nCams*origin_cnp]);
	
	//生成索引，并存入buffer
	int *iidx = (int*)emalloc(n2Dprojs * sizeof(int));
	int *jidx = (int*)emalloc(n2Dprojs * sizeof(int));
	int *blk_idx = (int*)emalloc(n3Dpts*nCams * sizeof(int));
	int *comm3DIdx = (int*)emalloc(n3Dpts*nCams*nCams * sizeof(int));
	int *comm3DIdxCnt = (int*)emalloc(nCams*nCams * sizeof(int));
	generate_idxs(nCams, n3Dpts, n2Dprojs, impts_data, vmask, comm3DIdx, comm3DIdxCnt, iidx, jidx, blk_idx);
	fill_idxBuffer(&psbaStruct, nCams, n3Dpts, n2Dprojs, comm3DIdx, comm3DIdxCnt, iidx, jidx, blk_idx);

	int iter_flag;
	start_time = clock();
	while (true) {
		//调用LM算法
		///*

		iter_flag = levmar(&psbaStruct, final_cnp, pnp, mnp, n3Dpts, nCams, n2Dprojs, &finalErr);
		if (iter_flag != ITER_TURN_TO_TR)	
			break;
		//*/

		//调用TR算法
		///*
		iter_flag = trust_region(&psbaStruct, final_cnp, pnp, mnp, n3Dpts, nCams, n2Dprojs, blk_idx, &finalErr);
		if (iter_flag != ITER_TURN_TO_LM)
			break;
		//*/
	}
	end_time = clock();
	printf("iter_flag=%d\n", iter_flag);


	//release_buffer(&psbaStruct);
	printf("time eclipse %lf s\n", double(end_time - start_time) / CLOCKS_PER_SEC);
	//计算最终误差
	printf("initial error: %.15E \n", sqrt(initErr)/n2Dprojs);
	printf("final error: %.15E \n",sqrt(finalErr)/n2Dprojs);
	printf("total iteration: %d\n", itno);
	
	printf("alloc mem tm:\t\t%f\n", alloc_tm / (double)CLOCKS_PER_SEC);
	printf("gen idx tm:\t\t%f\n", genidx_tm / (double)CLOCKS_PER_SEC);
	printf("jac tm:\t\t%lf s\n", double(jac_tm)/ CLOCKS_PER_SEC);
	printf("ex tm tm:\t\t%f\n", ex_tm / (double)CLOCKS_PER_SEC);
	printf("g tm tm:\t\t%f\n", g_tm / (double)CLOCKS_PER_SEC);
	printf("S tm tm:\t\t%f\n", S_tm / (double)CLOCKS_PER_SEC);
	printf("Sinv tm:\t\t%f \t\t cnt:%d\n", double(Sinv_tm) / CLOCKS_PER_SEC, Sinv_cnt);
	printf("pred ex tm:\t\t%f\n", pred_ex_tm / (double)CLOCKS_PER_SEC);

	system("pause");
    return 0;
}

