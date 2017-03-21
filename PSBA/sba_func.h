#pragma once

#include "cl_psba.h"
#include "psba.h"


/*
计算投影误差 ex
*/
void compute_exQT(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	cl_mem cams_buffer,
	cl_mem pts3D_buffer,
	//输出
	dtype *ex				//存储假设投影点hx_ij-->第i个3D点到第j个相机上的投影与测量之间的误差。按ex_11,ex12,...,ex_nm的顺序存放。
);


/*
计算jacobi矩阵，使用OpenCL
结果存放到JA_buffer和JB_buffer
*/
void compute_jacobiQT(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype *jac_A, dtype *jac_B
);

/*
计算U矩阵
结果存放到U_buffer中
*/
void compute_U(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype coeff,
	dtype *out);

/*
计算U矩阵
结果输出到V_buffer
*/
void compute_V(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype coeff,
	dtype *out);

dtype maxElmOfUV(PSBA_structPtr psbaPtr, int totalParas, dtype *UVdiag);

void update_UV(PSBA_structPtr psbaPtr, int cnp, int pnp, int n3Dpts, int nCams, 
	dtype mu, dtype *U, dtype *V);

dtype compute_Vinv(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype *V);

/*
计算W矩阵,分立存储方式
结果输出到W_buffer中
*/
void compute_Wblks(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	int *iidx, int *jidx,
	dtype coeff,
	dtype *Wblks);


void compute_Yblks(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	int *iidx, int *jidx,
	dtype *Yblks);


void compute_S(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype *S);

/*
计算g向量. g=(J^t)*ex
*/
void compute_g(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype coeff,
	dtype *g);

/*
计算ea的值，ea=ga-Y*gb
*/
void compute_ea(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype *ea);

/*
计算eb的值，eb=gb-(W^t)*dpa
*/
void compute_eb(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype *eab);

void compute_dpb(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int nCams, int n3Dpts,
	dtype *dp);		//输出到dpb部分

void compute_newp(PSBA_structPtr psbaPtr, int nCamParas, int n3DptsParas, dtype *new_p);
void update_p(PSBA_structPtr psbaPtr, int nCamParas, int n3DptsParas, dtype *p);