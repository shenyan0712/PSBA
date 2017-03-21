#pragma once

#include "cl_psba.h"
#include "psba.h"

int levmar(
	//输入
	PSBA_structPtr psbaPtr,						//
	int cnp, int pnp, int mnp,		//分别是相机参数维度，3D点坐标维度，2D图像点坐标维度
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype *finalErr);