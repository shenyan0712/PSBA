#include "stdafx.h"

#include <algorithm>

#include "cl_psba.h"
#include "psba.h"


extern FILE *debug_file;
extern int itno;

extern int *iidx;
extern int *jidx;


/*
计算J*X, 结果存在Jmul_buffer中
*/
inline void compute_Jmultiply(
	PSBA_structPtr psbaPtr, int mnp,
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	cl_mem X, dtype *out)
{
	cl_int err;

	//#############配置内核参数##############//
	err = clSetKernelArg(
		psbaPtr->kern_compute_Jmultiply, 0, sizeof(int), &nCams);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Jmultiply, 1, sizeof(int), &n3Dpts);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Jmultiply, 2, sizeof(int), &n2Dprojs);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Jmultiply, 3, sizeof(cl_mem), &psbaPtr->JA_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Jmultiply, 4, sizeof(cl_mem), &psbaPtr->JB_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Jmultiply, 5, sizeof(cl_mem), &psbaPtr->blkIdx_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Jmultiply, 6, sizeof(cl_mem), &X);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Jmultiply, 7, sizeof(cl_mem), &psbaPtr->Jmul_buffer);

	//############执行内核###########//
	size_t globalWorkSize[1] = { (size_t)(nCams*n3Dpts)*mnp };		//工作项总数为2D点的个数
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_compute_Jmultiply,
		1,	//1维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);
	checkErr(err, __FILE__, __LINE__);

	if (out != NULL) {
		err = clEnqueueReadBuffer(
			psbaPtr->queue, psbaPtr->Jmul_buffer, CL_TRUE, 0,
			sizeof(dtype)*(nCams*n3Dpts)*mnp,
			out, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}

#if DEBUG_JMULTI==1
	fprintf(debug_file, "*******************************iter %d*******************************\n", itno);
	//display for check
	for (int i = 0; i < n2Dprojs; i++) {
		fprintf(debug_file, "%d L2 proj err: %le   %le\n", i, ex[i * 2], ex[i * 2 + 1]);
		//fprintf(debug_file, "-----------------------------\n");
	}
	fflush(debug_file);
#endif
}


/*
计算投影误差 ex
*/
inline void compute_exQT(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	cl_mem cams_buffer,
	cl_mem pts3D_buffer,
	//输出
	dtype *ex				//存储假设投影点hx_ij-->第i个3D点到第j个相机上的投影与测量之间的误差。按ex_11,ex12,...,ex_nm的顺序存放。
	)
{
	cl_int err;

	//#############配置内核参数##############//
	//input buffer
	err = clSetKernelArg(
		psbaPtr->kern_compute_exQT, 0, sizeof(cl_mem), &psbaPtr->Kparas_buffer);
	err = clSetKernelArg(
		psbaPtr->kern_compute_exQT, 1, sizeof(cl_mem), &psbaPtr->impts_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_exQT, 2, sizeof(cl_mem), &psbaPtr->initcams_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_exQT, 3, sizeof(cl_mem), &cams_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_exQT, 4, sizeof(cl_mem), &pts3D_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_exQT, 5, sizeof(cl_mem), &psbaPtr->iidx_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_exQT, 6, sizeof(cl_mem), &psbaPtr->jidx_buffer);
	//out buffer
	err |= clSetKernelArg(
		psbaPtr->kern_compute_exQT, 7, sizeof(cl_mem), &psbaPtr->ex_buffer);

	//############执行内核###########//
	size_t globalWorkSize[1] = { n2Dprojs };		//工作项总数为2D点的个数
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_compute_exQT,
		1,	//1维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);
	checkErr(err, __FILE__, __LINE__);

	//#############取结果到ex ###############//
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->ex_buffer, CL_TRUE, 0,
		sizeof(dtype)*n2Dprojs*mnp,
		ex, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

#if DEBUG_EX==1
	fprintf(debug_file,"*******************************iter %d*******************************\n",itno);
	//display for check
	for (int i = 0; i < n2Dprojs; i++) {
		fprintf(debug_file, "%d L2 proj err: %le   %le\n",i, ex[i*2], ex[i * 2+1]);
		//fprintf(debug_file, "-----------------------------\n");
	}
	fflush(debug_file);
#endif

}



/*
计算jacobi矩阵，使用OpenCL
结果存放到JA_buffer和JB_buffer
*/
inline void compute_jacobiQT(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype *jac_A,dtype *jac_B
	)
{
	cl_int err;

	//#############配置内核参数##############//
	err = clSetKernelArg(
		psbaPtr->kern_compute_jacobiQT, 0, sizeof(cl_mem), &psbaPtr->Kparas_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_jacobiQT, 1, sizeof(cl_mem), &psbaPtr->impts_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_jacobiQT, 2, sizeof(cl_mem), &psbaPtr->initcams_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_jacobiQT, 3, sizeof(cl_mem), &psbaPtr->cams_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_jacobiQT, 4, sizeof(cl_mem), &psbaPtr->pts3D_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_jacobiQT, 5, sizeof(cl_mem), &psbaPtr->iidx_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_jacobiQT, 6, sizeof(cl_mem), &psbaPtr->jidx_buffer);
	//out
	err |= clSetKernelArg(
		psbaPtr->kern_compute_jacobiQT, 7, sizeof(cl_mem), &psbaPtr->JA_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_jacobiQT, 8, sizeof(cl_mem), &psbaPtr->JB_buffer);
	checkErr(err, __FILE__, __LINE__);


	//############执行内核###########//
	size_t globalWorkSize = n2Dprojs;		//工作项总数为2D点的个数
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_compute_jacobiQT,
		1,	//1维索引空间
		NULL,	//No offset
		&globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);
	checkErr(err, __FILE__, __LINE__);



#if DEBUG_JAC==1

	//#############取结果到jac_A,jac_A ###############//
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->JA_buffer, CL_TRUE, 0,
		sizeof(dtype)*n2Dprojs*mnp*cnp,
		jac_A, 0, NULL, NULL);
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->JB_buffer, CL_TRUE, 0,
		sizeof(dtype)*n2Dprojs*mnp*pnp,
		jac_B, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	fprintf(debug_file, "*********************************iter %d*******************************\n",itno);
	//display for check
	for (int i = 0; i < n2Dprojs; i++) {
		fprintf(debug_file,"Jac idx:%d\n", i);
		fprintf(debug_file,"JA(%d,%d):\n", iidx[i], jidx[i]);
		for (int r = 0; r < 2; r++) {
			for (int c = 0; c < cnp; c++)
				fprintf(debug_file,"%le  ", jac_A[i*2*cnp+r*cnp + c]);
			fprintf(debug_file,"\n");
		}
		fprintf(debug_file,"JB(%d,%d):\n", iidx[i], jidx[i]);
		for (int r = 0; r < 2; r++) {
			for (int c = 0;c < pnp;c++)
				fprintf(debug_file,"%le  ", jac_B[i*2*pnp+r * pnp + c]);
			fprintf(debug_file,"\n");
		}
		fprintf(debug_file,"---------------------------------------\n");
	}
	fflush(debug_file);
#endif
	
	//clReleaseMemObject(K_buffer);
	//clReleaseMemObject(impts_buffer);
	//clReleaseMemObject(cams_buffer);
	//clReleaseMemObject(pts3D_buffer);
	//clReleaseMemObject(vmask_buffer);
	//clReleaseMemObject(iidx_buffer);
	//clReleaseMemObject(jidx_buffer);
	//clReleaseMemObject(JA_buffer);
	//clReleaseMemObject(JB_buffer);
}


/*
计算U矩阵
结果存放到U_buffer中
*/
inline void compute_U(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype coeff,
	dtype *out)
{
	cl_int err;

	//err = clSetKernelArg(psbaPtr->kern_compute_U, 0, sizeof(int), &cnp);
	int argnum = 0;
	err |= clSetKernelArg(
		psbaPtr->kern_compute_U, argnum++, sizeof(int), &nCams);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_U, argnum++, sizeof(int), &n3Dpts);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_U, argnum++, sizeof(int), &n2Dprojs);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_U, argnum++, sizeof(cl_mem), &psbaPtr->JA_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_U, argnum++, sizeof(cl_mem), &psbaPtr->blkIdx_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_U, argnum++, sizeof(cl_mem), &psbaPtr->U_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_U, argnum++, sizeof(cl_mem), &psbaPtr->UVdiag_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_U, argnum++, sizeof(dtype), &coeff);

	//############执行内核###########//
	//内核执行的颗粒度为nCams*cnp*cnp，即U1,...,Um的每一个元素。处理时将U1,...,Um拼成一个矩阵。
	size_t globalWorkSize[2] = { (size_t)cnp,(size_t)nCams*cnp };
	//size_t globalWorkSize[2] = { 1,1 };
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_compute_U,
		2,	//2维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);
	checkErr(err, __FILE__, __LINE__);

	if (out != NULL)
	{
		err = clEnqueueReadBuffer(
			psbaPtr->queue, psbaPtr->U_buffer, CL_TRUE, 0,
			nCams*cnp*cnp * sizeof(dtype),
			out, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}


#if DEBUG_UMAT==1
	dtype *U;
	U = (dtype *)emalloc(nCams*cnp*cnp * sizeof(dtype));
	//#############取结果到U###############//
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->U_buffer, CL_TRUE, 0,
		nCams*cnp*cnp * sizeof(dtype),
		U, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	fprintf(debug_file, "**************************************iter %d***************************\n", itno);
	//display for check
	for (int j = 0; j<nCams; j++)
	{
		fprintf(debug_file, "U%d:\n", j);
		for (int r = 0; r < cnp; r++) {
			for (int c = 0; c < cnp; c++)
				fprintf(debug_file, "%lf  ", U[j*cnp*cnp + r*cnp + c]);
			fprintf(debug_file, "\n");
		}
	}
	fflush(debug_file);
	free(U);
#endif 

}

/*
计算U矩阵
结果输出到V_buffer
*/
inline void compute_V(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype coeff,
	dtype *out)
{
	cl_int err;

	//err = clSetKernelArg(psbaPtr->kern_compute_V, 0, sizeof(int), &cnp);
	//err |= clSetKernelArg(psbaPtr->kern_compute_V, 1, sizeof(int), &pnp);
	int argnum = 0;
	err |= clSetKernelArg(
		psbaPtr->kern_compute_V, argnum++, sizeof(int), &nCams);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_V, argnum++, sizeof(int), &n3Dpts);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_V, argnum++, sizeof(int), &n2Dprojs);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_V, argnum++, sizeof(cl_mem), &psbaPtr->JB_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_V, argnum++, sizeof(cl_mem), &psbaPtr->blkIdx_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_V, argnum++, sizeof(cl_mem), &psbaPtr->V_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_V, argnum++, sizeof(cl_mem), &psbaPtr->UVdiag_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_V, argnum++, sizeof(dtype), &coeff);

	//############执行内核###########//
	size_t globalWorkSize[2] = { (size_t)pnp,(size_t)n3Dpts*pnp };
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_compute_V,
		2,	//2维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);
	checkErr(err, __FILE__, __LINE__);

	if (out != NULL)
	{
		err = clEnqueueReadBuffer(
			psbaPtr->queue, psbaPtr->V_buffer, CL_TRUE, 0,
			n3Dpts*pnp*pnp * sizeof(dtype),
			out, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}


#if DEBUG_VMAT==1
	dtype *V;
	V = (dtype*)emalloc(n3Dpts*pnp*pnp * sizeof(dtype));
	//#############取结果到V###############//
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->V_buffer, CL_TRUE, 0,
		n3Dpts*pnp*pnp * sizeof(dtype),
		V, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	fprintf(debug_file, "************************************iter %d******************************\n", itno);
	//display for check
	for (int i = 0; i<n3Dpts; i++)
	{
		fprintf(debug_file, "V%d:\n", i);
		for (int r = 0; r < pnp; r++) {
			for (int c = 0; c < pnp; c++)
				fprintf(debug_file, "%lf  ", V[i*pnp*pnp + r*pnp + c]);
			fprintf(debug_file, "\n");
		}
	}
	fflush(debug_file);
	free(V);
#endif

}

/*
初始化damping因子，其等于U，V上最大元素乘以tau
*/
inline dtype maxElmOfUV(PSBA_structPtr psbaPtr, int nTotalPara,dtype *UVdiag)
{
	cl_int err;
	dtype max;

	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->UVdiag_buffer, CL_TRUE, 0,
		nTotalPara * sizeof(dtype),
		UVdiag, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	//
	max = 0.0;
	//printf("UVdiag:");
	for (int i = 0; i < nTotalPara; i++)
	{
		//printf("%le\t", UVdiag[i]);
		if (UVdiag[i] > max)
			max = UVdiag[i];
	}
	//printf("\n");
	return max;
}

/*
计算W矩阵,分立存储方式
结果输出到W_buffer中
*/
inline void compute_Wblks(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	int *iidx, int *jidx,
	dtype coeff,
	dtype *out)
{
	cl_int err;

	//err = clSetKernelArg(psbaPtr->kern_compute_Wblks, 0, sizeof(int), &cnp);
	//err = clSetKernelArg(psbaPtr->kern_compute_Wblks, 1, sizeof(int), &pnp);
	int argnum = 0;
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Wblks, argnum++, sizeof(int), &nCams);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Wblks, argnum++, sizeof(int), &n3Dpts);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Wblks, argnum++, sizeof(int), &n2Dprojs);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Wblks, argnum++, sizeof(cl_mem), &psbaPtr->JA_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Wblks, argnum++, sizeof(cl_mem), &psbaPtr->JB_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Wblks, argnum++, sizeof(cl_mem), &psbaPtr->iidx_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Wblks, argnum++, sizeof(cl_mem), &psbaPtr->jidx_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Wblks, argnum++, sizeof(cl_mem), &psbaPtr->W_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Wblks, argnum++, sizeof(dtype), &coeff);

	//############执行内核###########//
	//内核执行的颗粒度为cnp x n2Dprojs*pnp
	size_t globalWorkSize[2] = { (size_t)cnp, (size_t)n2Dprojs*pnp };
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_compute_Wblks,
		2,	//2维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);
	checkErr(err, __FILE__, __LINE__);

	if (out != NULL)
	{
		err = clEnqueueReadBuffer(
			psbaPtr->queue, psbaPtr->W_buffer, CL_TRUE, 0,
			n2Dprojs*cnp*pnp * sizeof(dtype),
			out, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}

#if DEBUG_WMAT==1
	dtype *W;
	W = (dtype*)emalloc(n2Dprojs*cnp*pnp * sizeof(dtype));
	//#############取结果到jac_A,jac_A ###############//
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->W_buffer, CL_TRUE, 0,
		n2Dprojs*cnp*pnp * sizeof(dtype),
		W, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	fprintf(debug_file, "*****************************************iter %d*********************************\n", itno);
	for (int idx = 0; idx < n2Dprojs; idx++) {
		fprintf(debug_file, "W%d%d:\n", iidx[idx], jidx[idx]);
		for (int r = 0; r < cnp; r++) {
			for (int c = 0; c < pnp; c++)
				fprintf(debug_file, "%lf  ", W[idx*cnp*pnp + r*pnp + c]);
			fprintf(debug_file, "\n");
		}
	}
	fflush(debug_file);
	free(W);
#endif
}


/*
计算g向量. g=coeff*(J^t)*ex
结果输出到g_buffer,以及g数组
*/
inline void compute_g(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype coeff,
	dtype *out)
{
	cl_int err;
	//err = clSetKernelArg(psbaPtr->kern_compute_g, 0, sizeof(int), &cnp);
	//err = clSetKernelArg(psbaPtr->kern_compute_g, 1, sizeof(int), &pnp);
	int argnum = 0;
	err |= clSetKernelArg(
		psbaPtr->kern_compute_g, argnum++, sizeof(int), &nCams);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_g, argnum++, sizeof(int), &n3Dpts);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_g, argnum++, sizeof(int), &n2Dprojs);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_g, argnum++, sizeof(dtype), &coeff);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_g, argnum++, sizeof(cl_mem), &psbaPtr->JA_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_g, argnum++, sizeof(cl_mem), &psbaPtr->JB_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_g, argnum++, sizeof(cl_mem), &psbaPtr->blkIdx_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_g, argnum++, sizeof(cl_mem), &psbaPtr->ex_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_g, argnum++, sizeof(cl_mem), &psbaPtr->g_buffer);

	//############执行内核###########//
	//内核执行的颗粒度为nCams*cnp + n3Dpts*pnp
	size_t globalWorkSize[1] = { (size_t)nCams*cnp + n3Dpts*pnp };
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_compute_g,
		1,	//2维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);

	if (out != NULL) {
		//#############取结果到gb ###############//
		err = clEnqueueReadBuffer(
			psbaPtr->queue, psbaPtr->g_buffer, CL_TRUE, 0,
			(n3Dpts*pnp + nCams*cnp) * sizeof(dtype),
			out, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}

#if DEBUG_G==1
	dtype *g;
	g = (dtype*)emalloc((n3Dpts*pnp + nCams*cnp) * sizeof(dtype));
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->g_buffer, CL_TRUE, 0,
		(n3Dpts*pnp + nCams*cnp) * sizeof(dtype),
		g, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	fprintf(debug_file, "************************************iter %d**************************************\n", itno);
	fprintf(debug_file, "ga:\n");
	for (int j = 0; j < nCams; j++) {
		fprintf(debug_file, "%d =>", j);
		for (int k = 0; k < cnp; k++)
			fprintf(debug_file, "%le\t", g[j*cnp + k]);
		fprintf(debug_file, "\n");
	}
	fprintf(debug_file, "gb:\n");
	for (int i = 0; i < n3Dpts; i++) {
		fprintf(debug_file, "%d =>", i);
		for (int k = 0; k < pnp; k++)
			fprintf(debug_file, "%le\t", g[nCams*cnp + i*pnp + k]);
		fprintf(debug_file, "\n");
	}
	fflush(debug_file);
	free(g);
#endif
	}


/*
U,V对角线元素加上mu
在U_buffer和V_buffer上操作
*/
inline void update_UV(PSBA_structPtr psbaPtr, int cnp, int pnp, int n3Dpts, int nCams, dtype mu, dtype *U,dtype *V)
{
	cl_int err;

	//err = clSetKernelArg(psbaPtr->kern_update_UV, 0, sizeof(int), &cnp);
	//err = clSetKernelArg(psbaPtr->kern_update_UV, 2, sizeof(int), &pnp);
	err |= clSetKernelArg(
		psbaPtr->kern_update_UV, 0, sizeof(int), &nCams);
	err |= clSetKernelArg(
		psbaPtr->kern_update_UV, 1, sizeof(int), &n3Dpts);
	err |= clSetKernelArg(
		psbaPtr->kern_update_UV, 2, sizeof(cl_mem), &psbaPtr->U_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_update_UV, 3, sizeof(cl_mem), &psbaPtr->V_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_update_UV, 4, sizeof(dtype), &mu);

	size_t globalWorkSize[1] = { n3Dpts*pnp+nCams*cnp };
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_update_UV,
		1,	//2维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);

#if (DEBUG_VMAT==1 && DEBUG_UMAT==1)
	//#############取结果到Vinv ###############//
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->U_buffer, CL_TRUE, 0,
		nCams*cnp*cnp* sizeof(dtype),
		U, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->V_buffer, CL_TRUE, 0,
		n3Dpts*pnp*pnp* sizeof(dtype),
		V, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
 
	fprintf(debug_file, "*******************************************************************\n");
	//display for check
	for (int j = 0; j<nCams; j++)
	{
		fprintf(debug_file, "U%d updated:\n", j);
		for (int r = 0; r < cnp; r++) {
			for (int c = 0; c < cnp; c++)
				fprintf(debug_file, "%le\t", U[j*cnp*cnp + r*cnp + c]);
			fprintf(debug_file, "\n");
		}
	}
	for (int i = 0; i<n3Dpts; i++)
	{
		fprintf(debug_file, "V%d updated:\n", i);
		for (int r = 0; r < pnp; r++) {
			for (int c = 0; c < pnp; c++)
				fprintf(debug_file, "%le\t", V[i*pnp*pnp + r*pnp + c]);
			fprintf(debug_file, "\n");
		}
	}
#endif

}


/*
用UVdiag来恢复U,V的对角线元素。
*/
inline void restore_UVdiag(PSBA_structPtr psbaPtr, int cnp, int pnp, int n3Dpts, int nCams)
{
	cl_int err;

	err |= clSetKernelArg(
		psbaPtr->kern_restore_UVdiag, 0, sizeof(int), &nCams);
	err |= clSetKernelArg(
		psbaPtr->kern_restore_UVdiag, 1, sizeof(int), &n3Dpts);
	err |= clSetKernelArg(
		psbaPtr->kern_restore_UVdiag, 2, sizeof(cl_mem), &psbaPtr->U_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_restore_UVdiag, 3, sizeof(cl_mem), &psbaPtr->V_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_restore_UVdiag, 4, sizeof(cl_mem), &psbaPtr->UVdiag_buffer);

	size_t globalWorkSize[1] = { n3Dpts*pnp + nCams*cnp };
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_restore_UVdiag,
		1,	//2维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);
}


/*
计算V矩阵的逆矩阵
结果仍然在V_buffer中，占用其下三角（包括对角线），上三角仍为V_jj。
*/
inline dtype compute_Vinv(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype *V					//
)
{
	cl_int err;
	dtype ret;

	//err = clSetKernelArg(psbaPtr->kern_compute_Vinv, 0, sizeof(int), &pnp);
	err = clSetKernelArg(
		psbaPtr->kern_compute_Vinv, 0, sizeof(int), &n3Dpts);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Vinv, 1, sizeof(cl_mem), &psbaPtr->V_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Vinv, 2, sizeof(cl_mem), &psbaPtr->ret_buffer);
	checkErr(err, __FILE__, __LINE__);

	//############执行内核###########//
	size_t globalWorkSize[1] = { n3Dpts };
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_compute_Vinv,
		1,	//2维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);
	checkErr(err, __FILE__, __LINE__);

	clEnqueueReadBuffer(psbaPtr->queue, psbaPtr->ret_buffer, CL_TRUE, 0,
		sizeof(dtype),
		&ret, 0, NULL, NULL);
	return ret;

#if DEBUG_VinvMAT==1
	//#############取结果到Vinv ###############//
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->V_buffer, CL_TRUE, 0,
		n3Dpts*pnp*pnp* sizeof(dtype),
		V, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	fprintf(debug_file,"*************************************iter %d*************************************\n",itno);
	//display Vinv for check
	for (int i = 0; i<n3Dpts; i++)
	{
		fprintf(debug_file,"Vinv%d:\n", i);
		for (int r = 0; r < pnp; r++) {
			for (int c = 0; c < pnp; c++)
				fprintf(debug_file,"%le  ", V[i*pnp*pnp + r*pnp + c]);
			fprintf(debug_file,"\n");
		}
	}
	fflush(debug_file);
#endif

}


/*
计算Y矩阵，Yij=Wij*(Vi^-1), 其中Vi^-1存储在Vi的下三角
结果输出到Y_buffer
*/
inline void compute_Yblks(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	int *iidx, int *jidx,
	dtype *out)
{
	cl_int err;

	//err = clSetKernelArg(psbaPtr->kern_compute_Yblks, 0, sizeof(int), &cnp);
	//err |= clSetKernelArg(psbaPtr->kern_compute_Yblks, 1, sizeof(int), &pnp);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Yblks, 0, sizeof(int), &nCams);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Yblks, 1, sizeof(int), &n3Dpts);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Yblks, 2, sizeof(cl_mem), &psbaPtr->iidx_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Yblks, 3, sizeof(cl_mem), &psbaPtr->W_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Yblks, 4, sizeof(cl_mem), &psbaPtr->V_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_Yblks, 5, sizeof(cl_mem), &psbaPtr->Y_buffer);

	//############执行内核###########//
	//内核执行的颗粒度为n3Dpts*pnp x nCams*cnp，即每个工作项处理Y中的一个元素
	size_t globalWorkSize[2] = { (size_t)cnp,(size_t)n2Dprojs*pnp };
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_compute_Yblks,
		2,	//2维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);
	checkErr(err, __FILE__, __LINE__);

#if DEBUG_YMAT==1
	dtype *Y;
	Y = (dtype*)emalloc(cnp*n2Dprojs*pnp * sizeof(dtype));
	//#############取结果到Vinv ###############//
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->Y_buffer, CL_TRUE, 0,
		cnp*n2Dprojs*pnp * sizeof(dtype),
		Y, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	fprintf(debug_file, "****************************************iter %d**********************************\n", itno);
	//display Yij for check
	for (int idx = 0; idx < n2Dprojs; idx++)
	{
		fprintf(debug_file, "Y%d_%d=[\n", iidx[idx], jidx[idx]);
		for (int r = 0; r < cnp; r++) {
			for (int c = 0; c < pnp; c++)
				fprintf(debug_file, "%le  ", Y[idx*cnp*pnp + r*pnp + c]);
			fprintf(debug_file, "...\n");
}
	}
	fflush(debug_file);
	free(Y);
#endif

}


/*
计算S矩阵，S=U-Y*W, 使用Yblks,Wblks
*/
inline void compute_S(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype *S)
{
	cl_int err;

	//err = clSetKernelArg(psbaPtr->kern_compute_S, 0, sizeof(int), &cnp);
	//err |= clSetKernelArg(psbaPtr->kern_compute_S, 1, sizeof(int), &pnp);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_S, 0, sizeof(int), &nCams);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_S, 1, sizeof(int), &n3Dpts);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_S, 2, sizeof(cl_mem), &psbaPtr->blkIdx_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_S, 3, sizeof(cl_mem), &psbaPtr->U_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_S, 4, sizeof(cl_mem), &psbaPtr->Y_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_S, 5, sizeof(cl_mem), &psbaPtr->W_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_S, 6, sizeof(cl_mem), &psbaPtr->comm3DIdx_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_S, 7, sizeof(cl_mem), &psbaPtr->comm3DIdxCnt_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_S, 8, sizeof(cl_mem), &psbaPtr->S_buffer);

	//############执行内核###########//
	//内核执行的颗粒度为n3Dpts*pnp x nCams*cnp，即每个工作项处理Y中的一个元素
	size_t globalWorkSize[2] = { nCams*cnp, nCams*cnp };
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_compute_S,
		2,	//2维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);

#if DEBUG_SMAT==1
	//#############取结果到Vinv ###############//
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->S_buffer, CL_TRUE, 0,
		nCams*cnp * nCams*cnp * sizeof(dtype),
		S, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	//display S for check
	fprintf(debug_file, "***********************************iter %d***************************************\n", itno);
	fprintf(debug_file, "S:\n");
	for (int r = 0; r < nCams*cnp; r++) {
		for (int c = 0; c < nCams * cnp; c++)
			fprintf(debug_file, "%le  ", S[r*nCams*cnp + c]);
		fprintf(debug_file, ";\n");
	}
	fflush(debug_file);
#endif

}


/*
计算ea的值，ea=ga-Y*gb
结果输出到ea_buffer
*/
inline void compute_ea(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype *ea)
{
	cl_int err;

	//
	//err = clSetKernelArg(psbaPtr->kern_compute_ea, 0, sizeof(int), &cnp);
	//err = clSetKernelArg(psbaPtr->kern_compute_ea, 1, sizeof(int), &pnp);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_ea, 0, sizeof(int), &nCams);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_ea, 1, sizeof(int), &n3Dpts);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_ea, 2, sizeof(int), &n2Dprojs);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_ea, 3, sizeof(cl_mem), &psbaPtr->blkIdx_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_ea, 4, sizeof(cl_mem), &psbaPtr->Y_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_ea, 5, sizeof(cl_mem), &psbaPtr->g_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_ea, 6, sizeof(cl_mem), &psbaPtr->eab_buffer);

	//############执行内核###########//
	//内核执行的颗粒度为nCams*cnp x n3Dpts*pnp
	size_t globalWorkSize[1] = { nCams*cnp};
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_compute_ea,
		1,	//2维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);

#if DEBUG_EAB==1
	//#############取结果到gb ###############//
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->eab_buffer, CL_TRUE, 0,
		(nCams*cnp) * sizeof(dtype),
		ea, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	fprintf(debug_file,"***********************************iter %d***************************************\n",itno);
	fprintf(debug_file,"ea:\n");
	for (int j = 0; j < nCams; j++) {
		fprintf(debug_file,"%d =>", j);
		for (int k = 0; k < cnp; k++)
			fprintf(debug_file,"%le\t", ea[j*cnp + k]);
		fprintf(debug_file,"\n");
	}
	fflush(debug_file);
#endif
}


/*
计算eb的值，eb=gb-(W^t)*dpa
*/
inline void compute_eb(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype *eab)
{
	cl_int err;

	//err = clSetKernelArg(psbaPtr->kern_compute_eb, 0, sizeof(int), &cnp);
	//err = clSetKernelArg(psbaPtr->kern_compute_eb, 1, sizeof(int), &pnp);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_eb, 0, sizeof(int), &nCams);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_eb, 1, sizeof(int), &n3Dpts);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_eb, 2, sizeof(int), &n2Dprojs);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_eb, 3, sizeof(cl_mem), &psbaPtr->blkIdx_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_eb, 4, sizeof(cl_mem), &psbaPtr->W_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_eb, 5, sizeof(cl_mem), &psbaPtr->dp_buffer);		//使用其中的dpa部分
	err |= clSetKernelArg(
		psbaPtr->kern_compute_eb, 6, sizeof(cl_mem), &psbaPtr->g_buffer);		//使用其中的gb部分
	err |= clSetKernelArg(
		psbaPtr->kern_compute_eb, 7, sizeof(cl_mem), &psbaPtr->eab_buffer);

	//############执行内核###########//
	size_t globalWorkSize[1] = { n3Dpts*pnp };
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_compute_eb,
		1,	//2维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);


#if DEBUG_EAB==1
	//#############取结果到eab ###############//
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->eab_buffer, CL_TRUE, 0,
		(nCams*cnp+n3Dpts*pnp) * sizeof(dtype),
		eab, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	fprintf(debug_file,"*************************************iter %d************************************\n",itno);
	fprintf(debug_file,"eb:\n");
	for (int i = 0; i < n3Dpts; i++) {
		fprintf(debug_file,"%d =>", i);
		for (int k = 0; k < pnp; k++)
			fprintf(debug_file,"%le\t", eab[nCams*cnp+i*pnp + k]);
		fprintf(debug_file,"\n");
	}
	fflush(debug_file);
#endif

}

/*
计算dpb值，dpb=(V^-1)*gb
*/
inline void compute_dpb(
	//输入
	PSBA_structPtr psbaPtr,
	int cnp,int pnp,int nCams,int n3Dpts,
	dtype *dp)		//输出到dpb部分
{
	cl_int err;

	//err = clSetKernelArg(psbaPtr->kern_compute_dpb, 0, sizeof(int), &cnp);
	err |= clSetKernelArg(psbaPtr->kern_compute_dpb, 1, sizeof(int), &pnp);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_dpb, 0, sizeof(int), &nCams);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_dpb, 1, sizeof(int), &n3Dpts);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_dpb, 2, sizeof(cl_mem), &psbaPtr->V_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_dpb, 3, sizeof(cl_mem), &psbaPtr->eab_buffer);		//gb
	err |= clSetKernelArg(
		psbaPtr->kern_compute_dpb, 4, sizeof(cl_mem), &psbaPtr->dp_buffer);		//dpb

	//############执行内核###########//
	//内核执行的颗粒度为n3Dpts*pnp
	size_t globalWorkSize[1] = { n3Dpts*pnp};
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_compute_dpb,
		1,	//2维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);

	//#############取结果到dp ###############//
	err = clEnqueueReadBuffer(
		psbaPtr->queue, psbaPtr->dp_buffer, CL_TRUE, 0,
		(nCams*cnp+n3Dpts*pnp) * sizeof(dtype),
		dp, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

#if DEBUG_DP==1
	fprintf(debug_file,"****************************************iter %d*********************************\n",itno);
	fprintf(debug_file,"dpb:\n");
	for (int i = 0; i < n3Dpts*pnp; i++) 
		fprintf(debug_file,"%le\t", dp[nCams*cnp+i]);
	fprintf(debug_file,"\n");
	fflush(debug_file);
#endif
}

/*
new_p=p+dp, 结果存放在newCams_buffer和newPts3D_buffer中
*/
inline void compute_newp(PSBA_structPtr psbaPtr,int nCamParas,int n3DptsParas, dtype *new_p)
{
	cl_int err;

	err = clSetKernelArg(
		psbaPtr->kern_compute_newp, 0, sizeof(int), &nCamParas);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_newp, 1, sizeof(int), &n3DptsParas);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_newp, 2, sizeof(cl_mem), &psbaPtr->cams_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_newp, 3, sizeof(cl_mem), &psbaPtr->pts3D_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_newp, 4, sizeof(cl_mem), &psbaPtr->dp_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_newp, 5, sizeof(cl_mem), &psbaPtr->newCams_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_compute_newp, 6, sizeof(cl_mem), &psbaPtr->newPts3D_buffer);

	//############执行内核###########//
	size_t globalWorkSize[1] = { nCamParas + n3DptsParas };
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_compute_newp,
		1,	//1维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);

	if (new_p != NULL)
	{
		err = clEnqueueReadBuffer(
			psbaPtr->queue, psbaPtr->dp_buffer, CL_TRUE, 0,
			(nCamParas + n3DptsParas) * sizeof(dtype),
			new_p, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}


}


/*
p=new_p
*/
inline void update_p(PSBA_structPtr psbaPtr,int nCamParas, int n3DptsParas, dtype *p)
{
	cl_int err;

	err = clSetKernelArg(
		psbaPtr->kern_update_p, 0, sizeof(int), &nCamParas);
	err |= clSetKernelArg(
		psbaPtr->kern_update_p, 1, sizeof(int), &n3DptsParas);
	err |= clSetKernelArg(
		psbaPtr->kern_update_p, 2, sizeof(cl_mem), &psbaPtr->cams_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_update_p, 3, sizeof(cl_mem), &psbaPtr->pts3D_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_update_p, 4, sizeof(cl_mem), &psbaPtr->newCams_buffer);
	err |= clSetKernelArg(
		psbaPtr->kern_update_p, 5, sizeof(cl_mem), &psbaPtr->newPts3D_buffer);

	//############执行内核###########//
	size_t globalWorkSize[1] = { nCamParas + n3DptsParas };
	err = clEnqueueNDRangeKernel(
		psbaPtr->queue,
		psbaPtr->kern_update_p,
		1,	//2维索引空间
		NULL,	//No offset
		globalWorkSize,
		NULL,	//Let opencl decide local worksize
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	err = clFinish(psbaPtr->queue);

	if (p != NULL)
	{
		err = clEnqueueReadBuffer(
			psbaPtr->queue, psbaPtr->cams_buffer, CL_TRUE, 0,
			nCamParas * sizeof(dtype),
			p, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);

		err = clEnqueueReadBuffer(
			psbaPtr->queue, psbaPtr->pts3D_buffer, CL_TRUE, 0,
			n3DptsParas * sizeof(dtype),
			&p[nCamParas], 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}

}