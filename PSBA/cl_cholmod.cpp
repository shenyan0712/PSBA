#include "stdafx.h"

#include <math.h>

#include "psba.h"
#include "cl_cholmod.h"

extern int itno;
extern FILE *debug_file;


/*
修正cholesky分解（modified cholesky）：
集成了对非正定的修正,enqueue on device
输入：
matBuf:待分解方阵,尺寸为matSize*matSize
blkBackupBuf:  >=3*matSize, 用于对列块的备份
diagBlkAuxBuf: >=3*matSize, 辅助计算，同时存放Ljj^-1结果
diagBuf: 存放原始矩阵的对角元素
输出：
matBuf：其下三角矩阵为分解的结果。
outMat：当copyOut为True时，结果拷贝到outMat
*/
//dtype cholesky_mod(int matsize, SPDInv_structPtr SPDInvPtr, dtype *outMat, bool copyOut)
void cholmod_blk(cl_command_queue queue, cl_kernel kern_cholmod_blk, cl_kernel kern_mat_max,
	cl_mem matBuf, cl_mem blkBackupBuf, cl_mem diagBlkAuxBuf, cl_mem diagBuf, cl_mem retBuf,
	int matSize, dtype *outMat)
{
	int argNum;
	cl_int err;
	dtype beta, delta, ret;
	dtype data[2 * 18];

	//printBuf2D(stdout, queue, matBuf, matSize, matSize, "************A:");

	//get delta,beta
	get_delta_beta(queue, kern_mat_max, matBuf, diagBlkAuxBuf, matSize, &delta, &beta);

	//set kernel arguments
	argNum = 0;
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(cl_mem), &matBuf);
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(cl_mem), &blkBackupBuf);
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(cl_mem), &diagBlkAuxBuf);
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(cl_mem), &diagBuf);
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(cl_mem), &retBuf);
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(dtype) * 9, NULL);		//T_ii块
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(dtype) * 6, NULL);		//L_ii,只存储下三角
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(int), &matSize);
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(dtype), &beta);
	err = clSetKernelArg(kern_cholmod_blk, argNum++, sizeof(dtype), &delta);
	checkErr(err, __FILE__, __LINE__);

	//for (int j = 0; j < matSize/3; j++) {
	int j = 0;
	err = clSetKernelArg(kern_cholmod_blk, argNum, sizeof(int), &j);		//块矩阵的第j列
																			//执行内核
	size_t global_size[2] = { 3,3 };
	size_t local_size[2] = { 3,3 };
	err = clEnqueueNDRangeKernel(queue, kern_cholmod_blk,
		2, NULL,	//1D work space
		global_size,
		local_size,
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFinish(queue);

	//printBuf2D(stdout, queue, matBuf, matSize, matSize, "***********L(cholesky):");
	//printBuf2D(stdout, queue, diagBlkAuxBuf, 3, matSize, "************Linv:");
	//printBuf2D(stdout, queue, blkBackupBuf, 3, matSize, "*************aux:");
	//printBuf1D(stdout, queue, diagBuf, matSize, "************diag:");
	//printf("*****%d*****\n", j);
	//printBuf1D(stdout, queue, retBuf, 1, "***********ret:");
	//}

	err = clEnqueueReadBuffer(queue, retBuf, CL_TRUE, 0,
		sizeof(dtype), &ret, 0, NULL, NULL);

	// 读取结果
	if (outMat != NULL)
	{
		err = clEnqueueReadBuffer(queue, matBuf, CL_TRUE, 0,
			sizeof(dtype)*matSize*matSize, outMat, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}

#if DEBUG_CHOLESKY==1
	dtype *debug;
	debug = (dtype*)malloc(sizeof(dtype)*matSize*matSize);
	err = clEnqueueReadBuffer(queue, matBuf, CL_TRUE, 0,
		sizeof(dtype)*matSize*matSize, debug, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	printf("L(cholesky):\n");
	for (int r = 0; r < matSize; r++)
	{
		for (int c = 0; c < matSize; c++)
			printf("%le\t", debug[r*matSize + c]);
		printf("\n");
	}
	free(debug);

#endif
}

/*********************************************************************************************/
/*********************************************************************************************/


void get_delta_beta(cl_command_queue queue, cl_kernel kern_delta_beta, cl_mem matBuf,cl_mem auxBuf,
	int matsize, dtype *delta, dtype *beta)
//void get_delta_beta(int matsize, SPDInv_structPtr SPDInvPtr, dtype *delta, dtype *beta)
{
	int argNum, excludeDiag, outOffset, inOffset;
	cl_int err;
	dtype res[2];
	//*********************step 1 得到每一行非对角元最大值，存到buf_diagBlk第一行， 同时将对角元素存到第二行
	argNum = 0;
	excludeDiag = 1;
	outOffset = 0;
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(cl_mem), &matBuf);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(cl_mem), &auxBuf);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(int), &outOffset);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(int), &matsize);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(int), &excludeDiag);
	checkErr(err, __FILE__, __LINE__);

	size_t row_size = matsize;
	err = clEnqueueNDRangeKernel(queue, kern_delta_beta,
		1, NULL,	//1D work space
		&row_size,
		NULL,
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFinish(queue);

	//err = clEnqueueReadBuffer(SPDInvPtr->queue, SPDInvPtr->buf_diagBlk, CL_TRUE, 0,
	//	sizeof(dtype) * 6, &out, 0, NULL, NULL);

	argNum = 0;
	excludeDiag = 0;
	outOffset =2*matsize;
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(cl_mem), &auxBuf);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(cl_mem), &auxBuf);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(int), &outOffset);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(int), &matsize);
	err = clSetKernelArg(kern_delta_beta, argNum++, sizeof(int), &excludeDiag);

	row_size = 2;
	err = clEnqueueNDRangeKernel(queue, kern_delta_beta,
		1, NULL,	//1D work space
		&row_size,
		NULL,
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFinish(queue);

	err = clEnqueueReadBuffer(queue, auxBuf, CL_TRUE, sizeof(dtype)*outOffset,
		sizeof(dtype) * 2, &res, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	*delta = 1e-15*fmax(res[0] + res[1], 1);		//1e-15为double的有效精度
	*beta = fmax(res[1], 1e-15);
	*beta = fmax(*beta, res[0] / sqrt(matsize*matsize - 1));
	*beta = sqrt(*beta);

	//*****************************************************
}

/*
计算modified cholesky修正的对角元素E
输入：
buf_diag --原矩阵的对角元素
输出：
buf_diag --修正的对角元素E
*/
void compute_cholmod_E(cl_command_queue queue, cl_kernel kern_cholmod_E,
	cl_mem matBuf, cl_mem buf_diag, int matSize, dtype *Eout)
{
	cl_int err;
	//set kernel arguments
	int argNum = 0;
	err = clSetKernelArg(kern_cholmod_E, argNum++, sizeof(cl_mem), &matBuf);
	err = clSetKernelArg(kern_cholmod_E, argNum++, sizeof(cl_mem), &buf_diag);
	err = clSetKernelArg(kern_cholmod_E, argNum++, sizeof(int), &matSize);

	//执行内核
	size_t global_size[1] = { matSize };
	err = clEnqueueNDRangeKernel(queue, kern_cholmod_E,
		1, NULL,	//1D work space
		global_size,
		NULL,
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFinish(queue);

	if (Eout != NULL)
	{
		err = clEnqueueReadBuffer(queue, buf_diag, CL_TRUE, 0,
			sizeof(dtype)*matSize, Eout, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}
}