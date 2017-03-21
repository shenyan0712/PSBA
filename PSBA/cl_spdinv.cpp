/*

本文件内容包含求解SPD矩阵的求逆所需的cholesky分解和三角矩阵求逆

*/


#include "stdafx.h"

#include <time.h>

#include "cl_psba.h"
#include "psba.h"
#include "cl_spdinv.h"



dtype SPDinv(cl_command_queue queue, cl_kernel kern_cholesky,cl_kernel kern_trigMat_inv,cl_kernel kern_trigMat_mul,
	cl_mem matBuf, cl_mem diagAuxBuf, cl_mem retBuf,int matSize, dtype *outMat)
{
	cl_int err;
	dtype ret = 0.0;

	//先对S进行备份
	//clEnqueueCopyBuffer(queue, S_buffer, psbaPtr->Saux_buffer, 0, 0,
	//	sizeof(dtype)*matSize*matSize,
	//	0, NULL, NULL);

	//对S进行分解。
	ret = cholesky(queue, kern_cholesky,
		matBuf, diagAuxBuf, retBuf, matSize, outMat);

	if (ret != 0.0) return ret;
	//得到下三角矩阵的逆矩阵,结果放在Saux_buffer的上三角中。
	ret = trigMat_inv(queue, kern_trigMat_inv, matBuf, diagAuxBuf, retBuf,
		matSize, outMat);

	trigMat_mul(queue, kern_trigMat_mul, matBuf, diagAuxBuf, matSize, outMat);

}


/******************************************************************************************************/

/*
cholesky分解
特性：
分块并行方法，enqueue on device
输入：
matBuf--待分解方阵, matSize*matSize
diagAuxBuf--辅助Buffer, >=3*matSize
输出：
matBuf--其下三角存放分解结果
diagAuxBuf--存放Ljj^-1子块
outMat!=NULL 则将分解结果拷贝到outMat
*/
dtype cholesky(cl_command_queue queue, cl_kernel kern_cholesky,
	cl_mem matBuf, cl_mem diagAuxBuf, cl_mem retBuf,
	int matSize, dtype *outMat)
{
	cl_int err;
	dtype ret;

	//set kernel arguments
	err = clSetKernelArg(kern_cholesky, 0, sizeof(cl_mem), &matBuf);
	err = clSetKernelArg(kern_cholesky, 1, sizeof(cl_mem), &diagAuxBuf);
	err = clSetKernelArg(kern_cholesky, 2, sizeof(cl_mem), &retBuf);
	err = clSetKernelArg(kern_cholesky, 3, sizeof(dtype) * 9, NULL);		//T_ii块
	err = clSetKernelArg(kern_cholesky, 4, sizeof(dtype) * 6, NULL);		//L_ii,只存储下三角
	err = clSetKernelArg(kern_cholesky, 5, sizeof(int), &matSize);

	//执行内核
	int j = 0;
	size_t global_size[2] = { matSize,3 };
	size_t local_size[2] = { 3,3 };
	err = clSetKernelArg(kern_cholesky, 6, sizeof(int), &j);		//块矩阵的第j列
	err = clEnqueueNDRangeKernel(queue, kern_cholesky,
		2, NULL,	//2D work space
		global_size,
		local_size,
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFinish(queue);

	err = clEnqueueReadBuffer(queue, retBuf, CL_TRUE, 0,
		sizeof(ret), &ret, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	// 读取结果
	if (outMat != NULL)
	{
		err = clEnqueueReadBuffer(queue, matBuf, CL_TRUE, 0,
			sizeof(dtype)*matSize*matSize, outMat, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}

#if DEBUG_CHOLESKY==1
	printBuf2D(stdout, queue, matBuf, matSize, matSize, "L(cholesky):");
	printBuf2D(stdout, queue, diagAuxBuf, 3, matSize, "diagInv:");

#endif
	return ret;
}




/*
求得下三角矩阵的逆矩阵
特性：
使用3x3的块，
使用opencl 2.0的QUEUE_ON_DEVICE特性
输入：
matBuf为下三角矩阵
auxBuf为3x3对角块的逆矩阵
输出：
返回值：=0表示求逆正确
结果存放buf_spd_A的上三角（包括对角块），下三角不做处理。
*/
dtype trigMat_inv(cl_command_queue queue, cl_kernel kern_trigMat_inv, 
	cl_mem matBuf, cl_mem auxBuf,cl_mem retBuf, int mat_size, dtype *outMat)
{
	cl_command_queue queue_device;
	cl_int err;
	dtype ret = 0.0;
	//int blk_mat_size;

	//********配置参数**********/
	int j = 0;
	err = clSetKernelArg(kern_trigMat_inv, 0, sizeof(cl_mem), &matBuf);
	err |= clSetKernelArg(kern_trigMat_inv, 1, sizeof(cl_mem), &auxBuf);
	err |= clSetKernelArg(kern_trigMat_inv, 2, sizeof(cl_mem), &retBuf);
	err |= clSetKernelArg(kern_trigMat_inv, 3, sizeof(int), &mat_size);
	err |= clSetKernelArg(kern_trigMat_inv, 4, sizeof(int), &j);

	//执行内核,
	size_t global_size[2] = { mat_size,3 };
	size_t local_size[2] = { 3,3 };
	err = clEnqueueNDRangeKernel(queue, kern_trigMat_inv,
		2, NULL,	//2D work space
		global_size,
		local_size,	//local_size,
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFinish(queue);

	err = clEnqueueReadBuffer(queue, retBuf, CL_TRUE, 0,
		sizeof(ret), &ret, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	if (outMat != NULL)
	{
		err = clEnqueueReadBuffer(queue, matBuf, CL_TRUE, 0,
			sizeof(dtype)*mat_size*mat_size, outMat, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}

#if DEBUG_TRIGMAT_INV==1
	printBuf2D(stdout, queue, matBuf, mat_size, mat_size, "Linv:");
#endif
	return ret;
}

/*
计算M_inv=(L^-t)*L^-1
buf_spd_A的上三角已经是L^-t

*/
void trigMat_mul(cl_command_queue queue, cl_kernel kern_trigMat_mul, cl_mem matBuf, 
	cl_mem auxBuf,int mat_size, dtype *outMat)
{
	cl_command_queue queue_device;
	cl_int err;
	dtype ret = 0.0;

	//********配置参数**********/
	int j = 0;
	err = clSetKernelArg(kern_trigMat_mul, 0, sizeof(cl_mem), &matBuf);
	err |= clSetKernelArg(kern_trigMat_mul, 1, sizeof(cl_mem), &auxBuf);
	err |= clSetKernelArg(kern_trigMat_mul, 2, sizeof(int), &mat_size);

	//执行内核,
	size_t global_size[2] = { mat_size,mat_size };
	//size_t local_size[2] = { 3,3 };
	err = clEnqueueNDRangeKernel(queue, kern_trigMat_mul,
		2, NULL,	//1D work space
		global_size,
		NULL,		//local_size,
		0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
	clFinish(queue);

	if (outMat != NULL)
	{
		err = clEnqueueReadBuffer(queue, matBuf, CL_TRUE, 0,
			sizeof(dtype)*mat_size*mat_size, outMat, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);
	}

#if DEBUG_TRIGMAT_INV==1
	printBuf2D(stdout, queue, matBuf, mat_size, mat_size, "Inverse:");

#endif
}