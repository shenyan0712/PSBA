#include "stdafx.h"

#include <time.h>

#include "cl_psba.h"
#include "psba.h"
#include "cl_linearalg.h"
#include "misc.h"

extern int itno;
extern FILE *debug_file;


/*
计算Mv,矩阵与向量的乘积，结果存到out中。
rmat_size --矩阵行尺寸
cmat_size --矩阵列尺寸
*/
void matVec_mul(cl_command_queue queue, cl_kernel kern_matVec_mul, 
	cl_mem M_buffer,cl_mem v_buffer, cl_mem out_buffer, int mat_rsize, int mat_csize, dtype *out)
{
	cl_int err;

	//执行内核
	//********按列计算下三角矩阵的逆**********/
	err = clSetKernelArg(kern_matVec_mul, 0, sizeof(int), &mat_csize);
	err |= clSetKernelArg(kern_matVec_mul, 1, sizeof(cl_mem), &M_buffer);
	err |= clSetKernelArg(kern_matVec_mul, 2, sizeof(cl_mem), &v_buffer);
	err |= clSetKernelArg(kern_matVec_mul, 3, sizeof(cl_mem), &out_buffer);

	size_t global_size[1] = {(size_t) mat_rsize };		//这个应设为矩阵的行数，以便可用于非方阵的情况
	err = clEnqueueNDRangeKernel(queue, kern_matVec_mul,
		1, NULL,	//工作项使用二维索引空间
		global_size,
		NULL,	//让OpenCL自行
		0, NULL, NULL);
	err = clFinish(queue);
	checkErr(err, __FILE__, __LINE__);

	if (out != NULL)
	{
		err = clEnqueueReadBuffer(queue, out_buffer, CL_TRUE, 0,
			sizeof(dtype)*mat_rsize, out, 0, NULL, NULL);
		checkErr(err, __FILE__, __LINE__);

	}

#if DEBUG_DP==1
	// 读取结果
	//err = clEnqueueReadBuffer(psbaPtr->queue, out_buffer, CL_TRUE, 0,
	//	sizeof(dtype)*rmat_size, out, 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
#endif

}
