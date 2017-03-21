
#include"stdafx.h"
#include <iostream>
#include <fstream>

#include "psba.h"
#include "cl_psba.h"
#include "misc.h"


using namespace std;

/*
配置好openCL库, 填写cl_device结构体
*/
void setup_cl(PSBA_structPtr psbaPtr,
	int cnp,int pnp,int mnp, int nCams,int n3Dpts,int n2Dprojs)
{
	cl_int err;
	psbaPtr->device= get_first_device();
	// 为设备创建上下文
	psbaPtr->context = clCreateContext(NULL, 1, &psbaPtr->device, NULL, NULL, &err);
	checkErr(err, __FILE__, __LINE__);

	// 创建命令对队
	psbaPtr->queue = clCreateCommandQueueWithProperties(psbaPtr->context, psbaPtr->device, 0, &err);
	checkErr(err, __FILE__, __LINE__);

	/* 在设备上创建一个命令队列 */
	cl_queue_properties properties[] = {
		CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT,
		//CL_QUEUE_SIZE,2,
		0 };
	psbaPtr->queue_device = clCreateCommandQueueWithProperties(psbaPtr->context, psbaPtr->device, properties, &err);
	checkErr(err, __FILE__, __LINE__);


	//创建相关缓存
	//##########创建输入buffer###########//
	psbaPtr->Kparas_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,	sizeof(dtype)*K_DIM*nCams, NULL, &err);
	psbaPtr->impts_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE ,sizeof(dtype)*n2Dprojs*mnp, NULL, &err);
	psbaPtr->initcams_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(dtype)*nCams * 4, NULL, &err);
	psbaPtr->cams_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(dtype)*nCams*cnp, NULL, &err);
	checkErr(err, __FILE__, __LINE__);

	psbaPtr->pts3D_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(dtype)*n3Dpts*pnp, NULL, &err);
	psbaPtr->iidx_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(int)*n2Dprojs, NULL, &err);
	psbaPtr->jidx_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(int)*n2Dprojs, NULL, &err);
	psbaPtr->comm3DIdx_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(int)*nCams*nCams*n3Dpts, NULL, &err);
	checkErr(err, __FILE__, __LINE__);

	psbaPtr->comm3DIdxCnt_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(int)*nCams*nCams, NULL, &err);
	psbaPtr->blkIdx_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(int)*nCams*n3Dpts, NULL, &err);
	psbaPtr->ex_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(dtype)*n2Dprojs*mnp, NULL, &err);
	psbaPtr->JA_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(dtype)*n2Dprojs*mnp*cnp, NULL, &err);
	checkErr(err, __FILE__, __LINE__);

	psbaPtr->JB_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE, sizeof(dtype)*n2Dprojs*mnp*pnp, NULL, &err);
	psbaPtr->Jmul_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE, sizeof(dtype)*(nCams*n3Dpts)*mnp, NULL, &err);
	psbaPtr->U_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,	sizeof(dtype)*nCams*cnp*cnp, NULL, &err);
	psbaPtr->V_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(dtype)*n3Dpts*pnp*pnp, NULL, &err);
	//psbaPtr->Vinv_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE, sizeof(dtype)*n3Dpts*pnp*pnp, NULL, &err);
	psbaPtr->UVdiag_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(dtype)*(n3Dpts*pnp+nCams*cnp), NULL, &err);
	checkErr(err, __FILE__, __LINE__);

	psbaPtr->W_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,	sizeof(dtype)*n2Dprojs*cnp*pnp, NULL, &err);
	psbaPtr->Y_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(dtype)*n2Dprojs*cnp*pnp, NULL, &err);
	psbaPtr->S_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,	sizeof(dtype)*nCams*cnp*nCams*cnp, NULL, &err);
	psbaPtr->Saux_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE, sizeof(dtype)*nCams*cnp*nCams*cnp, NULL, &err);
	psbaPtr->diagBlkAux_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE, sizeof(dtype) * 3 * (6 * nCams), NULL, &err);
	psbaPtr->blkBackup_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE, sizeof(dtype) * 3 * (6 * nCams), NULL, &err);
	psbaPtr->E_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE, sizeof(dtype) * cnp * nCams, NULL, &err);
	checkErr(err, __FILE__, __LINE__);

	psbaPtr->dp_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(dtype)*(cnp*nCams + n3Dpts*pnp), NULL, &err);
	//psbaPtr->Q0_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE, sizeof(dtype)*nCams, NULL, &err);
	psbaPtr->g_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(dtype)*(cnp*nCams + n3Dpts*pnp), NULL, &err);
	psbaPtr->Bg_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE, sizeof(dtype)*(cnp*nCams + n3Dpts*pnp), NULL, &err);
	checkErr(err, __FILE__, __LINE__);

	psbaPtr->eab_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(dtype)*(cnp*nCams+ n3Dpts*pnp), NULL, &err);
	psbaPtr->newCams_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(dtype)*(cnp*nCams), NULL, &err);
	psbaPtr->newPts3D_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(dtype)*(n3Dpts*pnp), NULL, &err);
	psbaPtr->ret_buffer = clCreateBuffer(psbaPtr->context, CL_MEM_READ_WRITE,sizeof(dtype), NULL, &err);
	checkErr(err, __FILE__, __LINE__);

	/********************************************************************************/
	/********************************************************************************/
	//
	psbaPtr->program_new = build_program(psbaPtr->context, psbaPtr->device, CL_FILE);
	checkErr(err, __FILE__, __LINE__);

	psbaPtr->kern_compute_exQT	= clCreateKernel(psbaPtr->program_new, "kern_compute_exQT", &err);
	psbaPtr->kern_compute_jacobiQT = clCreateKernel(psbaPtr->program_new, "kern_compute_jacobiQT", &err);
	psbaPtr->kern_compute_U = clCreateKernel(psbaPtr->program_new, "kern_compute_U", &err);
	psbaPtr->kern_compute_V = clCreateKernel(psbaPtr->program_new, "kern_compute_V", &err);

	psbaPtr->kern_compute_Wblks = clCreateKernel(psbaPtr->program_new, "kern_compute_Wblks", &err);
	psbaPtr->kern_compute_g = clCreateKernel(psbaPtr->program_new, "kern_compute_g", &err);
	psbaPtr->kern_update_UV = clCreateKernel(psbaPtr->program_new, "kern_update_UV", &err);
	psbaPtr->kern_restore_UVdiag = clCreateKernel(psbaPtr->program_new, "kern_restore_UVdiag", &err);
	psbaPtr->kern_compute_Vinv = clCreateKernel(psbaPtr->program_new, "kern_compute_Vinv", &err);
	checkErr(err, __FILE__, __LINE__);

	psbaPtr->kern_compute_Yblks = clCreateKernel(psbaPtr->program_new, "kern_compute_Yblks", &err);
	psbaPtr->kern_compute_S = clCreateKernel(psbaPtr->program_new, "kern_compute_S", &err);
	psbaPtr->kern_compute_ea = clCreateKernel(psbaPtr->program_new, "kern_compute_ea", &err);

	psbaPtr->kern_cholesky = clCreateKernel(psbaPtr->program_new, "kern_cholesky", &err);				
	psbaPtr->kern_trigMat_inv = clCreateKernel(psbaPtr->program_new, "kern_trigMat_inv", &err);
	psbaPtr->kern_trigMat_mul = clCreateKernel(psbaPtr->program_new, "kern_trigMat_mul", &err);
	checkErr(err, __FILE__, __LINE__);

	psbaPtr->kern_matVec_mul = clCreateKernel(psbaPtr->program_new, "kern_matVec_mul", &err);
	psbaPtr->kern_compute_eb = clCreateKernel(psbaPtr->program_new, "kern_compute_eb", &err);
	psbaPtr->kern_compute_dpb = clCreateKernel(psbaPtr->program_new, "kern_compute_dpb", &err);

	psbaPtr->kern_compute_newp = clCreateKernel(psbaPtr->program_new, "kern_compute_newp", &err);
	psbaPtr->kern_update_p = clCreateKernel(psbaPtr->program_new, "kern_update_p", &err);
	checkErr(err, __FILE__, __LINE__);

	/************************************************/
	//Trust-region算法用
	psbaPtr->kern_compute_Jmultiply = clCreateKernel(psbaPtr->program_new, "kern_compute_Jmultiply", &err);
	psbaPtr->kern_compute_Bg = clCreateKernel(psbaPtr->program_new, "kern_compute_Bg", &err);
	//checkErr(err, __FILE__, __LINE__);
	psbaPtr->kern_cholmod_blk = clCreateKernel(psbaPtr->program_new, "kern_cholmod_blk", &err);
	psbaPtr->kern_mat_max = clCreateKernel(psbaPtr->program_new, "kern_mat_max", &err);
	psbaPtr->kern_cholmod_E = clCreateKernel(psbaPtr->program_new, "kern_cholmod_E", &err);
	checkErr(err, __FILE__, __LINE__);

	/***********************************************/
}

/*

*/
void fill_initBuffer2(PSBA_structPtr psbaPtr,
	int cnp, int pnp, int mnp, int nCams, int n3Dpts, int n2Dprojs,
	dtype *Kparas,
	dtype *impts_data,		//2D图像点数据
	dtype *initcams_data,	//初始的相机旋转四元数(4-vec)
	dtype *camsExParas,		//相机参数(9+3, R,t)+
	dtype *pts3Ds			//3D点坐标(X,Y,Z)
)
{
	cl_int err;
	err = clEnqueueWriteBuffer(psbaPtr->queue, psbaPtr->Kparas_buffer, CL_TRUE, 0,
		sizeof(dtype)*K_DIM*nCams,
		static_cast<void *>(Kparas), 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	err = clEnqueueWriteBuffer(psbaPtr->queue, psbaPtr->impts_buffer, CL_TRUE, 0,
		sizeof(dtype)*n2Dprojs*mnp,
		static_cast<void *>(impts_data), 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	err = clEnqueueWriteBuffer(psbaPtr->queue, psbaPtr->initcams_buffer, CL_TRUE, 0,
		sizeof(dtype)*nCams * 4,
		static_cast<void *>(initcams_data), 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	err = clEnqueueWriteBuffer(psbaPtr->queue, psbaPtr->cams_buffer, CL_TRUE, 0,
		sizeof(dtype)*nCams*cnp,
		static_cast<void *>(camsExParas), 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	err = clEnqueueWriteBuffer(psbaPtr->queue, psbaPtr->pts3D_buffer, CL_TRUE, 0,
		sizeof(dtype)*n3Dpts*pnp,
		static_cast<void *>(pts3Ds), 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);
}



void fill_idxBuffer(PSBA_structPtr psbaPtr,
	int nCams, int n3Dpts, int n2Dprojs,
	int *comm3DIdx, int *comm3DIdxCnt,
	int *iidx, int *jidx, int *blk_idx)
{
	cl_int err;
	err=clEnqueueWriteBuffer(psbaPtr->queue, psbaPtr->iidx_buffer, CL_TRUE, 0,
		sizeof(int)*n2Dprojs,
		static_cast<void *>(iidx), 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	err=clEnqueueWriteBuffer(psbaPtr->queue, psbaPtr->jidx_buffer, CL_TRUE, 0,
		sizeof(int)*n2Dprojs,
		static_cast<void *>(jidx), 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	err=clEnqueueWriteBuffer(psbaPtr->queue, psbaPtr->blkIdx_buffer, CL_TRUE, 0,
		sizeof(int)*nCams*n3Dpts,
		static_cast<void *>(blk_idx), 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	err = clEnqueueWriteBuffer(psbaPtr->queue, psbaPtr->comm3DIdx_buffer, CL_TRUE, 0,
		sizeof(int)*nCams*nCams*n3Dpts,
		static_cast<void *>(comm3DIdx), 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

	err = clEnqueueWriteBuffer(psbaPtr->queue, psbaPtr->comm3DIdxCnt_buffer, CL_TRUE, 0,
		sizeof(int)*nCams*nCams,
		static_cast<void *>(comm3DIdxCnt), 0, NULL, NULL);
	checkErr(err, __FILE__, __LINE__);

}


void release_buffer(PSBA_structPtr psbaPtr)
{
	clReleaseMemObject(psbaPtr->Kparas_buffer);
	clReleaseMemObject(psbaPtr->impts_buffer);
	clReleaseMemObject(psbaPtr->initcams_buffer);
	clReleaseMemObject(psbaPtr->cams_buffer);
	clReleaseMemObject(psbaPtr->pts3D_buffer);
	clReleaseMemObject(psbaPtr->vmask_buffer);
	clReleaseMemObject(psbaPtr->iidx_buffer);
	clReleaseMemObject(psbaPtr->jidx_buffer);
	clReleaseMemObject(psbaPtr->ex_buffer);
	clReleaseMemObject(psbaPtr->blkIdx_buffer);
	clReleaseMemObject(psbaPtr->JA_buffer);
	clReleaseMemObject(psbaPtr->JB_buffer);
	clReleaseMemObject(psbaPtr->U_buffer);
	clReleaseMemObject(psbaPtr->V_buffer);
	clReleaseMemObject(psbaPtr->UVdiag_buffer);
	clReleaseMemObject(psbaPtr->W_buffer);
	clReleaseMemObject(psbaPtr->Y_buffer);
	clReleaseMemObject(psbaPtr->S_buffer);
	clReleaseMemObject(psbaPtr->diagBlkAux_buffer);
	clReleaseMemObject(psbaPtr->blkBackup_buffer);
	clReleaseMemObject(psbaPtr->Saux_buffer);
	clReleaseMemObject(psbaPtr->g_buffer);
	clReleaseMemObject(psbaPtr->dp_buffer);
	clReleaseMemObject(psbaPtr->eab_buffer);
	clReleaseMemObject(psbaPtr->newCams_buffer);
	clReleaseMemObject(psbaPtr->newPts3D_buffer);

	clReleaseMemObject(psbaPtr->ret_buffer);

}



/** 读取文件并将其转为字符串 */
int convertToString(const char *filename, std::string& s)
{
	size_t size;
	char*  str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));
	if (f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size + 1];
		if (!str)
		{
			f.close();
			return 0;
		}
		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	cout << "Error: failed to open file\n:" << filename << endl;
	return -1;
}


inline void checkErr(cl_int err, const char* file, int num)
{
	if (CL_SUCCESS != err)
	{
		printf("OpenCL error(%d) at file %s(%d).\n", err, file, num - 1);
		system("Pause");
		exit(EXIT_FAILURE);
	}
}

/* 找到第一个平台，返回第一个设备ID  */
cl_device_id get_first_device() {

	cl_uint numPlatforms;
	cl_platform_id *platformIDs;
	cl_device_id dev;
	int err;

	//获取平台的个数
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr(err, __FILE__, __LINE__);

	//根据个数创建cl_platform_id对象
	platformIDs = (cl_platform_id *)malloc(
		sizeof(cl_platform_id) * numPlatforms);

	//获取平台ID
	err = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
	checkErr(err, __FILE__, __LINE__);

	/* 获取第一个设备的ID */
	err = clGetDeviceIDs(platformIDs[0], CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	if (err == CL_DEVICE_NOT_FOUND) {
		err = clGetDeviceIDs(platformIDs[1], CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	checkErr(err, __FILE__, __LINE__);
	return dev;
}


/* 创建并编建程序对象 */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

	cl_program program;
	char *program_log;
	size_t program_size, log_size;
	int err;

	/* 读取程序文件 */
	string sourceStr;
	err = convertToString(filename, sourceStr);
	const char *program_str = sourceStr.c_str();
	program_size = strlen(program_str);

	/* 从程序文件创建程序对象 */
	program = clCreateProgramWithSource(ctx, 1,
		(const char**)&program_str, &program_size, &err);
	checkErr(err, __FILE__, __LINE__);

	/* 编程程序 */
	err = clBuildProgram(program, 1, &dev, "-cl-std=CL2.0 -D CL_VERSION_2_0", NULL, NULL);
	if (err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		system("pause");
		exit(1);
	}
	return program;
}


/*
*/
void printBuf2D(FILE *file, cl_command_queue queue, cl_mem buf, int offset, int rsize, int csize, char *title)
{
	dtype *ptr;
	int bufSize = sizeof(dtype)*rsize*csize;
	ptr =(dtype*) emalloc(bufSize);

	clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, bufSize, ptr, 0, NULL, NULL);

	fprintf(file,"%s:\n", title);
	for (int i = 0;i < rsize;i++)
	{
		for (int j = 0;j < csize;j++)
		{
			if(j==(csize-1))
				fprintf(file,"%le  ", ptr[i*csize + j]);
			else
				fprintf(file, "%le,  ", ptr[i*csize + j]);
		}
		fprintf(file,"\n");
	}
	fflush(debug_file);
	free(ptr);
}


/*
*/
void printBuf2D_blk(FILE *file, cl_command_queue queue, cl_mem buf, int offset, int rsize, int csize, char *title)
{
	dtype *ptr;
	int bufSize = sizeof(dtype)*rsize*csize;
	ptr = (dtype*)emalloc(bufSize);

	clEnqueueReadBuffer(queue, buf, CL_TRUE, offset*sizeof(dtype), bufSize, ptr, 0, NULL, NULL);

	fprintf(file, "%s:\n", title);
	for (int i = 0; i < rsize; i++)
	{
		for (int j = 0; j < csize; j++)
		{
			if (j == (csize - 1))
				fprintf(file, "%.15E  ", ptr[i*csize + j]);
			else
				fprintf(file, "%.15E,  ", ptr[i*csize + j]);
			if (isnan(ptr[i])) {
				printf("\n[%d,%d]there is a NaN.\n", i,j);
				break;
			}
		}
		fprintf(file, "\n");
	}
	fflush(debug_file);
	free(ptr);
}

void printBuf1D(FILE *file,cl_command_queue queue, cl_mem buf,int offset, int size, char *title)
{
	dtype *ptr;
	int bufSize = sizeof(dtype)*size;
	ptr = (dtype*)emalloc(bufSize);

	clEnqueueReadBuffer(queue, buf, CL_TRUE, offset*sizeof(dtype), bufSize, ptr, 0, NULL, NULL);

	fprintf(file,"%s:\n", title);
	for (int i = 0;i < size;i++)
	{
		fprintf(file,"%le,  ", ptr[i]);
		if (isnan(ptr[i])) {
			printf("\n[%d]there is a NaN.\n",i);
			break;
		}
	}
	fprintf(file,"\n");

	free(ptr);
}