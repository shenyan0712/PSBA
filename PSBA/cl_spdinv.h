#pragma once

#include "psba.h"
#include "cl_psba.h"


dtype SPDinv(cl_command_queue queue, cl_kernel kern_cholesky, cl_kernel kern_trigMat_inv, cl_kernel kern_trigMat_mul,
	cl_mem matBuf, cl_mem diagAuxBuf, cl_mem retBuf, int matSize, dtype *outMat);

dtype cholesky(cl_command_queue queue, cl_kernel kern_cholesky,
	cl_mem matBuf, cl_mem diagAuxBuf, cl_mem retBuf,
	int matSize, dtype *outMat);

dtype trigMat_inv(cl_command_queue queue, cl_kernel kern_trigMat_inv,
	cl_mem matBuf, cl_mem auxBuf, cl_mem retBuf, int matSize, dtype *outMat);

void trigMat_mul(cl_command_queue queue, cl_kernel kern_trigMat_mul, cl_mem matBuf,
	cl_mem auxBuf, int matSize, dtype *outMat);