#pragma once

#include "psba.h"
#include "cl_psba.h"


void cholesky_mod(cl_command_queue queue, cl_kernel kern_cholesky_mod, cl_kernel kern_delta_beta,
	cl_mem matBuf, cl_mem auxBuf, cl_mem E_Buf, cl_mem retBuf, int matSize, dtype *outMat);

void cholmod_blk(cl_command_queue queue, cl_kernel kern_cholmod_blk, cl_kernel kern_mat_max,
	cl_mem matBuf, cl_mem blkBackupBuf, cl_mem diagBlkAuxBuf, cl_mem diagBuf, cl_mem retBuf,
	int matSize, dtype *outMat);


void compute_cholmod_E(cl_command_queue queue, cl_kernel kern_cholmod_E,
	cl_mem matBuf, cl_mem buf_diag, int matSize, dtype *Eout);

void get_delta_beta(cl_command_queue queue, cl_kernel kern_delta_beta, cl_mem matBuf, cl_mem auxBuf,
	int matsize, dtype *delta, dtype *beta);