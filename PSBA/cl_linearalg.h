#pragma once


#include "cl_psba.h"
#include "psba.h"


void matVec_mul(cl_command_queue queue, cl_kernel kern_matVec_mul,
	cl_mem M_buffer, cl_mem v_buffer, cl_mem out_buffer, int mat_rsize, int mat_csize, dtype *out);