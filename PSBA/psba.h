#pragma once

#define K_DIM	5
#define dtype double

#define PSBA_INIT_MU		1E-03
#define PSBA_STOP_THRESH	1e-12
#define PSBA_EPSILON       1E-12
#define PSBA_EPSILON2		1E-12
#define PSBA_EPSILON_SQ    ( (PSBA_EPSILON)*(PSBA_EPSILON) )

#define ITER_TURN_TO_LM			1
#define ITER_TURN_TO_TR			2
#define ITER_CONTINUE			3
#define ITER_ERR				4
#define ITER_DP_NO_CHANGE		5		//参数基本不再改变
#define ITER_ERR_SMALL_ENOUGH	6		//误差已足够小
#define ITER_PASS				7		//TR算法用，接受本次迭代
 

#define verbose			0

#define DEBUG_EX		0
#define DEBUG_JAC		0
#define DEBUG_UMAT		0
#define DEBUG_VMAT		0
#define DEBUG_VinvMAT	0
#define DEBUG_WMAT		0
#define DEBUG_YMAT		0
#define DEBUG_SMAT		0
#define DEBUG_G			0
#define DEBUG_SinvMAT	0
#define DEBUG_EAB		0
#define DEBUG_DP		0

#define CL_FILE "e:/sync_directory/workspace/PSBA/CL_files/PSBA.cl"
//#define LM_FILE "e:/sync_directory/workspace/PSBA/CL_files/levmar.cl"



extern FILE *debug_file;




