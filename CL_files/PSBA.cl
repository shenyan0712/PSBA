
#define dtype double
#define dtype3 double3

#define cnp	6
#define pnp 3
#define mnp 2


#include "e:/sync_directory/workspace/PSBA/CL_files/compute_exQT.cl"
#include "e:/sync_directory/workspace/PSBA/CL_files/compute_jacobiQT.cl"
#include "e:/sync_directory/workspace/PSBA/CL_files/compute_U.cl"
#include "e:/sync_directory/workspace/PSBA/CL_files/compute_V.cl"

#include "e:/sync_directory/workspace/PSBA/CL_files/compute_Wblks.cl"
#include "e:/sync_directory/workspace/PSBA/CL_files/compute_g.cl"
#include "e:/sync_directory/workspace/PSBA/CL_files/update_UV.cl"
#include "e:/sync_directory/workspace/PSBA/CL_files/compute_Vinv.cl"
#include "e:/sync_directory/workspace/PSBA/CL_files/restore_UVdiag.cl"

#include "e:/sync_directory/workspace/PSBA/CL_files/compute_Yblks.cl"
#include "e:/sync_directory/workspace/PSBA/CL_files/compute_S.cl"
#include "e:/sync_directory/workspace/PSBA/CL_files/compute_ea.cl"

#include "e:/sync_directory/workspace/PSBA/CL_files/SPD_inv.cl"

#include "e:/sync_directory/workspace/PSBA/CL_files/matVec_mul.cl"
#include "e:/sync_directory/workspace/PSBA/CL_files/compute_eb.cl"
#include "e:/sync_directory/workspace/PSBA/CL_files/compute_dpb.cl"

#include "e:/sync_directory/workspace/PSBA/CL_files/compute_newp.cl"
#include "e:/sync_directory/workspace/PSBA/CL_files/update_p.cl"

/***************/
//for trust-region
#include "e:/sync_directory/workspace/PSBA/CL_files/cholmod_blk.cl"
#include "e:/sync_directory/workspace/PSBA/CL_files/compute_Bg.cl"
#include "e:/sync_directory/workspace/PSBA/CL_files/compute_Jmultiply.cl"
/**************/