

/*
计算eb，eb=gb-(W^t)*dpa
*/
__kernel void kern_compute_eb(
	//const int cnp,
	//const int pnp,
	const int nCams,
	const int n3Dpts,
	const int n2Dprojs,
	__global int *blk_idx,
	__global dtype *Wblks,
	__global dtype *dp,		//使用dpa,不需要偏移
	__global dtype *g,		//使用gb,需要ga的偏移量
	__global dtype *eab)
{
	int tr = get_global_id(0);		//第r个参数
	int i = (int)(tr / pnp);		//第i个3D点
	int c = tr - i*pnp;				//得到W_i*的c列。共pnp个元素

	dtype sum = 0;

	//计算Y_0j*gb0,Y_1j*gb1,...,Y_nj*gbn
	int Wbase = 0;
	int dpBase = 0;
	int gbBase = nCams*cnp;
	for (int j = 0; j < nCams; j++) {		//W是分块矩阵
		int idx = blk_idx[i*nCams + j];		//找到对应的Wij块
		if (idx >= 0) {
			Wbase = idx*cnp*pnp;
			for (int k = 0; k < cnp; k++) {		//W_ij的c列与dpa_j的点积
				sum = sum + Wblks[Wbase + k*pnp + c] * dp[dpBase + k];
			}
		}
		dpBase = dpBase + cnp;
	}

	eab[gbBase + tr] = g[gbBase + tr] - sum;
	//gb[tr] = i;
}