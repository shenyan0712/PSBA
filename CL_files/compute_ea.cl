

/*
计算ea的值，ea=ga-Y*gb
*/
__kernel void kern_compute_ea(
	//const int cnp,
	//const int pnp,
	const int nCams,
	const int n3Dpts,
	const int n2Dprojs,
	__global int *blk_idx,
	__global dtype *Yblks,
	__global dtype *g,
	__global dtype *ea)
{
	int tr = get_global_id(0);		//第r个参数
	int j = (int)(tr / cnp);		//第j个相机
	int r = tr - j*cnp;				//得到Y_*j的r行

	dtype sum = 0;

	//计算Y_0j*gb0,Y_1j*gb1,...,Y_nj*gbn
	int Ybase = 0;
	int gbase = nCams*cnp;
	for (int i = 0; i < n3Dpts; i++) {		//Y是分块矩阵
		int idx = blk_idx[i*nCams + j];		//找到对应的Yij块
		if (idx >= 0) {
			Ybase = idx*cnp*pnp + r*pnp;
			sum = sum + Yblks[Ybase] * g[gbase]
				+ Yblks[Ybase + 1] * g[gbase + 1]
				+ Yblks[Ybase + 2] * g[gbase + 2];
		}
		gbase = gbase + pnp;
	}
	ea[tr] = g[tr] - sum;
}