
/*
使用mu更新UV的对角线元素
*/
__kernel void kern_update_UV(
	//const int cnp,
	//const int pnp,
	const int nCams,
	const int n3Dpts,
	__global dtype *U,
	__global dtype *V,
	const dtype mu)
{
	int j, rc;
	int idx = get_global_id(0);

	if (idx < nCams*cnp)
	{
		j = (int)(idx / cnp);	//得到对应相机j
		rc = idx - j*cnp;		//得到相机j上的对角元素
		rc = j*cnp*cnp + rc*cnp + rc;
		U[rc] = U[rc] + mu;
	}
	else {
		idx = idx - nCams*cnp;
		j = (int)(idx / pnp);	//得到对应3D点j
		rc = idx - j*pnp;
		rc = j*pnp*pnp + rc*pnp + rc;
		V[rc] = V[rc] + mu;
	}
}