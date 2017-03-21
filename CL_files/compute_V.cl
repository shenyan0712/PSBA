

/*
计算V矩阵
*/
__kernel void kern_compute_V(
	//const int cnp,
	//const int pnp,
	const int nCams,
	const int n3Dpts,
	const int n2Dprojs,
	__global dtype *JB,
	__global int *jac_idx,
	__global dtype *V,
	__global dtype *UVdiag,
	const dtype coeff)
{
	int r = get_global_id(0);
	int tc = get_global_id(1);	//
								//计算得出属于哪个Vi,以及元素在Vi中的列c
	int i = (int)(tc / pnp);	//Vi
	int c = tc - i*pnp;

	dtype sum = 0;
	for (int j = 0; j < nCams; j++)
	{
		//计算(Bij)^T*Bij的第r,c个元素, 即Bij的第r列与第c列的点积
		int idx = jac_idx[i*nCams + j];	//先找到对应的Bij块在JB中的索引位置
		if (idx >= 0)
			sum = sum + JB[idx * 2 * pnp + r] * JB[idx * 2 * pnp + c]
			+ JB[idx * 2 * pnp + pnp + r] * JB[idx * 2 * pnp + pnp + c];
	}
	//将sum值存入V(r,tc)
	sum = coeff*sum;
	V[i*pnp*pnp + r*pnp + c] = sum;
	if (r == c)
		UVdiag[nCams*cnp + i*pnp + r] = sum;
}
