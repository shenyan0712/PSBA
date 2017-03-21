/*
计算U矩阵
结果存放在U中，同时其对角线存放在UVdiag中
*/
__kernel void kern_compute_U(
	//const int cnp,
	const int nCams,
	const int n3Dpts,
	const int n2Dprojs,
	__global dtype *JA,
	__global int *jac_idx,
	__global dtype *U,
	__global dtype *UVdiag,
	const dtype coeff)
{
	int r = get_global_id(0);
	int tc = get_global_id(1);		//在U合成矩阵中的col
	int j = (int)(tc / cnp);			//Uj
	int c = tc - j*cnp;					//Uj中的col

	dtype sum = 0;
	for (int i = 0; i < n3Dpts; i++)
	{
		//计算(Aij)^T*Aij的第r,c个元素, 即Aij的第r列与第c列的点积
		int idx = jac_idx[i*nCams + j];	//先找到对应的Aij块在JA中的索引位置
		if (idx >= 0)
			sum = sum + JA[idx * 2 * cnp + r] * JA[idx * 2 * cnp + c]
			+ JA[idx * 2 * cnp + cnp + r] * JA[idx * 2 * cnp + cnp + c];
	}
	//将sum值存入Uj(r,c)
	sum = coeff*sum;
	U[j*cnp*cnp + r*cnp + c] = sum;
	if (r == c)
		UVdiag[j*cnp + r] = sum;
}