

/*
计算矩阵Wblks, W11,W21,..,Wn1,..,Wnm
每个工作项计算W中的一个元素
*/
__kernel void kern_compute_Wblks(
	//const int cnp,
	//const int pnp,
	const int nCams,
	const int n3Dpts,
	const int n2Dprojs,
	__global dtype *JA,
	__global dtype *JB,
	__global int *iidx,
	__global int *jidx,
	__global dtype *Wblks,
	const dtype coeff)
{
	int r = get_global_id(0);		//在W_ij中的row
	int tc = get_global_id(1);
	int idx = (int)(tc / pnp);		//
	int c = tc - idx*pnp;			//在W_ij中的col

	dtype sum = 0;

	//计算Wij的r,c。取Aij的r列与Bij的c列作点积，Aij和Bij共两行
	sum = JA[idx * 2 * cnp + r] * JB[idx * 2 * pnp + c]
		+ JA[idx * 2 * cnp + cnp + r] * JB[idx * 2 * pnp + pnp + c];

	//存放到W中
	sum = coeff*sum;
	Wblks[idx*cnp*pnp + r*pnp + c] = sum;
}