
/*
计算dpb, dpb=(V^-1)*gb, Vinv只存储在下三角

*/
__kernel void kern_compute_dpb(
	//const int cnp,
	//const int pnp,
	const int nCams,
	const int n3Dpts,
	__global dtype *Vinv,
	__global dtype *eab,		//使用eb
	__global dtype *dp)		//使用dpb
{
	int tr = get_global_id(0);		//第tr个参数
	int i = (int)(tr / pnp);		//第i个3D点
	int r = tr - i*pnp;				//得到Vinv_ii的r行

	dtype sum = 0;

	int Vinv_base = i*pnp*pnp;
	int gbase = nCams*cnp + i*pnp;
	if (r < 2) {	//前二行
		sum = Vinv[Vinv_base + r*pnp] * eab[gbase]					//第一个元素(r,0)不变
			+ Vinv[Vinv_base + pnp + r] * eab[gbase + 1]		//第二个元素(r,1)使用对角元素(1,r)
			+ Vinv[Vinv_base + 2 * pnp + r] * eab[gbase + 2];	//第三个元素(r,2)使用对角元素(2,r)
	}
	else {
		sum = Vinv[Vinv_base + r*pnp] * eab[gbase]
			+ Vinv[Vinv_base + r*pnp + 1] * eab[gbase + 1]
			+ Vinv[Vinv_base + r*pnp + 2] * eab[gbase + 2];
	}

	dp[nCams*cnp + tr] = sum;
}