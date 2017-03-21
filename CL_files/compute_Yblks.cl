

/*
计算Yij块矩阵, Y_ij=W_ij*(Vi^-1),
*/
__kernel void kern_compute_Yblks(
	//const int cnp,
	//const int pnp,
	const int nCams,
	const int n3Dpts,
	__global int *iidx,
	__global dtype *Wblks,
	__global dtype *Vinv,	//只存放在下三角
	__global dtype *Yblks)
{
	int r = get_global_id(0);		//当前工作项处理的Y_ij中的row
	int tc = get_global_id(1);
	int idx = (int)(tc / pnp);		//
	int c = tc - idx*pnp;			//当前工作项处理的Y_ij中的col

									//Yij=Wij*(Vi^-1), 从而Yij中的(r,c)位置就是Wij中的r行点乘Vi^-1中的c列
	int YWbase = idx*cnp*pnp + r*pnp;
	//得到Vi^-1的基索引
	int VinvBase = iidx[idx] * pnp*pnp;

	if (c>0) {		//Yij的后二列元素
		Yblks[YWbase + c] = Wblks[YWbase] * Vinv[VinvBase + c*pnp]		//c列的第一个元素(0,c)使用对角的元素(c,0)。
			+ Wblks[YWbase + 1] * Vinv[VinvBase + c*pnp + 1]				//c列的第一个元素(1,c)使用对角的元素(c,1)。
			+ Wblks[YWbase + 2] * Vinv[VinvBase + 2 * pnp + c];				//最后一列使有原始的元素

																			//Yblks[YWbase + c] = Vinv[VinvBase + c*pnp]+ Vinv[VinvBase + c*pnp + 1]+	 Vinv[VinvBase + 2 * pnp + c];	
	}
	else {
		Yblks[YWbase + c] = Wblks[YWbase] * Vinv[VinvBase]
			+ Wblks[YWbase + 1] * Vinv[VinvBase + 1 * pnp]
			+ Wblks[YWbase + 2] * Vinv[VinvBase + 2 * pnp];
	}

}