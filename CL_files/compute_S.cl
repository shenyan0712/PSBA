

/*
计算S矩阵 U-YW^t
*/
__kernel void kern_compute_S(
	//const int cnpp,
	//const int pnp,
	const int nCams,
	const int n3Dpts,
	__global int *blk_idx,
	__global dtype *U,
	__global dtype *Yblks,
	__global dtype *Wblks,
	__global int *comm3DIdx,
	__global int *comm3DIdxCnt,
	__global dtype *S)
{
	int idx;
	int Yidx, Widx, Ybase, Wbase;
	int tr = get_global_id(0);		//
	int tc = get_global_id(1);		//
	//int cnp = cnpp;
	//将r,c变到对应的Ui
	int k = (int)(tr / cnp);		//S中S_kl块所在k行
	int l = (int)(tc / cnp);		//S中S_kl块所在l列
	int r = tr - k*cnp;				//本工作项处理的S_kl块中的元素的r行号
	int c = tc - l*cnp;				//本工作项处理的S_kl块中的元素的c行号

	//S_kl的(r,c)元素是sum(Y_ik的r行和W_il的c行的点积) for all i=1,..,n3Dpts
	dtype sum = 0;
	dtype3 *ptr1, *ptr2;

	int WYblksize = cnp*pnp;
	int comm3DCnt = comm3DIdxCnt[k*nCams + l];
	int comm3Dbase = k*nCams*n3Dpts + l*n3Dpts;

	///*
	for (int i = 0; i <comm3DCnt; i++)
	{
		idx = comm3DIdx[comm3Dbase + i];		//得到3D点的索引

		Yidx = blk_idx[idx*nCams + k];
		Widx = blk_idx[idx*nCams + l];
		Ybase = Yidx*WYblksize + r*pnp;
		Wbase = Widx*WYblksize + c*pnp;

		//向量方式
		ptr1 = (dtype3 *)&Yblks[Ybase];
		ptr2 = (dtype3 *)&Wblks[Wbase];
		sum += dot(*ptr1, *ptr2);
	}
	//*/

	/*
	for (int i = 0; i < n3Dpts; i++)
	{
		//得到Yik索引
		//得到Wil索引
		Yidx = blk_idx[i*nCams + k];
		Widx = blk_idx[i*nCams + l];
		if (Yidx < 0 || Widx < 0)
			continue;
		Ybase = Yidx*WYblksize + r*pnp;
		Wbase = Widx*WYblksize + c*pnp;
		ptr1 = (dtype3 *)&Yblks[Ybase];
		ptr2 = (dtype3 *)&Wblks[Wbase];
		sum += dot(*ptr1, *ptr2);
	}
	*/

	//如果是对角块，Uk-Y_ik*W_il^t
	if (k == l)
		sum = U[k*cnp*cnp + r*cnp + c] - sum;
	else
		sum = -sum;
	S[tr*nCams*cnp + tc] = sum;
}