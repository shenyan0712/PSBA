
/*
计算Bga=U*ga+W*gb
*/
__kernel void kern_compute_Bg(		//const int cnp,
	//const int pnp,
	const int nCams,
	const int n3Dpts,
	__global int *blk_idx,
	__global dtype *g,
	__global dtype *U,
	__global dtype *V,
	__global dtype *Wblks,
	__global dtype *Bg)
{ 
	dtype3 *ptr1, *ptr2;
	dtype3 *ptr3, *ptr4;
	int Widx, UVWbase, gbase;
	int r = get_global_id(0);
	private dtype sum = 0.0;
	int WYblksize = cnp*pnp;

	if(r<nCams*cnp) 
	{
		int j = (int)(r / cnp);	//对应的Uj
		int rj = r - j*cnp;		//对应的Uj,W_ij的第rj行， 同时指示Bga_j的第rj个元素

		//***** 计算U_j的第rj行与ga_j的乘积
		UVWbase = j*cnp*cnp + rj*cnp;
		gbase = j*cnp;
		sum = U[UVWbase++] * g[gbase++];
		sum += U[UVWbase++] * g[gbase++];
		sum += U[UVWbase++] * g[gbase++];
		sum += U[UVWbase++] * g[gbase++];
		sum += U[UVWbase++] * g[gbase++];
		sum += U[UVWbase] * g[gbase];

		//*****计算W_ij的第rj行与gb_i的乘积
		for (int i = 0;i < n3Dpts;i++)
		{
			Widx = blk_idx[i*nCams + j];		//W_ij的存储索引
			if (Widx < 0) continue;
			ptr1 = (dtype3 *)&Wblks[Widx*WYblksize + rj*pnp];	//W_ij的第rj行的地址
			ptr2 = (dtype3 *)&g[nCams*cnp + pnp*i];			//gb_i的地址
			sum += dot(*ptr1, *ptr2);
		}
	}
	else {
		int br = r - nCams*cnp;
		int i = (int)br / pnp;	//对应的Vi
		int ri = br - i*pnp;		//对应的Vi的第ri行, (W_ij)^T的ri行（W_ij的ri列）， 同时指示Bgb_i的第ri个元素

		//*****计算(W_ij)^T的第ri列与ga_j的乘积
		for (int j = 0;j < nCams;j++)
		{
			Widx = blk_idx[i*nCams + j];		//W_ij的存储索引
			if (Widx < 0) continue;
			UVWbase = Widx*WYblksize + ri;			//W_ij的(0,ri)
			gbase = j*cnp;
			sum += Wblks[UVWbase] * g[gbase++];
			UVWbase += pnp;							//W_ij的(1,ri)
			sum += Wblks[UVWbase] * g[gbase++];
			UVWbase += pnp;							//W_ij的(2,ri)
			sum += Wblks[UVWbase] * g[gbase++];
			UVWbase += pnp;							//W_ij的(3,ri)
			sum += Wblks[UVWbase] * g[gbase++];
			UVWbase += pnp;							//W_ij的(4,ri)
			sum += Wblks[UVWbase] * g[gbase++];
			UVWbase += pnp;							//W_ij的(5,ri)
			sum += Wblks[UVWbase] * g[gbase];
		}

		//计算Vi的ri行与gb_i的乘积
		UVWbase = i*pnp*pnp + ri*pnp;
		gbase = nCams*cnp + i*pnp;
		sum += V[UVWbase++] * g[gbase++];
		sum += V[UVWbase++] * g[gbase++];
		sum += V[UVWbase] * g[gbase];
	}

	Bg[r] = sum;

}