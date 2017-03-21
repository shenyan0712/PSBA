

/*
J*x
*/
__kernel void kern_compute_Jmultiply(
	const int nCams,
	const int n3Dpts,
	const int n2Dprojs,
	__global dtype *JA,
	__global dtype *JB,
	__global int *blk_idx,
	__global dtype *x,
	__global dtype *out
	)
{ 
	int tr = get_global_id(0);		//结果向量中的第tr个元素
	int idx, addr1,addr2;
	int i, j, k, rblk_size;
	dtype sum = 0;

	rblk_size = mnp*nCams;
	//确定对应的A_ij, B_hij
	i = tr / rblk_size;
	j = (tr - i*rblk_size)/mnp;
	k = tr - i*rblk_size - j*mnp;	//A_ij, B_ij的第k行

	idx = blk_idx[i*nCams + j];
	if(idx>=0) { 
		//A_ij的第k行与x的 j*cnp开始cnp个值的点积
		addr1 = idx*mnp*cnp + k*cnp; addr2 = j*cnp;
		sum += JA[addr1++] * x[addr2++];
		sum += JA[addr1++] * x[addr2++];
		sum += JA[addr1++] * x[addr2++];
		sum += JA[addr1++] * x[addr2++];
		sum += JA[addr1++] * x[addr2++];
		sum += JA[addr1] * x[addr2];

		//B_ij的第k行与x的nCams*cnp + i*pnp开始pnp个值的点积
		addr1 = idx*mnp*pnp + k*pnp; addr2 = nCams*cnp + i*pnp;
		sum += JB[addr1++] * x[addr2++];
		sum += JB[addr1++] * x[addr2++];
		sum += JB[addr1] * x[addr2];
	}


	//out[tr] = i*rblk_size + j*mnp + k;
	out[tr] = sum;



}