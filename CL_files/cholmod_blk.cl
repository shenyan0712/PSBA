
__kernel void kern_cholmod_blk_step2(
	__global dtype* mat,
	__global dtype* aux,
	__global dtype* diagInv,
	__global dtype* diag,
	__global dtype* ret,
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j);
__kernel void kern_cholmod_blk_step3(
	__global dtype* mat,
	__global dtype* aux,
	__global dtype* diagInv,
	__global dtype* diag,
	__global dtype* ret,
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j);

__kernel void kern_cholmod_step1(
	__global dtype* mat,
	__global dtype* aux,
	__global dtype* diagInv,
	__global dtype* diag,
	__global dtype* ret,
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j,
	const int col);
__kernel void kern_cholmod_step2(
	__global dtype* mat,
	__global dtype* aux,
	__global dtype* diagInv,
	__global dtype* diag,
	__global dtype* ret,
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j,
	const int col);
__kernel void kern_cholmod_step3(
	__global dtype* mat,
	__global dtype* aux,
	__global dtype* diagInv,
	__global dtype* diag,
	__global dtype* ret,
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j,
	const int col);
__kernel void kern_cholmod_step4(
	__global dtype* mat,
	__global dtype* aux,
	__global dtype* diagInv,
	__global dtype* diag,
	__global dtype* ret,
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j,
	const int col);
__kernel void kern_cholmod_diaginv(
	__global dtype* mat,
	__global dtype* aux,
	__global dtype* diagInv,
	__global dtype* diag,
	__global dtype* ret,
	__local dtype *L,
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j);


/*
modified cholesky分解
step 1: 计算Ljj=Ajj-sum(Ljk*Ljk^t)中的uv元素，以及处理对角块
j为块矩阵的当前列
aux ---尺寸>=3*matsize, 第一组matsize个元素存放d_j,第二组matsize个元素存放当前列的C_ij。还用做列块的备份

*/
__kernel void kern_cholmod_blk(
	__global dtype *mat,		//原始矩阵 和 结果矩阵
	__global dtype *aux,		//用于备份
	__global dtype *diagInv,	//输出的Lii^-1, 可供三角矩阵求逆使用
	__global dtype *diag,		//保存原始矩阵的对角线元素
	__global dtype *ret,
	__local dtype *T,
	__local dtype *L,			//3x3
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j)
{
	dtype3 *ptr1, *ptr2;
	dtype sum, t1, t2, t3, t4, t5;
	int u = get_local_id(0);
	int v = get_local_id(1);

	int backup_addr = j * 3 * 3 + u * 3 + v;
	int jjuv_addr = (j * 3 + u)*mat_size + j * 3 + v;

	if (get_global_id(0) == 0 && get_global_id(1) == 0) *ret = 0.0;

	//计算Tjj = Ajj - sum(Ljk*Ljk^t)中的uv元素
	sum = mat[jjuv_addr];	//Aij_uv元素
							//备份Ajj, 并将对角线元素保存在diag中。
	aux[backup_addr] = sum;
	if (u == v) diag[j * 3 + u] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步

	for (int k = 0; k < j; k++)
	{
		int Ljku_base = (j * 3 + u)*mat_size + k * 3;		//Ljk的第u行
		int Ljkv_base = (j * 3 + v)*mat_size + k * 3;		//Ljk的第v行
															//sum -= mat[Lik_base++] * mat[Ljk_base++];
															//sum -= mat[Lik_base++] * mat[Ljk_base++];
															//sum -= mat[Lik_base] * mat[Ljk_base];
		ptr1 = (dtype3 *)&mat[Ljku_base];
		ptr2 = (dtype3 *)&mat[Ljkv_base];
		sum -= dot(*ptr1, *ptr2);
	}
	//mat[jjuv_addr] = sum;		//保存Tij_uv到原矩阵
	T[u * 3 + v] = sum;		//同时保存到本地缓存，对角线块使用
	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步

										//计算Ljj, 从Ljj*Ljj^t=Tjj得来
	switch (u * 3 + v)
	{
	case 0:		//L00 or L(0)
		L[0] = T[0];
		if (!isfinite(L[0]) || L[0] <= 0) {
			*ret = 1.0; break;
		}
		L[0] = sqrt(L[0]);
		mat[jjuv_addr] = L[0];
		break;
	case 1:		//L01,L02
	case 2:
		mat[jjuv_addr] = 0;
		break;
	case 3:		//L10
		L[1] = T[1 * 3 + 0] / sqrt(T[0]);
		mat[jjuv_addr] = L[1];
		if (!isfinite(L[1])) {
			*ret = 1.0; break;
		}
		break;
	case 4:		//L11
		L[2] = T[1 * 3 + 1] - T[1 * 3 + 0] * T[1 * 3 + 0] / T[0];
		if (!isfinite(L[2]) || L[2] <= 0) {
			*ret = 1.0; break;
		}
		L[2] = sqrt(L[2]);
		mat[jjuv_addr] = L[2];
		break;
	case 5:		//L12
		mat[jjuv_addr] = 0;
		break;
	case 6:		//L20 or L(3)
		L[3] = T[2 * 3 + 0] / sqrt(T[0]);
		if (!isfinite(L[3])) {
			*ret = 1.0; break;
		}
		mat[jjuv_addr] = L[3];
		break;
	case 7:		//L21 or L(4)
		L[4] = sqrt(T[0] / (T[0] * T[1 * 3 + 1] - T[1 * 3 + 0] * T[1 * 3 + 0]))*(T[2 * 3 + 1] - T[1 * 3 + 0] * T[2 * 3 + 0] / T[0]);
		if (!isfinite(L[4])) {
			*ret = 1.0; break;
		}
		mat[jjuv_addr] = L[4];
		break;
	case 8:		//L22 or (5)
		t1 = -T[2 * 3 + 2] * T[1 * 3 + 0] * T[1 * 3 + 0];
		t2 = 2 * T[2 * 3 + 1] * T[1 * 3 + 0] * T[2 * 3 + 0];
		t3 = -T[1 * 3 + 1] * T[2 * 3 + 0] * T[2 * 3 + 0];
		t4 = T[0 * 3 + 0] * (T[1 * 3 + 1] * T[2 * 3 + 2] - T[2 * 3 + 1] * T[2 * 3 + 1]);
		t5 = -T[1 * 3 + 0] * T[1 * 3 + 0] + T[0 * 3 + 0] * T[1 * 3 + 1];
		L[5] = (t1 + t2 + t3 + t4) / t5;
		if (!isfinite(L[5]) || L[5] <= 0) {
			*ret = 1.0; break;
		}
		L[5] = sqrt(L[5]);
		mat[jjuv_addr] = L[5];
		break;
	default:
		break;
	}//end of compute Ljj
	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步

										//如果有对角元素<=0，启用单列处理方式处理这三列
	if (*ret != 0.0) mat[jjuv_addr] = aux[backup_addr];  //先恢复该列元素
	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步
	if (*ret != 0.0 && get_global_id(0) == 0 && get_global_id(1) == 0)
	{
		//调用单列的modified cholesky, kern_cholmod_diag
		int col = 0;
		void(^kern_wrapper)(void) = ^{
			kern_cholmod_step1(mat,aux, diagInv,diag, ret, mat_size,beta,delta, j,col);
		};
		size_t    global_size = 1;
		ndrange_t ndrange = ndrange_1D(global_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kern_wrapper);
		return;
	}
	if (*ret != 0.0) return;

	//后面是对角元素>0的情况
	//计算Ljj_inv
	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步
	int diagInv_addr = j * 3 * 3 + u * 3 + v;
	switch (u * 3 + v)
	{
	case 0:		//L00 or L(0)
		t1 = 1 / L[0];
		diagInv[diagInv_addr] = t1;
		if (!isfinite(t1)) *ret = 1.0;
		break;
	case 1:		//L01,L02
	case 2:
		diagInv[diagInv_addr] = 0;
		break;
	case 3:		//L10 or L(1)
		t1 = -L[1] / (L[0] * L[2]);
		diagInv[diagInv_addr] = t1;
		if (!isfinite(t1)) *ret = 1.0;
		break;
	case 4:		//L11 or L(2)
		t1 = 1 / L[2];
		diagInv[diagInv_addr] = t1;
		if (!isfinite(t1)) *ret = 1.0;
		break;
	case 5:		//L12
		diagInv[diagInv_addr] = 0;
		break;
	case 6:		//L20 or L(3)
		t1 = (L[1] * L[4] - L[2] * L[3]) / (L[0] * L[2] * L[5]);
		diagInv[diagInv_addr] = t1;
		if (!isfinite(t1)) *ret = 1.0;
		break;
	case 7:		//L21 or L(4)
		t1 = -L[4] / (L[2] * L[5]);
		diagInv[diagInv_addr] = t1;
		if (!isfinite(t1)) *ret = 1.0;
		break;
	case 8:		//L22 or L(5)
		t1 = 1 / L[5];
		diagInv[diagInv_addr] = t1;
		if (!isfinite(t1)) *ret = 1.0;
		break;
	default:
		break;
	}
	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步

	/************************************************************************/
	//调用kern_cholmod_nondiagblk
	if (get_global_id(0) == 0 && get_global_id(1) == 0 && (mat_size - (j + 1) * 3) >= 3)
	{
		*ret = 0.0;
		void(^kern_wrapper)() = ^ () {
			kern_cholmod_blk_step2(mat, aux, diagInv, diag, ret, mat_size, beta, delta, j);
		};
		size_t    global_size[2] = { mat_size - (j + 1) * 3, 3 };
		size_t    local_size[2] = { 3,3 };
		ndrange_t ndrange = ndrange_2D(global_size, local_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kern_wrapper
			);
	}
}


/*
计算剩余块,  workitem size=mat_size - (j+1)* 3
1, Tij=Aij-sum(Lik*Ljk^t),  i=j+1:blkmatsize, k=0:j-1
2, Lij=Tij*(Ljj^-t)
*/
__kernel void kern_cholmod_blk_step2(
	__global dtype* mat,
	__global dtype* aux,
	__global dtype* diagInv,
	__global dtype* diag,
	__global dtype* ret,
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j)
{
	dtype3 *ptr1, *ptr2;
	dtype sum = 0.0;
	int u = get_local_id(0);
	int v = get_local_id(1);
	int i = j + get_group_id(0) + 1;

	//计算Tij = Aij - sum(Lik*Ljk^t)中的uv元素
	int backup_addr = i * 3 * 3 + u * 3 + v;
	int ijuv_addr = (i * 3 + u)*mat_size + j * 3 + v;
	int jivu_addr = (j * 3 + v)*mat_size + i * 3 + u;

	sum = mat[ijuv_addr];	//Aij_uv元素
	aux[backup_addr] = sum;		//备份原始元素
	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步

	for (int k = 0; k < j; k++)
	{
		int Lik_base = (i * 3 + u)*mat_size + k * 3;		//Lik的第u行
		int Ljk_base = (j * 3 + v)*mat_size + k * 3;		//Ljk的第v行
		ptr1 = (dtype3 *)&mat[Lik_base];
		ptr2 = (dtype3 *)&mat[Ljk_base];
		sum -= dot(*ptr1, *ptr2);
	}
	mat[ijuv_addr] = sum;		//保存Tij_uv到原矩阵
	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步

										//计算Lij=Tij*(Ljj^-t)中的uv元素
	int urow_base = (i * 3 + u)*mat_size + j * 3;	//Tij的第u行
	int vrow_base = j * 3 * 3 + v * 3;				//(Ljj^-1)的第v行

	ptr1 = (dtype3 *)&mat[urow_base];
	ptr2 = (dtype3 *)&diagInv[vrow_base];
	sum = dot(*ptr1, *ptr2);
	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步

										//后面的时非对角块元素<beta的情况
	mat[ijuv_addr] = sum;
	mat[jivu_addr] = 0;

	if (sum > beta)
		*ret = 1.0;
	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步

										//调用kern_cholmod_blk_step3
	if (get_global_id(0) == 0 && get_global_id(1) == 0)
	{
		//urow_base = mat_size - (j + 1) * 3;		//剩余块的总行数
		void(^kern_wrapper)() = ^ () {
			kern_cholmod_blk_step3(mat, aux, diagInv, diag, ret, mat_size, beta, delta, j);
		};
		size_t    global_size[2] = { mat_size - (j + 1) * 3, 3 };
		size_t    local_size[2] = { 3,3 };
		ndrange_t ndrange = ndrange_2D(global_size, local_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kern_wrapper
			);
	}
}


/*
workitem size=mat_size - (j+1)* 3
*/
__kernel void kern_cholmod_blk_step3(
	__global dtype* mat,
	__global dtype* aux,
	__global dtype* diagInv,
	__global dtype* diag,
	__global dtype* ret,
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j)
{
	int u = get_local_id(0);
	int v = get_local_id(1);
	int i = j + get_group_id(0) + 1;

	//计算Tij = Aij - sum(Lik*Ljk^t)中的uv元素
	int backup_addr = i * 3 * 3 + u * 3 + v;
	int ijuv_addr = (i * 3 + u)*mat_size + j * 3 + v;

	//如果非对角块元素>beta,则仍用单列处理方式处理这三列
	if (*ret != 0.0) mat[ijuv_addr] = aux[backup_addr];  //先恢复该列元素
	if (*ret != 0.0 && get_global_id(0) == 0 && get_global_id(1) == 0)
	{
		//记得恢复对角块
		int jj_baseaddr = (j * 3)*mat_size + j * 3;
		backup_addr = j * 3 * 3;
		mat[jj_baseaddr] = aux[backup_addr];
		jj_baseaddr = jj_baseaddr + mat_size;  backup_addr = backup_addr + 3;
		mat[jj_baseaddr] = aux[backup_addr];
		mat[jj_baseaddr + 1] = aux[backup_addr + 1];
		jj_baseaddr = jj_baseaddr + mat_size;  backup_addr = backup_addr + 3;
		mat[jj_baseaddr++] = aux[backup_addr++];
		mat[jj_baseaddr++] = aux[backup_addr++];
		mat[jj_baseaddr++] = aux[backup_addr++];

		//调用kern_cholmod_step1
		//调用单列的modified cholesky, kern_cholmod_diag
		int col = 0;
		void(^kern_wrapper)(void) = ^{
			kern_cholmod_step1(mat,aux, diagInv,diag, ret, mat_size,beta,delta, j,col);
		};
		size_t    global_size = 1;
		ndrange_t ndrange = ndrange_1D(global_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kern_wrapper);
		return;
	}
	if (*ret != 0.0) return;
	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步
	/***********************************************************/
	//调用kern_cholmod_blk处理j+1的列块
	if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_size(0) >= 3)
	{
		void(^kern_wrapper)(local void *, local void *) = ^ (local void *TT, local void *LL) {
			kern_cholmod_blk(mat, aux, diagInv, diag, ret, TT, LL, mat_size, beta, delta, j + 1);
		};
		size_t    global_size[2] = { 3, 3 };
		size_t    local_size[2] = { 3,3 };
		ndrange_t ndrange = ndrange_2D(global_size, local_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kern_wrapper,
			(unsigned int)(3 * 3 * sizeof(dtype)),		//size of local memory TT
			(unsigned int)(6 * sizeof(dtype))			//size of local memory LL
			);
	}
}


/******************************************************************************/
/******************************************************************************/

/*
按单列的方式计算j*3+col列的modified cholesky
step 1 计算d_j
workitem size=1
*/
__kernel void kern_cholmod_step1(
	__global dtype* mat,
	__global dtype* aux,
	__global dtype* diagInv,
	__global dtype* diag,
	__global dtype* ret,
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j,
	const int col)
{
	dtype d_j, L_jk, sum;
	int xj = j * 3 + col;		//实际的单列列号
	int jj_addr = xj*mat_size + xj;

	*ret = 0;

	//step 1 计算d_j
	sum = mat[jj_addr];
	for (int k = 0; k < xj; k++)
	{
		L_jk = mat[xj*mat_size + k];
		sum -= L_jk*L_jk;
	}
	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步
	sum = fabs(sum);
	d_j = fmax(sum, delta);
	aux[xj] = d_j;
	mat[jj_addr] = sqrt(d_j);

	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步

	//调用step2计算C_ij,L_ij
	int remain = mat_size - (xj + 1);	//该列剩余元素
	if (remain >0) {
		void(^kern_wrapper)(void) = ^{
			kern_cholmod_step2(mat,aux, diagInv,diag, ret, mat_size,beta,delta, j,col);
		};
		size_t    global_size = remain;			//处理数目为该列剩余元素
		ndrange_t ndrange = ndrange_1D(global_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kern_wrapper);
	}
	else if (remain == 0 && col == 2) {
		//当是整个矩阵最后一个对角元素时，调用kern_cholmod_diaginv计算j对角块列的逆
		void(^kern_wrapper)(local void *) = ^ (local void *LL) {
			kern_cholmod_diaginv(mat, aux, diagInv, diag, ret, LL, mat_size, beta, delta, j);
		};
		size_t    global_size[2] = { 3, 3 };
		size_t    local_size[2] = { 3,3 };
		ndrange_t ndrange = ndrange_2D(global_size, local_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kern_wrapper,
			(unsigned int)(3 * 3 * sizeof(dtype))		//size of local memory TT
			);
	}
}


/*
计算该列剩余的元素
workitem size=mat_size-j*3-1
*/
__kernel void kern_cholmod_step2(
	__global dtype* mat,
	__global dtype* aux,
	__global dtype* diagInv,
	__global dtype* diag,
	__global dtype* ret,
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j,
	const int col)
{
	dtype d_j, L_jk, C_ij;
	int xj = j * 3 + col;		//实际的单列列号
	int i = get_global_id(0) + xj + 1;

	int jj_addr = xj*mat_size + xj;
	int ij_addr = i*mat_size + xj;

	//step 2 计算C_ij和L_ij,存到下三角, 计算临时L_ij，存到aux的第2个matsize空间
	C_ij = mat[ij_addr];
	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步

	for (int k = 0; k < xj; k++)
	{
		C_ij = C_ij - (mat[i*mat_size + k] * mat[xj*mat_size + k]);
	}
	aux[mat_size + i] = C_ij;
	mat[ij_addr] = C_ij / mat[jj_addr];	//L_ij=C_ij/sqrt(d_j)
										//mat[ij_addr] = C_ij;

	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步

	mat[xj*mat_size + i] = 0;

	if (mat[ij_addr] > beta)		//check |L_ij|>beta
		*ret = 1.0;

	//调用step3
	if (get_global_id(0) == 0 && col<3)
	{
		void(^kern_wrapper)(void) = ^{
			kern_cholmod_step3(mat,aux, diagInv,diag, ret, mat_size,beta,delta, j,col);
		};
		size_t    global_size = 1;			//处理数目为1
		ndrange_t ndrange = ndrange_1D(global_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kern_wrapper);
	}
}


/*
step 3  workitem size=1
*/
__kernel void kern_cholmod_step3(
	__global dtype* mat,
	__global dtype* aux,
	__global dtype* diagInv,
	__global dtype* diag,
	__global dtype* ret,
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j,
	const int col)
{
	dtype theta, tmp, L[5];
	int t;
	int xj = j * 3 + col;
	int jj_addr = xj*mat_size + xj;

	//step 3 检查L_ij, 如果不满足条件，算出theta，修改d_j，然后调用step4重新计算L_ij
	if (*ret == 1.0)
	{
		*ret = 0.0;
		theta = 0.0;
		t = 2 * mat_size;
		for (int k = mat_size + xj + 1; k < t; k++) {
			tmp = fabs(aux[k]);
			theta = fmax(theta, tmp);
		}
		mat[jj_addr] = theta / beta;		//modified d_j
		aux[xj] = mat[jj_addr] * mat[jj_addr];

		//调用step4重新计算L_ij
		void(^kern_wrapper)(void) = ^{
			kern_cholmod_step4(mat,aux, diagInv,diag, ret, mat_size,beta,delta, j,col);
		};
		size_t    global_size = mat_size - (xj + 1);			//处理数目为该列剩余元素
		ndrange_t ndrange = ndrange_1D(global_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kern_wrapper);
	}
	else
	{
		if (col < (3 - 1)) {
			//调用kern_cholmod_step1处理下一单列
			void(^kern_wrapper)(void) = ^{
				kern_cholmod_step1(mat,aux, diagInv,diag, ret, mat_size,beta,delta, j,col + 1);
			};
			size_t    global_size = 1;			//处理数目为1
			ndrange_t ndrange = ndrange_1D(global_size);
			enqueue_kernel(
				get_default_queue(),
				CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
				ndrange, kern_wrapper);
		}
		else if (col == 2) {
			//调用kern_cholmod_diaginv计算j对角块列的逆
			void(^kern_wrapper)(local void *) = ^ (local void *LL) {
				kern_cholmod_diaginv(mat, aux, diagInv, diag, ret, LL, mat_size, beta, delta, j);
			};
			size_t    global_size[2] = { 3, 3 };
			size_t    local_size[2] = { 3,3 };
			ndrange_t ndrange = ndrange_2D(global_size, local_size);
			enqueue_kernel(
				get_default_queue(),
				CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
				ndrange, kern_wrapper,
				(unsigned int)(3 * 3 * sizeof(dtype))		//size of local memory TT
				);
		}
	}
}


/*
重新计算Lij
step4 workitem size=[mat_size-j-1]
*/
__kernel void kern_cholmod_step4(
	__global dtype* mat,
	__global dtype* aux,
	__global dtype* diagInv,
	__global dtype* diag,
	__global dtype* ret,
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j,
	const int col)
{
	int xj = j * 3 + col;
	int i = get_global_id(0) + xj + 1;
	if (get_global_id(0) == 0) 	*ret = 0.0;

	//step 4 重新计算L_ij
	mat[i*mat_size + xj] = aux[mat_size + i] / mat[xj*mat_size + xj];

	if (get_global_id(0) == 0 && col < (3 - 1)) {
		*ret = 0.0;
		//调用step1处理下一单列
		void(^kern_wrapper)(void) = ^{
			kern_cholmod_step1(mat,aux, diagInv,diag, ret, mat_size,beta,delta, j,col + 1);
		};
		size_t    global_size = 1;			//处理数目为1
		ndrange_t ndrange = ndrange_1D(global_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kern_wrapper);
	}
	else if (get_global_id(0) == 0 && col == 2)
	{
		//调用kern_cholmod_diaginv计算j对角块列的逆
		void(^kern_wrapper)(local void *) = ^ (local void *LL) {
			kern_cholmod_diaginv(mat, aux, diagInv, diag, ret, LL, mat_size, beta, delta, j);
		};
		size_t    global_size[2] = { 3, 3 };
		size_t    local_size[2] = { 3,3 };
		ndrange_t ndrange = ndrange_2D(global_size, local_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kern_wrapper,
			(unsigned int)(3 * 3 * sizeof(dtype))		//size of local memory TT
			);
	}
}


/*
计算(j,j)对角列块的逆，存入diagInv
*/
__kernel void kern_cholmod_diaginv(
	__global dtype* mat,
	__global dtype* aux,
	__global dtype* diagInv,
	__global dtype* diag,
	__global dtype* ret,
	__local dtype *L,
	const int mat_size,
	const dtype beta,
	const dtype delta,
	const int j)
{
	dtype t1;
	int u = get_global_id(0);
	int v = get_global_id(1);

	int diagInv_addr = j * 3 * 3 + u * 3 + v;
	int jj_addr = j * 3 * mat_size + j * 3;

	L[0] = mat[jj_addr];
	jj_addr = jj_addr + mat_size;
	L[1] = mat[jj_addr];	L[2] = mat[jj_addr + 1];
	jj_addr = jj_addr + mat_size;
	L[3] = mat[jj_addr++];	L[4] = mat[jj_addr++];	L[5] = mat[jj_addr];

	switch (u * 3 + v)
	{
	case 0:		//L00 or L(0)
		t1 = 1 / L[0];
		diagInv[diagInv_addr] = t1;
		break;
	case 1:		//L01,L02
	case 2:
		diagInv[diagInv_addr] = 0;
		break;
	case 3:		//L10 or L(1)
		t1 = -L[1] / (L[0] * L[2]);
		diagInv[diagInv_addr] = t1;
		break;
	case 4:		//L11 or L(2)
		t1 = 1 / L[2];
		diagInv[diagInv_addr] = t1;
		break;
	case 5:		//L12
		diagInv[diagInv_addr] = 0;
		break;
	case 6:		//L20 or L(3)
		t1 = (L[1] * L[4] - L[2] * L[3]) / (L[0] * L[2] * L[5]);
		diagInv[diagInv_addr] = t1;
		break;
	case 7:		//L21 or L(4)
		t1 = -L[4] / (L[2] * L[5]);
		diagInv[diagInv_addr] = t1;
		break;
	case 8:		//L22 or L(5)
		t1 = 1 / L[5];
		diagInv[diagInv_addr] = t1;
		break;
	default:
		break;
	}
	barrier(CLK_LOCAL_MEM_FENCE);		//工作组同步

	/*********************************************************************/
	//调用kern_cholmod_blk处理j+1的列块
	if (get_global_id(0) == 0 && get_global_id(1) == 0 && (mat_size / 3) >(j + 1))
	{
		void(^s1_wrapper)(local void *, local void *) = ^ (local void *TT, local void *LL) {
			kern_cholmod_blk(mat, aux, diagInv, diag, ret, TT, LL, mat_size, beta, delta, j + 1);
		};
		size_t    global_size[2] = { 3, 3 };
		size_t    local_size[2] = { 3,3 };
		ndrange_t ndrange = ndrange_2D(global_size, local_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, s1_wrapper,
			(unsigned int)(3 * 3 * sizeof(dtype)),		//size of local memory TT
			(unsigned int)(6 * sizeof(dtype))			//size of local memory LL
			);
	}
}




/***********************************************************************************************/
/***********************************************************************************************/

/*
取每一行的最大值，存到out
如果excludeDiag不为0，则排除对角元素，并将对角元素存到out的第二行（此时out需两行的空间）。
*/
__kernel void kern_mat_max(
	__global dtype *mat,
	__global dtype *out,
	const int outOffset,
	const int mat_size,
	const int excludeDiag)
{
	dtype max, t1;
	int r, inAddr, outAddr;

	r = get_global_id(0);
	inAddr = r*mat_size;
	max = 0;
	for (int k = 0; k < mat_size; k++)
	{
		if (excludeDiag && r == k) {
			inAddr++;

			continue;
		}
		t1 = fabs(mat[inAddr++]);
		if (t1 > max)
			max = t1;
	}
	outAddr = r + outOffset;
	out[outAddr] = max;
	//如果excludeDiag为真，将该行的对角元素进行存储
	if (excludeDiag)
		out[outAddr + mat_size] = mat[r*mat_size + r];
}




__kernel void kern_cholmod_E(
	__global dtype *mat,
	__global dtype *diag,		//原矩阵的对角元素, 以及结果的E
	const int mat_size)
{
	int addr;
	dtype sum;
	int i = get_global_id(0);

	sum = 0.0;
	addr = i*mat_size;
	for (int k = 0;k <= i;k++)
	{
		sum += mat[addr] * mat[addr];
		addr++;
	}
	diag[i] = sum - diag[i];
}