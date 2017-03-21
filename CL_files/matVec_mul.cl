


/*
计算Mv，矩阵与向量的乘积
*/
__kernel void kern_matVec_mul(const int dim,
	__global dtype *M,
	__global dtype *v,
	__global dtype *out)
{
	int i = get_global_id(0);
	dtype sum = 0;

	for (int k = 0; k < dim; k++)
		sum = sum + M[i*dim + k] * v[k];
	out[i] = sum;
}