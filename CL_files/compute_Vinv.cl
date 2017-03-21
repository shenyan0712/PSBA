

/*
计算V的逆, 使用原始V存放在上三角， V逆存放到原始矩阵的下三角(会覆盖对角线元素)
*/
__kernel void kern_compute_Vinv(
	const int n3Dpts,
	__global dtype *V,
	__global dtype *ret)
{
	int i = get_global_id(0);		//Vi
	dtype a[3][3];
	dtype tmp, T, det;

	dtype a11, a12, a13, a21,a22,a23,a31,a32,a33;

	 a11=a[0][0] = V[i*pnp*pnp + 0 * pnp + 0];
	 a12=a[0][1] = V[i*pnp*pnp + 0 * pnp + 1];
	 a13=a[0][2] = V[i*pnp*pnp + 0 * pnp + 2];

	 a21=a[1][0] = V[i*pnp*pnp + 1 * pnp + 0];
	 a22=a[1][1] = V[i*pnp*pnp + 1 * pnp + 1];
	 a23=a[1][2] = V[i*pnp*pnp + 1 * pnp + 2];

	 a31=a[2][0] = V[i*pnp*pnp + 2 * pnp + 0];
	 a32=a[2][1] = V[i*pnp*pnp + 2 * pnp + 1];
	 a33=a[2][2] = V[i*pnp*pnp + 2 * pnp + 2];

	 T = (a33*a12*a12 - 2 * a12*a13*a23 + a22*a13*a13 + a11*a23 *a23 - a11*a22*a33);

	if ( fabs(T)<1e-16) {
		*ret = 1.0;
		int max_idx = 0;
		if (a[0][0] < a[1][0]) max_idx = 1;
		if (a[max_idx][0] < a[2][0]) max_idx = 2;
		if (max_idx != 0) {
			tmp = a[0][0]; a[0][0] = a[max_idx][0]; a[max_idx][0] = tmp;
			tmp = a[0][1]; a[0][1] = a[max_idx][1]; a[max_idx][1] = tmp;
			tmp = a[0][2]; a[0][2] = a[max_idx][2]; a[max_idx][2] = tmp;
		}
		if (a[0, 0] != 0) {
			a[1][0] = a[1][0] / a[0][0];
			a[2][0] = a[2][0] / a[0][0];
			a[1][1] = a[1][1] - a[1][0] * a[0][1];
			a[1][2] = a[1][2] - a[1][0] * a[0][2];
			a[2][1] = a[2][1] - a[2][0] * a[0][1];
			a[2][2] = a[2][2] - a[2][0] * a[0][2];
		}

		if (a[1][1] < a[2][1]) {
			tmp = a[1][0]; a[1][0] = a[2][0]; a[2][0] = tmp;
			tmp = a[1][1]; a[1][1] = a[2][1]; a[2][1] = tmp;
			tmp = a[1][2]; a[1][2] = a[2][2]; a[2][2] = tmp;
		}
		if (a[1][1] != 0.0) {
			a[2][1] = a[2][1] / a[1][1];
			a[2][2] = a[2][2] - a[2][1] * a[1][2];
		}
		T = a[0][0] * a[1][1] * a[2][2];

		V[i*pnp*pnp + 0 * pnp + 0] = (a22*a33 - a23*a32) / T;
		//V[i*pnp*pnp + 0 * pnp + 1] = -(a12*a33 - a13*a32) / base;
		//V[i*pnp*pnp + 0 * pnp + 2] = (a12*a23 - a13*a22) / base;

		V[i*pnp*pnp + 1 * pnp + 0] = -(a21*a33 - a23*a31) / T;
		V[i*pnp*pnp + 1 * pnp + 1] = (a11*a33 - a13*a31) / T;
		//V[i*pnp*pnp + 1 * pnp + 2] = -(a11*a23 - a13*a21) / base;

		V[i*pnp*pnp + 2 * pnp + 0] = (a21*a32 - a22*a31) / T;
		V[i*pnp*pnp + 2 * pnp + 1] = -(a11*a32 - a12*a31) / T;
		V[i*pnp*pnp + 2 * pnp + 2] = (a11*a22 - a12*a21) / T;
		return;
	}

	///*
	V[i*pnp*pnp + 0 * pnp + 0] = -(-a23*a23 + a22*a33) / T;
	//V[i*pnp*pnp + 0 * pnp + 1] = -(a12*a33 - a13*a32) / base;
	//V[i*pnp*pnp + 0 * pnp + 2] = (a12*a23 - a13*a22) / base;

	V[i*pnp*pnp + 1 * pnp + 0] = -(a13*a23 - a12*a33) / T;
	V[i*pnp*pnp + 1 * pnp + 1] = -(-a13*a13 + a11*a33) /T;
	//V[i*pnp*pnp + 1 * pnp + 2] = -(a11*a23 - a13*a21) / base;

	V[i*pnp*pnp + 2 * pnp + 0] = -(a12*a23 - a13*a22) / T;
	V[i*pnp*pnp + 2 * pnp + 1] = -(a12*a13 - a11*a23) / T;
	V[i*pnp*pnp + 2 * pnp + 2] = -(-a12*a12 + a11*a22) / T;
	//*/

	*ret = 0.0;
}