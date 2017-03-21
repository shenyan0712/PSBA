

/*
new_p=p+dp
*/
__kernel void kern_compute_newp(
	const int nCamParas,
	const int n3DptsParas,
	__global dtype *cams,
	__global dtype *pts3D,
	__global dtype *dp,
	__global dtype *new_cams,
	__global dtype *new_pts3D
	)
{
	int idx = get_global_id(0);

	if (idx <nCamParas) {
		new_cams[idx] = cams[idx] + dp[idx];
	}
	else {
		idx = idx - nCamParas;
		new_pts3D[idx] = pts3D[idx] + dp[nCamParas + idx];
	}

}