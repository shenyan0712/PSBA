

/*
p=new_p
*/
__kernel void kern_update_p(
	const int nCamParas,
	const int n3DptsParas,
	__global dtype *cams,
	__global dtype *pts3D,
	__global dtype *new_cams,
	__global dtype *new_pts3D
	)
{
	int idx = get_global_id(0);

	if (idx <nCamParas) {
		cams[idx] = new_cams[idx];
	}
	else {
		idx = idx - nCamParas;
		pts3D[idx] = new_pts3D[idx];
	}

}