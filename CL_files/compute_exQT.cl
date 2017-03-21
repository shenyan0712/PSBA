
#define QtX		cams[j*6+3]
#define QtY		cams[j*6+4]
#define QtZ		cams[j*6+5]

#define X		pts3D[i*3]
#define Y		pts3D[i*3+1]
#define Z		pts3D[i*3+2]
#define ax		K[0]
#define ay		K[1]
#define x0		K[2]
#define y0		K[3]

/*
计算投影误差 x-x^
使用四元数
*/
__kernel void kern_compute_exQT(
	__global dtype *K,
	__global dtype *impts,
	__global dtype *initcams,
	__global dtype *cams,
	__global dtype *pts3D,
	__global int *iidx,
	__global int *jidx,
	__global dtype *ex
)
{
	int idx = get_global_id(0);		//获出本工作项处理的图像点的索引
	int i = iidx[idx];				//得到对应的3D点索引
	int j = jidx[idx];				//得到对应的相机的索引

									//先计算初始旋转与增量旋转的四元数乘
	int base = j * 4;
	int Kbase = j*5;
	dtype s0 = initcams[base++];	//初始qauternion
	dtype v0_1 = initcams[base++];
	dtype v0_2 = initcams[base++];
	dtype v0_3 = initcams[base];
	base = j * 6;
	dtype vi_1 = cams[base++];	//增量qauternion
	dtype vi_2 = cams[base++];
	dtype vi_3 = cams[base];
	dtype si = sqrt(1 - vi_1*vi_1 - vi_2 *vi_2 - vi_3*vi_3);
	//qi*q0
	dtype s = si*s0 - (v0_1*vi_1 + v0_2*vi_2 + v0_3*vi_3);
	dtype v1 = s0*vi_1 + si*v0_1 + v0_3*vi_2 - v0_2*vi_3;
	dtype v2 = s0*vi_2 + si*v0_2 + v0_1*vi_3 - v0_3*vi_1;
	dtype v3 = s0*vi_3 + si*v0_3 + v0_2*vi_1 - v0_1*vi_2;

	//到这里得到合成的四元数 <s,v1,v2,v3>
	dtype t1 = K[Kbase +0];
	dtype t2 = v1;
	dtype t3 = pts3D[i * 3];
	dtype t5 = v2;
	dtype t6 = pts3D[i * 3 + 1];
	dtype t8 = v3;
	dtype t9 = pts3D[i * 3 + 2];
	dtype t11 = -t3*t2 - t5*t6 - t8*t9;
	dtype t13 = s;
	dtype t17 = t13*t3 + t5*t9 - t8*t6;
	dtype t22 = t6*t13 + t8*t3 - t9*t2;
	dtype t27 = t13*t9 + t6*t2 - t5*t3;
	dtype t38 = -t5*t11 + t13*t22 - t27*t2 + t8*t17 + QtY;
	dtype t46 = -t11*t8 + t13*t27 - t5*t17 + t2*t22 + QtZ;
	dtype t49 = 1 / t46;

	ex[idx * 2] = impts[idx * 2] - (t1*(-t2*t11 + t13*t17 - t22*t8 + t5*t27 + QtX) + K[Kbase + 4] * t38 + K[Kbase + 1] * t46)*t49;
	ex[idx * 2 + 1] = impts[idx * 2 + 1] - (t1*K[Kbase + 3] * t38 + K[Kbase + 2] * t46)*t49;

}



/*
计算投影误差 x-x^
使用四元数
*/
__kernel void kern_compute_exQT_backup(
	__global dtype *K,
	__global dtype *impts,
	__global dtype *initcams,
	__global dtype *cams,
	__global dtype *pts3D,
	__global int *iidx,
	__global int *jidx,
	__global dtype *ex
)
{
	int idx = get_global_id(0);		//获出本工作项处理的图像点的索引
	int i = iidx[idx];				//得到对应的3D点索引
	int j = jidx[idx];				//得到对应的相机的索引

									//先计算初始旋转与增量旋转的四元数乘
	int base = j * 4;
	dtype s0 = initcams[base++];	//初始qauternion
	dtype v0_1 = initcams[base++];
	dtype v0_2 = initcams[base++];
	dtype v0_3 = initcams[base];
	base = j * 6;
	dtype vi_1 = cams[base++];	//增量qauternion
	dtype vi_2 = cams[base++];
	dtype vi_3 = cams[base];
	dtype si = sqrt(1 - vi_1*vi_1 - vi_2 *vi_2 - vi_3*vi_3);
	//qi*q0
	dtype s = si*s0 - (v0_1*vi_1 + v0_2*vi_2 + v0_3*vi_3);
	dtype v1 = s0*vi_1 + si*v0_1 + v0_3*vi_2 - v0_2*vi_3;
	dtype v2 = s0*vi_2 + si*v0_2 + v0_1*vi_3 - v0_3*vi_1;
	dtype v3 = s0*vi_3 + si*v0_3 + v0_2*vi_1 - v0_1*vi_2;

	//到这里得到合成的四元数 <s,v1,v2,v3>
	dtype t1 = K[0];
	dtype t2 = v1;
	dtype t3 = pts3D[i * 3];
	dtype t5 = v2;
	dtype t6 = pts3D[i * 3 + 1];
	dtype t8 = v3;
	dtype t9 = pts3D[i * 3 + 2];
	dtype t11 = -t3*t2 - t5*t6 - t8*t9;
	dtype t13 = s;
	dtype t17 = t13*t3 + t5*t9 - t8*t6;
	dtype t22 = t6*t13 + t8*t3 - t9*t2;
	dtype t27 = t13*t9 + t6*t2 - t5*t3;
	dtype t38 = -t5*t11 + t13*t22 - t27*t2 + t8*t17 + QtY;
	dtype t46 = -t11*t8 + t13*t27 - t5*t17 + t2*t22 + QtZ;
	dtype t49 = 1 / t46;

	ex[idx * 2] = impts[idx * 2] - (t1*(-t2*t11 + t13*t17 - t22*t8 + t5*t27 + QtX) + K[4] * t38 + K[1] * t46)*t49;
	ex[idx * 2 + 1] = impts[idx * 2 + 1] - (t1*K[3] * t38 + K[2] * t46)*t49;

}
