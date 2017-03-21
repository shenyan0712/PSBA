

/*
计算Jacobian
with Quaternion
*/
__kernel void kern_compute_jacobiQT(
	__global dtype *K,			//nCams*5
	__global dtype *impts,
	__global dtype *initcams,	//初始相机旋转的四元数 4-vec
	__global dtype *cams,	//使用四元数向量部分(3 parameters)
	__global dtype *pts3D,
	__global int *iidx,
	__global int *jidx,
	__global dtype *JA,
	__global dtype *JB)
{
	int idx = get_global_id(0);		//获出本工作项处理的图像点的索引
	int i = iidx[idx];				//得到对应的3D点索引
	int j = jidx[idx];				//得到对应的相机的索引
	int Kbase = j * 5;

	dtype t1 = K[Kbase + 0];
	dtype t2 = cams[j * 6 + 0];
	dtype t3 = t2*t2;
	dtype t4 = cams[j * 6 + 1];
	dtype t5 = t4*t4;
	dtype t6 = cams[j * 6 + 2];
	dtype t7 = t6*t6;
	dtype t9 = sqrt(1.0 - t3 - t5 - t7);
	dtype t10 = 1 / t9;
	dtype t11 = initcams[j * 4 + 1];
	dtype t12 = t11*t10;
	dtype t14 = initcams[j * 4 + 0];
	dtype t15 = -t12*t2 + t14;
	dtype t16 = pts3D[i * 3 + 0];
	dtype t18 = initcams[j * 4 + 2];
	dtype t19 = t18*t10;
	dtype t21 = initcams[j * 4 + 3];
	dtype t22 = -t19*t2 - t21;
	dtype t23 = pts3D[i * 3 + 1];
	dtype t25 = t10*t21;
	dtype t27 = -t25*t2 + t18;
	dtype t28 = pts3D[i * 3 + 2];
	dtype t30 = -t15*t16 - t22*t23 - t27*t28;
	dtype t35 = -t9*t11 - t2*t14 - t4*t21 + t6*t18;
	dtype t37 = -t35;
	dtype t43 = t9*t18 + t4*t14 + t6*t11 - t2*t21;
	dtype t49 = t9*t21 + t6*t14 + t2*t18 - t11*t4;
	dtype t51 = -t37*t16 - t43*t23 - t49*t28;
	dtype t52 = -t15;
	dtype t54 = t10*t14;
	dtype t56 = -t54*t2 - t11;
	dtype t61 = t9*t14 - t2*t11 - t4*t18 - t6*t21;
	dtype t65 = t61*t16 + t43*t28 - t23*t49;
	dtype t70 = t56*t16 + t22*t28 - t23*t27;
	dtype t75 = t56*t23 + t27*t16 - t28*t15;
	dtype t76 = -t49;
	dtype t81 = t61*t23 + t49*t16 - t37*t28;
	dtype t82 = -t27;
	dtype t87 = t56*t28 + t23*t15 - t22*t16;
	dtype t88 = -t43;
	dtype t93 = t61*t28 + t37*t23 - t43*t16;
	dtype t94 = -t22;
	dtype t98 = K[Kbase + 4];
	dtype t107 = t30*t88 + t94*t51 + t56*t81 + t61*t75 + t87*t35 + t93*t52 - t70*t76 - t82*t65;
	dtype t109 = K[Kbase + 1];
	dtype t118 = t30*t76 + t82*t51 + t56*t93 + t61*t87 + t70*t88 + t65*t94 - t35*t75 - t81*t52;
	dtype t126 = t76*t51 + t61*t93 + t65*t88 - t81*t35 + cams[j * 6 + 3 + 2];
	dtype t127 = 1 / t126;
	dtype t141 = t51*t88 + t61*t81 + t93*t35 - t65*t76 + cams[j * 6 + 3 + 1];
	dtype t145 = t126*t126;
	dtype t146 = 1 / t145;
	dtype t147 = (t1*(t35*t51 + t61*t65 + t81*t76 - t93*t88 + cams[j * 6 + 3 + 0]) + t98*t141 + t126*t109)*t146;
	JA[idx * 12] = (t1*(t30*t35 + t52*t51 + t56*t65 + t61*t70 + t76*t75 + t81*t82 - t88*t87
		- t93*t94) + t98*t107 + t109*t118)*t127 - t118*t147;
	dtype t150 = t1*K[Kbase + 3];
	dtype t152 = K[Kbase + 2];
	dtype t159 = (t150*t141 + t126*t152)*t146;
	JA[idx * 12 + 6] = (t107*t150 + t152*t118)*t127 - t159*t118;
	dtype t162 = -t12*t4 + t21;
	dtype t165 = -t19*t4 + t14;
	dtype t168 = -t25*t4 - t11;
	dtype t170 = -t162*t16 - t165*t23 - t168*t28;
	dtype t172 = -t162;
	dtype t175 = -t54*t4 - t18;
	dtype t180 = t175*t16 + t165*t28 - t168*t23;
	dtype t185 = t175*t23 + t168*t16 - t162*t28;
	dtype t187 = -t168;
	dtype t192 = t175*t28 + t162*t23 - t165*t16;
	dtype t194 = -t165;
	dtype t206 = t170*t88 + t51*t194 + t175*t81 + t61*t185 + t192*t35 + t93*t172 - t76*t180 - t65*t187;
	dtype t216 = t170*t76 + t51*t187 + t93*t175 + t61*t192 + t180*t88 + t65*t194 - t185*t35 - t81*t172;
	JA[idx * 12 + 1] = (t1*(t170*t35 + t172*t51 + t175*t65 + t180*t61 + t185*t76 + t81*t187 -
		t192*t88 - t93*t194) + t98*t206 + t109*t216)*t127 - t147*t216;
	JA[idx * 12 + 7] = (t150*t206 + t152*t216)*t127 - t159*t216;
	dtype t227 = -t12*t6 - t18;
	dtype t230 = -t19*t6 + t11;
	dtype t233 = -t25*t6 + t14;
	dtype t235 = -t227*t16 - t23*t230 - t233*t28;
	dtype t237 = -t227;
	dtype t240 = -t54*t6 - t21;
	dtype t245 = t240*t16 + t230*t28 - t233*t23;
	dtype t250 = t23*t240 + t233*t16 - t227*t28;
	dtype t252 = -t233;
	dtype t257 = t240*t28 + t227*t23 - t230*t16;
	dtype t259 = -t230;
	dtype t271 = t235*t88 + t51*t259 + t81*t240 + t61*t250 + t257*t35 + t93*t237 - t245*t76 - t65*t252;
	dtype t281 = t235*t76 + t51*t252 + t240*t93 + t61*t257 + t245*t88 + t259*t65 - t250*t35 - t81*t237;
	JA[idx * 12 + 2] = (t1*(t235*t35 + t237*t51 + t240*t65 + t61*t245 + t250*t76 + t81*t252 -
		t257*t88 - t93*t259) + t271*t98 + t281*t109)*t127 - t147*t281;
	JA[idx * 12 + 8] = (t150*t271 + t281*t152)*t127 - t159*t281;
	JA[idx * 12 + 3] = t127*t1;
	JA[idx * 12 + 9] = 0.0;
	JA[idx * 12 + 4] = t98*t127;
	JA[idx * 12 + 10] = t150*t127;
	JA[idx * 12 + 5] = t109*t127 - t147;
	JA[idx * 12 + 11] = t152*t127 - t159;
	dtype t293 = t35*t35;
	dtype t294 = t61*t61;
	dtype t296 = t88*t88;
	dtype t299 = t35*t88;
	dtype t302 = t61*t76;
	dtype t303 = 2.0*t299 + t61*t49 - t302;
	dtype t305 = t35*t76;
	dtype t306 = t61*t88;
	dtype t309 = t305 + 2.0*t306 - t49*t35;
	JB[idx * 6] = (t1*(t293 + t294 + t49*t76 - t296) + t98*t303 + t109*t309)*t127 - t147*t309;
	JB[idx * 6 + 3] = (t150*t303 + t152*t309)*t127 - t159*t309;
	dtype t324 = t76*t76;
	dtype t325 = t296 + t294 + t35*t37 - t324;
	dtype t327 = t76*t88;
	dtype t330 = t61*t35;
	dtype t331 = 2.0*t327 + t61*t37 - t330;
	JB[idx * 6 + 1] = (t1*(t299 + 2.0*t302 - t37*t88) + t98*t325 + t109*t331)*t127 - t147*t331;
	JB[idx * 6 + 4] = (t150*t325 + t152*t331)*t127 - t159*t331;
	dtype t347 = t327 + 2.0*t330 - t43*t76;
	dtype t350 = t324 + t294 + t43*t88 - t293;
	JB[idx * 6 + 2] = (t1*(2.0*t305 + t61*t43 - t306) + t98*t347 + t350*t109)*t127 - t147*t350;
	JB[idx * 6 + 5] = (t150*t347 + t152*t350)*t127 - t159*t350;
}



/*
计算Jacobian
with Quaternion
*/
__kernel void kern_compute_jacobiQT_bak(
	__global dtype *K,			//
	__global dtype *impts,
	__global dtype *initcams,	//初始相机旋转的四元数 4-vec
	__global dtype *cams,	//使用四元数向量部分(3 parameters)
	__global dtype *pts3D,
	__global int *iidx,
	__global int *jidx,
	__global dtype *JA,
	__global dtype *JB)
{
	int idx = get_global_id(0);		//获出本工作项处理的图像点的索引
	int i = iidx[idx];				//得到对应的3D点索引
	int j = jidx[idx];				//得到对应的相机的索引

	dtype t1 = K[0];
	dtype t2 = cams[j * 6 + 0];
	dtype t3 = t2*t2;
	dtype t4 = cams[j * 6 + 1];
	dtype t5 = t4*t4;
	dtype t6 = cams[j * 6 + 2];
	dtype t7 = t6*t6;
	dtype t9 = sqrt(1.0 - t3 - t5 - t7);	//
	dtype t10 = 1 / t9;
	dtype t11 = initcams[j * 4 + 1];
	dtype t12 = t11*t10;
	dtype t14 = initcams[j * 4 + 0];
	dtype t15 = -t12*t2 + t14;
	dtype t16 = pts3D[i * 3 + 0];
	dtype t18 = initcams[j * 4 + 2];
	dtype t19 = t18*t10;
	dtype t21 = initcams[j * 4 + 3];
	dtype t22 = -t19*t2 - t21;
	dtype t23 = pts3D[i * 3 + 1];
	dtype t25 = t10*t21;
	dtype t27 = -t25*t2 + t18;
	dtype t28 = pts3D[i * 3 + 2];
	dtype t30 = -t15*t16 - t22*t23 - t27*t28;
	dtype t35 = -t9*t11 - t2*t14 - t4*t21 + t6*t18;
	dtype t37 = -t35;
	dtype t43 = t9*t18 + t4*t14 + t6*t11 - t2*t21;
	dtype t49 = t9*t21 + t6*t14 + t2*t18 - t11*t4;
	dtype t51 = -t37*t16 - t43*t23 - t49*t28;
	dtype t52 = -t15;
	dtype t54 = t10*t14;
	dtype t56 = -t54*t2 - t11;
	dtype t61 = t9*t14 - t2*t11 - t4*t18 - t6*t21;
	dtype t65 = t61*t16 + t43*t28 - t23*t49;
	dtype t70 = t56*t16 + t22*t28 - t23*t27;
	dtype t75 = t56*t23 + t27*t16 - t28*t15;
	dtype t76 = -t49;
	dtype t81 = t61*t23 + t49*t16 - t37*t28;
	dtype t82 = -t27;
	dtype t87 = t56*t28 + t23*t15 - t22*t16;
	dtype t88 = -t43;
	dtype t93 = t61*t28 + t37*t23 - t43*t16;
	dtype t94 = -t22;
	dtype t98 = K[4];
	dtype t107 = t30*t88 + t94*t51 + t56*t81 + t61*t75 + t87*t35 + t93*t52 - t70*t76 - t82*t65;
	dtype t109 = K[2];
	dtype t118 = t30*t76 + t82*t51 + t56*t93 + t61*t87 + t70*t88 + t65*t94 - t35*t75 - t81*t52;
	dtype t126 = t76*t51 + t61*t93 + t65*t88 - t81*t35 + cams[j * 6 + 3 + 2];
	dtype t127 = 1 / t126;
	dtype t141 = t51*t88 + t61*t81 + t93*t35 - t65*t76 + cams[j * 6 + 3 + 1];
	dtype t145 = t126*t126;
	dtype t146 = 1 / t145;
	dtype t147 = (t1*(t35*t51 + t61*t65 + t81*t76 - t93*t88 + cams[j * 6 + 3 + 0]) + t98*t141 + t126*t109)*t146;
	JA[idx * 12] = (t1*(t30*t35 + t52*t51 + t56*t65 + t61*t70 + t76*t75 + t81*t82 - t88*t87
		- t93*t94) + t98*t107 + t109*t118)*t127 - t118*t147;
	//dtype t150 = t1*K[3];
	dtype t150 = K[1];
	dtype t152 = K[3];
	dtype t159 = (t150*t141 + t126*t152)*t146;
	JA[idx * 12 + 6] = (t107*t150 + t152*t118)*t127 - t159*t118;
	dtype t162 = -t12*t4 + t21;
	dtype t165 = -t19*t4 + t14;
	dtype t168 = -t25*t4 - t11;
	dtype t170 = -t162*t16 - t165*t23 - t168*t28;
	dtype t172 = -t162;
	dtype t175 = -t54*t4 - t18;
	dtype t180 = t175*t16 + t165*t28 - t168*t23;
	dtype t185 = t175*t23 + t168*t16 - t162*t28;
	dtype t187 = -t168;
	dtype t192 = t175*t28 + t162*t23 - t165*t16;
	dtype t194 = -t165;
	dtype t206 = t170*t88 + t51*t194 + t175*t81 + t61*t185 + t192*t35 + t93*t172 - t76*t180 - t65*t187;
	dtype t216 = t170*t76 + t51*t187 + t93*t175 + t61*t192 + t180*t88 + t65*t194 - t185*t35 - t81*t172;
	JA[idx * 12 + 1] = (t1*(t170*t35 + t172*t51 + t175*t65 + t180*t61 + t185*t76 + t81*t187 -
		t192*t88 - t93*t194) + t98*t206 + t109*t216)*t127 - t147*t216;
	JA[idx * 12 + 7] = (t150*t206 + t152*t216)*t127 - t159*t216;
	dtype t227 = -t12*t6 - t18;
	dtype t230 = -t19*t6 + t11;
	dtype t233 = -t25*t6 + t14;
	dtype t235 = -t227*t16 - t23*t230 - t233*t28;
	dtype t237 = -t227;
	dtype t240 = -t54*t6 - t21;
	dtype t245 = t240*t16 + t230*t28 - t233*t23;
	dtype t250 = t23*t240 + t233*t16 - t227*t28;
	dtype t252 = -t233;
	dtype t257 = t240*t28 + t227*t23 - t230*t16;
	dtype t259 = -t230;
	dtype t271 = t235*t88 + t51*t259 + t81*t240 + t61*t250 + t257*t35 + t93*t237 - t245*t76 - t65*t252;
	dtype t281 = t235*t76 + t51*t252 + t240*t93 + t61*t257 + t245*t88 + t259*t65 - t250*t35 - t81*t237;
	JA[idx * 12 + 2] = (t1*(t235*t35 + t237*t51 + t240*t65 + t61*t245 + t250*t76 + t81*t252 -
		t257*t88 - t93*t259) + t271*t98 + t281*t109)*t127 - t147*t281;
	JA[idx * 12 + 8] = (t150*t271 + t281*t152)*t127 - t159*t281;
	JA[idx * 12 + 3] = t127*t1;
	JA[idx * 12 + 9] = 0.0;
	JA[idx * 12 + 4] = t98*t127;
	JA[idx * 12 + 10] = t150*t127;
	JA[idx * 12 + 5] = t109*t127 - t147;
	JA[idx * 12 + 11] = t152*t127 - t159;
	dtype t293 = t35*t35;
	dtype t294 = t61*t61;
	dtype t296 = t88*t88;
	dtype t299 = t35*t88;
	dtype t302 = t61*t76;
	dtype t303 = 2.0*t299 + t61*t49 - t302;
	dtype t305 = t35*t76;
	dtype t306 = t61*t88;
	dtype t309 = t305 + 2.0*t306 - t49*t35;
	JB[idx * 6] = (t1*(t293 + t294 + t49*t76 - t296) + t98*t303 + t109*t309)*t127 - t147*t309;
	JB[idx * 6 + 3] = (t150*t303 + t152*t309)*t127 - t159*t309;
	dtype t324 = t76*t76;
	dtype t325 = t296 + t294 + t35*t37 - t324;
	dtype t327 = t76*t88;
	dtype t330 = t61*t35;
	dtype t331 = 2.0*t327 + t61*t37 - t330;
	JB[idx * 6 + 1] = (t1*(t299 + 2.0*t302 - t37*t88) + t98*t325 + t109*t331)*t127 - t147*t331;
	JB[idx * 6 + 4] = (t150*t325 + t152*t331)*t127 - t159*t331;
	dtype t347 = t327 + 2.0*t330 - t43*t76;
	dtype t350 = t324 + t294 + t43*t88 - t293;
	JB[idx * 6 + 2] = (t1*(2.0*t305 + t61*t43 - t306) + t98*t347 + t350*t109)*t127 - t147*t350;
	JB[idx * 6 + 5] = (t150*t347 + t152*t350)*t127 - t159*t350;
}