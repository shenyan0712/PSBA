#include "stdafx.h"

#include <math.h>

#include "psba.h"

extern FILE *debug_file;
extern int itno;		//当前迭代次数


void calcImgProjJacRTS(dtype a[5], dtype qr0[4], dtype v[3], dtype t[3],
	dtype M[3], dtype jacmRT[2][6], dtype jacmS[2][3]);
void quatMultFast(dtype q1[4], dtype q2[4], dtype p[4]);
void calcImgProjFullR(dtype a[5], dtype qr0[4], dtype t[3], dtype M[3],
	dtype n[2]);


/*
a: K参数
qr0: 初始四元数
v:  改变四元数
t:  位移
M:  3D点坐标
jacmRT：输出的jac_A
jacmS：输出的jac_B
*/
//计算jacobian
void compute_jacobian(int nCams, int n3Dpts, int cnp, int pnp,int n2Dprojs, 
	dtype *K,
	dtype *initCamParas,
	dtype *motion_data,
	int *iidx,int *jidx,
	dtype *jac_A,dtype *jac_B)
{
	int i, j;
	dtype *ptrA, *ptrB, *pts3D, *v, *t, *qr0;
	for (int idx = 0; idx < n2Dprojs; idx++)
	{
		i = iidx[idx];		//对应第i个3D点
		j = jidx[idx];
		v = &motion_data[j * cnp];
		t = &motion_data[j * cnp + 3];
		ptrA = &jac_A[idx * 2 * cnp];
		ptrB = &jac_B[idx * 2 * pnp];
		pts3D = &motion_data[nCams * cnp + i * pnp];
		qr0 = &initCamParas[j * 4];

		calcImgProjJacRTS(K, qr0, v, t, pts3D, (dtype(*)[6])ptrA, (dtype(*)[3])ptrB);
	}

	//
	fprintf(debug_file, "****************************** [iter %d] ******************************\n",itno);
	//display for check
	for (int i = 0; i < n2Dprojs; i++) {
		fprintf(debug_file, "3DP %d:\t", iidx[i]);
		for (int j = 0; j < pnp; j++)
			fprintf(debug_file, "%lf  ", motion_data[nCams*cnp + iidx[i] * pnp + j]);
		fprintf(debug_file, "\n");
		fprintf(debug_file, "Cam %d:\t", jidx[i]);
		for (int j = 0; j < cnp; j++)
			fprintf(debug_file, "%lf  ", motion_data[jidx[i] * cnp + j]);
		fprintf(debug_file, "\n");
		fprintf(debug_file, "Jac idx:%d\n", i);
		fprintf(debug_file, "JA(%d,%d):\n", iidx[i], jidx[i]);
		for (int r = 0; r < 2; r++) {
			for (int c = 0; c < cnp; c++)
				fprintf(debug_file, "%le  ", jac_A[i * 2 * cnp + r*cnp + c]);
			fprintf(debug_file, "\n");
		}
		fprintf(debug_file, "JB(%d,%d):\n", iidx[i], jidx[i]);
		for (int r = 0; r < 2; r++) {
			for (int c = 0; c < pnp; c++)
				fprintf(debug_file, "%le  ", jac_B[i * 2 * pnp + r * pnp + c]);
			fprintf(debug_file, "\n");
		}
		fprintf(debug_file, "---------------------------------------\n");
	}

}


void compute_proj_err(
	int cnp, int pnp, int mnp,		//分别是相机参数维度(12)，3D点坐标维度(3)，2D图像点坐标维度(2)
	int n3Dpts, int nCams, int n2Dprojs,				//3D点数量，相机数量，2D图像点数量
	dtype *K,
	dtype *impts_data,		//2D图像点数据
	dtype *initcams_data,	//初始相机参数
	dtype *motion_data,		//相机参数(9+3, R,t)+3D点坐标(X,Y,Z)
	int *iidx,				//大小为n2Dprojs，指示对应2D点对应的3D点索引
	int *jidx,				//大小为n2Dprojs，指示对应2D点对应的相机索引
	//输出
	dtype *ex				//存储假设投影点hx_ij-->第i个3D点到第j个相机上的投影与测量之间的误差。按ex_11,ex12,...,ex_nm的顺序存放。
	)
{
	int i, j;
	dtype trot[4], lrot[4];
	dtype *ptr1, *ptr2, *ptr3, *ptr4, *ptr5;

	for (int k = 0;k < n2Dprojs;k++)
	{
		i = iidx[k]; j = jidx[k];
		ptr1 = &initcams_data[j * 4];		//qr0
		ptr2 = &motion_data[j * 6];
		ptr3 = &motion_data[nCams * 6 + i * 3];	//3D point
		ptr4 = &ex[2 * k];		//2D x,y estimate
		ptr5 = &impts_data[2 * k];
		lrot[0] = sqrt(1 - ptr2[0] * ptr2[0] - ptr2[1] * ptr2[1] - ptr2[2] * ptr2[2]);
		lrot[1] = ptr2[0];
		lrot[2] = ptr2[1];
		lrot[3] = ptr2[2];
		quatMultFast(lrot, ptr1, trot);
		calcImgProjFullR(K, trot, &ptr2[3], ptr3, ptr4);

		ptr4[0] = ptr5[0] - ptr4[0];
		ptr4[1] = ptr5[1] - ptr4[1]; 
	}

#if DEBUG_EX==1
	fprintf(debug_file, "*******************************iter %d*******************************\n", itno);
	//display for check
	for (int i = 0; i < n2Dprojs; i++) {
		/*
		fprintf(debug_file, "3D Point (%d):\t", iidx[i]);
		for (int j = 0;j < pnp; j++)
		fprintf(debug_file, "%lf  ", motion_data[nCams*cnp+iidx[i] * pnp + j]);
		fprintf(debug_file, "\n");
		fprintf(debug_file, "Cam (%d):\t", jidx[i]);
		for (int j = 0;j < cnp; j++)
		fprintf(debug_file, "%lf  ", motion_data[jidx[i] * cnp + j]);
		fprintf(debug_file, "\n");
		fprintf(debug_file, "2D Point: %lf  %lf\n", impts_data[i*2], impts_data[i * 2+1]);
		*/
		fprintf(debug_file, "(%d %d)L2 proj err: %le   %le\n", iidx[i], jidx[i], ex[i * 2], ex[i * 2 + 1]);
		//fprintf(debug_file, "-----------------------------\n");
	}
	fflush(debug_file);
#endif


}



/*
四元数相乘计算
*/
inline static void quatMultFast(dtype q1[4], dtype q2[4], dtype p[4])
{
	dtype t1, t2, t3, t4, t5, t6, t7, t8, t9;
	//dtype t10, t11, t12;

	t1 = (q1[0] + q1[1])*(q2[0] + q2[1]);
	t2 = (q1[3] - q1[2])*(q2[2] - q2[3]);
	t3 = (q1[1] - q1[0])*(q2[2] + q2[3]);
	t4 = (q1[2] + q1[3])*(q2[1] - q2[0]);
	t5 = (q1[1] + q1[3])*(q2[1] + q2[2]);
	t6 = (q1[1] - q1[3])*(q2[1] - q2[2]);
	t7 = (q1[0] + q1[2])*(q2[0] - q2[3]);
	t8 = (q1[0] - q1[2])*(q2[0] + q2[3]);

#if 0
	t9 = t5 + t6;
	t10 = t7 + t8;
	t11 = t5 - t6;
	t12 = t7 - t8;

	p[0] = t2 + 0.5*(-t9 + t10);
	p[1] = t1 - 0.5*(t9 + t10);
	p[2] = -t3 + 0.5*(t11 + t12);
	p[3] = -t4 + 0.5*(t11 - t12);
#endif

	/* following fragment it equivalent to the one above */
	t9 = 0.5*(t5 - t6 + t7 + t8);
	p[0] = t2 + t9 - t5;
	p[1] = t1 - t9 - t6;
	p[2] = -t3 + t9 - t8;
	p[3] = -t4 + t9 - t7;
}


/*
计算M的投影，a为K参数，qr0为旋转四元数，t为位移
*/
void calcImgProjFullR(dtype a[5], dtype qr0[4], dtype t[3], dtype M[3],
	dtype n[2])
{
	dtype t1;
	dtype t11;
	dtype t13;
	dtype t17;
	dtype t2;
	dtype t22;
	dtype t27;
	dtype t3;
	dtype t38;
	dtype t46;
	dtype t49;
	dtype t5;
	dtype t6;
	dtype t8;
	dtype t9;
	{
		t1 = a[0];
		t2 = qr0[1];
		t3 = M[0];
		t5 = qr0[2];
		t6 = M[1];
		t8 = qr0[3];
		t9 = M[2];
		t11 = -t3*t2 - t5*t6 - t8*t9;
		t13 = qr0[0];
		t17 = t13*t3 + t5*t9 - t8*t6;
		t22 = t6*t13 + t8*t3 - t9*t2;
		t27 = t13*t9 + t6*t2 - t5*t3;
		t38 = -t5*t11 + t13*t22 - t27*t2 + t8*t17 + t[1];
		t46 = -t11*t8 + t13*t27 - t5*t17 + t2*t22 + t[2];
		t49 = 1 / t46;
		n[0] = (t1*(-t2*t11 + t13*t17 - t22*t8 + t5*t27 + t[0]) + a[4] * t38 + a[1] * t46)*t49;
		n[1] = (t1*a[3] * t38 + a[2] * t46)*t49;
		return;
	}
}



void calcImgProjJacRTS(dtype a[5], dtype qr0[4], dtype v[3], dtype t[3],
	dtype M[3], dtype jacmRT[2][6], dtype jacmS[2][3])
{
	dtype t1;
	dtype t10;
	dtype t107;
	dtype t109;
	dtype t11;
	dtype t118;
	dtype t12;
	dtype t126;
	dtype t127;
	dtype t14;
	dtype t141;
	dtype t145;
	dtype t146;
	dtype t147;
	dtype t15;
	dtype t150;
	dtype t152;
	dtype t159;
	dtype t16;
	dtype t162;
	dtype t165;
	dtype t168;
	dtype t170;
	dtype t172;
	dtype t175;
	dtype t18;
	dtype t180;
	dtype t185;
	dtype t187;
	dtype t19;
	dtype t192;
	dtype t194;
	dtype t2;
	dtype t206;
	dtype t21;
	dtype t216;
	dtype t22;
	dtype t227;
	dtype t23;
	dtype t230;
	dtype t233;
	dtype t235;
	dtype t237;
	dtype t240;
	dtype t245;
	dtype t25;
	dtype t250;
	dtype t252;
	dtype t257;
	dtype t259;
	dtype t27;
	dtype t271;
	dtype t28;
	dtype t281;
	dtype t293;
	dtype t294;
	dtype t296;
	dtype t299;
	dtype t3;
	dtype t30;
	dtype t302;
	dtype t303;
	dtype t305;
	dtype t306;
	dtype t309;
	dtype t324;
	dtype t325;
	dtype t327;
	dtype t330;
	dtype t331;
	dtype t347;
	dtype t35;
	dtype t350;
	dtype t37;
	dtype t4;
	dtype t43;
	dtype t49;
	dtype t5;
	dtype t51;
	dtype t52;
	dtype t54;
	dtype t56;
	dtype t6;
	dtype t61;
	dtype t65;
	dtype t7;
	dtype t70;
	dtype t75;
	dtype t76;
	dtype t81;
	dtype t82;
	dtype t87;
	dtype t88;
	dtype t9;
	dtype t93;
	dtype t94;
	dtype t98;
	{
		t1 = a[0];
		t2 = v[0];
		t3 = t2*t2;
		t4 = v[1];
		t5 = t4*t4;
		t6 = v[2];
		t7 = t6*t6;
		t9 = sqrt(1.0 - t3 - t5 - t7);
		t10 = 1 / t9;
		t11 = qr0[1];
		t12 = t11*t10;
		t14 = qr0[0];
		t15 = -t12*t2 + t14;
		t16 = M[0];
		t18 = qr0[2];
		t19 = t18*t10;
		t21 = qr0[3];
		t22 = -t19*t2 - t21;
		t23 = M[1];
		t25 = t10*t21;
		t27 = -t25*t2 + t18;
		t28 = M[2];
		t30 = -t15*t16 - t22*t23 - t27*t28;
		t35 = -t9*t11 - t2*t14 - t4*t21 + t6*t18;
		t37 = -t35;
		t43 = t9*t18 + t4*t14 + t6*t11 - t2*t21;
		t49 = t9*t21 + t6*t14 + t2*t18 - t11*t4;
		t51 = -t37*t16 - t43*t23 - t49*t28;
		t52 = -t15;
		t54 = t10*t14;
		t56 = -t54*t2 - t11;
		t61 = t9*t14 - t2*t11 - t4*t18 - t6*t21;
		t65 = t61*t16 + t43*t28 - t23*t49;
		t70 = t56*t16 + t22*t28 - t23*t27;
		t75 = t56*t23 + t27*t16 - t28*t15;
		t76 = -t49;
		t81 = t61*t23 + t49*t16 - t37*t28;
		t82 = -t27;
		t87 = t56*t28 + t23*t15 - t22*t16;
		t88 = -t43;
		t93 = t61*t28 + t37*t23 - t43*t16;
		t94 = -t22;
		t98 = a[4];
		t107 = t30*t88 + t94*t51 + t56*t81 + t61*t75 + t87*t35 + t93*t52 - t70*t76 - t82*t65;
		t109 = a[1];
		t118 = t30*t76 + t82*t51 + t56*t93 + t61*t87 + t70*t88 + t65*t94 - t35*t75 - t81*t52;
		t126 = t76*t51 + t61*t93 + t65*t88 - t81*t35 + t[2];
		t127 = 1 / t126;
		t141 = t51*t88 + t61*t81 + t93*t35 - t65*t76 + t[1];
		t145 = t126*t126;
		t146 = 1 / t145;
		t147 = (t1*(t35*t51 + t61*t65 + t81*t76 - t93*t88 + t[0]) + t98*t141 + t126*t109)*t146;
		jacmRT[0][0] = (t1*(t30*t35 + t52*t51 + t56*t65 + t61*t70 + t76*t75 + t81*t82 - t88*t87
			- t93*t94) + t98*t107 + t109*t118)*t127 - t118*t147;
		t150 = t1*a[3];
		t152 = a[2];
		t159 = (t150*t141 + t126*t152)*t146;
		jacmRT[1][0] = (t107*t150 + t152*t118)*t127 - t159*t118;
		t162 = -t12*t4 + t21;
		t165 = -t19*t4 + t14;
		t168 = -t25*t4 - t11;
		t170 = -t162*t16 - t165*t23 - t168*t28;
		t172 = -t162;
		t175 = -t54*t4 - t18;
		t180 = t175*t16 + t165*t28 - t168*t23;
		t185 = t175*t23 + t168*t16 - t162*t28;
		t187 = -t168;
		t192 = t175*t28 + t162*t23 - t165*t16;
		t194 = -t165;
		t206 = t170*t88 + t51*t194 + t175*t81 + t61*t185 + t192*t35 + t93*t172 - t76*t180 - t65*
			t187;
		t216 = t170*t76 + t51*t187 + t93*t175 + t61*t192 + t180*t88 + t65*t194 - t185*t35 - t81*
			t172;
		jacmRT[0][1] = (t1*(t170*t35 + t172*t51 + t175*t65 + t180*t61 + t185*t76 + t81*t187 -
			t192*t88 - t93*t194) + t98*t206 + t109*t216)*t127 - t147*t216;
		jacmRT[1][1] = (t150*t206 + t152*t216)*t127 - t159*t216;
		t227 = -t12*t6 - t18;
		t230 = -t19*t6 + t11;
		t233 = -t25*t6 + t14;
		t235 = -t227*t16 - t23*t230 - t233*t28;
		t237 = -t227;
		t240 = -t54*t6 - t21;
		t245 = t240*t16 + t230*t28 - t233*t23;
		t250 = t23*t240 + t233*t16 - t227*t28;
		t252 = -t233;
		t257 = t240*t28 + t227*t23 - t230*t16;
		t259 = -t230;
		t271 = t235*t88 + t51*t259 + t81*t240 + t61*t250 + t257*t35 + t93*t237 - t245*t76 - t65*
			t252;
		t281 = t235*t76 + t51*t252 + t240*t93 + t61*t257 + t245*t88 + t259*t65 - t250*t35 - t81*
			t237;
		jacmRT[0][2] = (t1*(t235*t35 + t237*t51 + t240*t65 + t61*t245 + t250*t76 + t81*t252 -
			t257*t88 - t93*t259) + t271*t98 + t281*t109)*t127 - t147*t281;
		jacmRT[1][2] = (t150*t271 + t281*t152)*t127 - t159*t281;
		jacmRT[0][3] = t127*t1;
		jacmRT[1][3] = 0.0;
		jacmRT[0][4] = t98*t127;
		jacmRT[1][4] = t150*t127;
		jacmRT[0][5] = t109*t127 - t147;
		jacmRT[1][5] = t152*t127 - t159;
		t293 = t35*t35;
		t294 = t61*t61;
		t296 = t88*t88;
		t299 = t35*t88;
		t302 = t61*t76;
		t303 = 2.0*t299 + t61*t49 - t302;
		t305 = t35*t76;
		t306 = t61*t88;
		t309 = t305 + 2.0*t306 - t49*t35;
		jacmS[0][0] = (t1*(t293 + t294 + t49*t76 - t296) + t98*t303 + t109*t309)*t127 - t147*
			t309;
		jacmS[1][0] = (t150*t303 + t152*t309)*t127 - t159*t309;
		t324 = t76*t76;
		t325 = t296 + t294 + t35*t37 - t324;
		t327 = t76*t88;
		t330 = t61*t35;
		t331 = 2.0*t327 + t61*t37 - t330;
		jacmS[0][1] = (t1*(t299 + 2.0*t302 - t37*t88) + t98*t325 + t109*t331)*t127 - t147*
			t331;
		jacmS[1][1] = (t150*t325 + t152*t331)*t127 - t159*t331;
		t347 = t327 + 2.0*t330 - t43*t76;
		t350 = t324 + t294 + t43*t88 - t293;
		jacmS[0][2] = (t1*(2.0*t305 + t61*t43 - t306) + t98*t347 + t350*t109)*t127 - t147*
			t350;
		jacmS[1][2] = (t150*t347 + t152*t350)*t127 - t159*t350;
		return;
	}
}
