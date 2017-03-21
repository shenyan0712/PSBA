#include "stdafx.h"

#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "readparams.h"
#include "psba.h"

#define NOCOV     0
#define FULLCOV   1
#define TRICOV    2

#define MAXSTRLEN  2048 /* 2K */

/* get rid of the rest of a line upto \n or EOF */
inline void SKIP_LINE(FILE *fp){
	char buf[MAXSTRLEN];															\
	while (!feof(fp))																\
		if (!fgets(buf, MAXSTRLEN - 1, fp) || buf[strlen(buf) - 1] == '\n') break;   \
}




/* 
读取相机的数量
* 相机文件中的每一行对应一个相机
*/
static int findNcameras(FILE *fp)
{
	int lineno, ncams, ch;

	lineno = ncams = 0;
	while (!feof(fp)) {
		if ((ch = fgetc(fp)) == '#') { /* skip comments */
			SKIP_LINE(fp);
			++lineno;
			continue;
		}
		if (feof(fp)) break;
		ungetc(ch, fp);
		SKIP_LINE(fp);
		++lineno;
		if (ferror(fp)) {
			fprintf(stderr, "findNcameras(): error reading input file, line %d\n", lineno);
			exit(1);
		}
		++ncams;
	}
	return ncams;
}


/* reads (from "fp") "nvals" doubles into "vals".
* Returns number of doubles actually read, EOF on end of file, EOF-1 on error
*/
static int readNDoubles(FILE *fp, dtype *vals, int nvals)
{
	register int i;
	int n, j;

	for (i = n = 0; i<nvals; ++i) {
		if(sizeof(dtype)==4)
			j = fscanf_s(fp, "%f", vals + i);
		else 
			j = fscanf_s(fp, "%lf", vals + i);
		if (j == EOF) return EOF;

		if (j != 1 || ferror(fp)) return EOF - 1;

		n += j;
	}

	return n;
}

/* 
reads (from "fp") "nvals" doubles without storing them.
* Returns EOF on end of file, EOF-1 on error
*/
static int skipNDoubles(FILE *fp, int nvals)
{
	register int i;
	int j;

	for (i = 0; i<nvals; ++i) {
		j = fscanf_s(fp, "%*f");
		if (j == EOF) return EOF;

		if (ferror(fp)) return EOF - 1;
	}

	return nvals;
}

/* 
读取nvals个整数到vals中。
返回实际读取的整数的个数，或EOF/EOF-1
*/
static int readNInts(FILE *fp, int *vals, int nvals)
{
	register int i;
	int n, j;

	for (i = n = 0; i<nvals; ++i) {
		j = fscanf_s(fp, "%d", vals + i);
		if (j == EOF) return EOF;

		if (j != 1 || ferror(fp)) return EOF - 1;

		n += j;
	}

	return n;
}

/* 
返回文本文件中第一行的double数值的数量. rewinds file.
*/
static int countNDoubles(FILE *fp)
{
	int lineno, ch, np, i;
	char buf[MAXSTRLEN], *s;
	dtype dummy;
	double tmp;

	lineno = 0;
	while (!feof(fp)) {
		if ((ch = fgetc(fp)) == '#') { /* skip comments */
			SKIP_LINE(fp);
			++lineno;
			continue;
		}
		if (feof(fp)) return 0;
		ungetc(ch, fp);
		++lineno;
		if (!fgets(buf, MAXSTRLEN - 1, fp)) { /* read the line found... */
			fprintf(stderr, "countNDoubles(): error reading input file, line %d\n", lineno);
			exit(1);
		}
		/* ...and count the number of doubles it has */
		for (np = i = 0, s = buf; 1; ++np, s += i) {
			if (sizeof(dtype) == 4) {
				ch = sscanf_s(s, "%lf%n", &tmp, &i);
				dummy = (dtype)tmp;
			}
			else
				ch = sscanf_s(s, "%lf%n", &dummy, &i);
			if (ch == 0 || ch == EOF) break;
		}
		rewind(fp);
		return np;
	}
	return 0; // should not reach this point
}


/*
读取相机参数,
输入参数：
	cnp				-->相机参数维度
	infilter函数	-->用于将四元组变为旋转矩阵
	filecnp			-->
输出：
	params			-->存放各相机的参数

*/
static void readCameraParams(FILE *fp, int cnp,
	void(*infilter)(dtype *pin, int nin, dtype *pout, int nout), int filecnp,
	dtype *params, dtype *initrot)
{
	int lineno, n, ch;
	dtype *tofilter;

	if (filecnp>0 && infilter) {
		if ((tofilter = (dtype *)malloc(filecnp * sizeof(dtype))) == NULL) {
			;
			fprintf(stderr, "memory allocation failed in readCameraParams()\n");
			exit(1);
		}
	}
	else { // camera params will be used as read
		infilter = NULL;
		tofilter = NULL;
		filecnp = cnp;
	}
	/* make sure that the parameters file contains the expected number of parameters per line */
	if ((n = countNDoubles(fp)) != filecnp) {
		fprintf(stderr, "readCameraParams(): expected %d camera parameters, first line contains %d!\n", filecnp, n);
		exit(1);
	}

	lineno = 0;
	while (!feof(fp)) {
		if ((ch = fgetc(fp)) == '#') { /* skip comments */
			SKIP_LINE(fp);
			++lineno;
			continue;
		}

		if (feof(fp)) break;

		ungetc(ch, fp);
		++lineno;
		if (infilter) {
			n = readNDoubles(fp, tofilter, filecnp);
			(*infilter)(tofilter, filecnp, params, cnp);
		}
		else
			n = readNDoubles(fp, params, cnp);
		if (n == EOF) break;
		if (n != filecnp) {
			fprintf(stderr, "readCameraParams(): line %d contains %d parameters, expected %d!\n", lineno, n, filecnp);
			exit(1);
		}
		if (ferror(fp)) {
			fprintf(stderr, "findNcameras(): error reading input file, line %d\n", lineno);
			exit(1);
		}

		/* save rotation assuming the last 3 parameters correspond to translation */
		initrot[1] = params[cnp - 6];
		initrot[2] = params[cnp - 5];
		initrot[3] = params[cnp - 4];
		initrot[0] = sqrt(1.0 - initrot[1] * initrot[1] - initrot[2] * initrot[2] - initrot[3] * initrot[3]);

		params += cnp;
		initrot += FULLQUATSZ;
	}
	if (tofilter) free(tofilter);
}


/*
点数据文件中，第一行对应一个3D点，以及其在各相机上的投影点。
* X_0...X_{pnp-1}  nframes  frame0 x0 y0 [cov0]  frame1 x1 y1 [cov1] ...
* The portion of the line starting at "frame0" is ignored for all but the first line
输入参数：
	fp			-->点数据文件
	mnp			-->2D点的维度，一般为2
输出：
	n3Dpts		-->总3D点数量
	nprojs		-->总投影2D点数量
	havecov		-->是否有协方差
*/
static void readNpointsAndNprojections(FILE *fp, int *n3Dpts, int pnp, int *nprojs, int mnp, int *havecov)
{
	int nfirst, lineno, npts, nframes, ch, n;

	/* #parameters for the first line */
	nfirst = countNDoubles(fp);
	*havecov = NOCOV;

	*n3Dpts = *nprojs = lineno = npts = 0;
	while (!feof(fp)) {
		if ((ch = fgetc(fp)) == '#') { /* 如果该行以#开头，跳过该行 */
			SKIP_LINE(fp);
			++lineno;
			continue;
		}
		if (feof(fp)) break;
		ungetc(ch, fp);
		++lineno;
		skipNDoubles(fp, pnp);	//跳过pnp个double值
		n = readNInts(fp, &nframes, 1);	//读取3D点对应帧数量
		if (n != 1) {
			fprintf(stderr, "readNpointsAndNprojections(): error reading input file, line %d: "
				"expecting number of frames for 3D point\n", lineno);
			exit(1);
		}
		if (npts == 0) { /* 检查第一行，看是否有协方差 */
			nfirst -= (pnp + 1); /* 减去pnp和帧数项，剩下的是帧编号及帧上对应2D点 */
			if (nfirst == nframes*(mnp + 1 + mnp*mnp)) { /* full mnpxmnp covariance */
				*havecov = FULLCOV;
			}
			else if (nfirst == nframes*(mnp + 1 + mnp*(mnp + 1) / 2)) { /* 三角形式矩阵的协方差 */
				*havecov = TRICOV;
			}
			else {
				*havecov = NOCOV;
			}
		}
		SKIP_LINE(fp);
		*nprojs += nframes;		//总投影点数累加
		++npts;
	}

	*n3Dpts = npts;
}


/* reads the number of (double) parameters contained in the first non-comment row of a file */
int readNumParams(char *fname)
{
	FILE *fp;
	int n;

	if ( fopen_s(&fp,fname, "r") != 0) {
		fprintf(stderr, "error opening file %s!\n", fname);
		exit(1);
	}

	n = countNDoubles(fp);
	fclose(fp);

	return n;
}

/* 
从点数据文件读取3D点及对应投影点
输入：


输出：
	parmas		-->3D点
	projs		-->2D投影点

* "params", "projs" & "vmask" are assumed preallocated, pointing to
* memory blocks large enough to hold the parameters of 3D points,
* their projections in all images and the point visibility mask, respectively.
* Also, if "covprojs" is non-NULL, it is assumed preallocated and pointing to
* a memory block suitable to hold the covariances of image projections.
* Each 3D point is assumed to be defined by pnp parameters and each of its projections
* by mnp parameters. Optionally, the mnp*mnp covariance matrix in row-major order
* follows each projection. All parameters are stored in a single line.
*
* 文件格式 is X_{0}...X_{pnp-1}  nframes  frame0 x0 y0 [covx0^2 covx0y0 covx0y0 covy0^2] frame1 x1 y1 [covx1^2 covx1y1 covx1y1 covy1^2] ...
* with the parameters in angle brackets being optional. To save space, only the upper
* triangular part of the covariance can be specified, i.e. [covx0^2 covx0y0 covy0^2], etc
*/
static void readPointParamsAndProjections(FILE *fp, dtype *params, int pnp, dtype *projs, dtype *covprojs,
	int havecov, int mnp, char *vmask, int ncams)
{
	int nframes, ch, lineno, ptno, frameno, n;
	int ntord, covsz = mnp*mnp, tricovsz = mnp*(mnp + 1) / 2, nshift;
	register int i, ii, jj, k;

	lineno = ptno = 0;
	while (!feof(fp)) {
		if ((ch = fgetc(fp)) == '#') { /* skip comments */
			SKIP_LINE(fp);
			lineno++;
			continue;
		}
		if (feof(fp)) break;
		ungetc(ch, fp);
		n = readNDoubles(fp, params, pnp); /* read in point parameters */
		if (n == EOF) break;
		if (n != pnp) {
			fprintf(stderr, "readPointParamsAndProjections(): error reading input file, line %d:\n"
				"expecting %d parameters for 3D point, read %d\n", lineno, pnp, n);
			exit(1);
		}
		params += pnp;

		n = readNInts(fp, &nframes, 1);  /* read in number of image projections */
		if (n != 1) {
			fprintf(stderr, "readPointParamsAndProjections(): error reading input file, line %d:\n"
				"expecting number of frames for 3D point\n", lineno);
			exit(1);
		}

		for (i = 0; i<nframes; ++i) {
			n = readNInts(fp, &frameno, 1); /* read in frame number... */
			if (frameno >= ncams) {
				fprintf(stderr, "readPointParamsAndProjections(): line %d contains an image projection for frame %d "
					"but only %d cameras have been specified!\n", lineno + 1, frameno, ncams);
				exit(1);
			}

			n += readNDoubles(fp, projs, mnp); /* ...and image projection */
			projs += mnp;
			if (n != mnp + 1) {
				fprintf(stderr, "readPointParamsAndProjections(): error reading image projections from line %d [n=%d].\n"
					"Perhaps line contains fewer than %d projections?\n", lineno + 1, n, nframes);
				exit(1);
			}

			if (covprojs != NULL) {
				if (havecov == TRICOV) {
					ntord = tricovsz;
				}
				else {
					ntord = covsz;
				}
				n = readNDoubles(fp, covprojs, ntord); /* read in covariance values */
				if (n != ntord) {
					fprintf(stderr, "readPointParamsAndProjections(): error reading image projection covariances from line %d [n=%d].\n"
						"Perhaps line contains fewer than %d projections?\n", lineno + 1, n, nframes);
					exit(1);
				}
				if (havecov == TRICOV) {
					/* complete the full matrix from the triangular part that was read.
					* First, prepare upper part: element (ii, mnp-1) is at position mnp-1 + ii*(2*mnp-ii-1)/2.
					* Row ii has mnp-ii elements that must be shifted by ii*(ii+1)/2
					* positions to the right to make space for the lower triangular part
					*/
					for (ii = mnp; --ii; ) {
						k = mnp - 1 + ((ii*((mnp << 1) - ii - 1)) >> 1); //mnp-1 + ii*(2*mnp-ii-1)/2
						nshift = (ii*(ii + 1)) >> 1; //ii*(ii+1)/2;
						for (jj = 0; jj<mnp - ii; ++jj) {
							covprojs[k - jj + nshift] = covprojs[k - jj];
							//covprojs[k-jj]=0.0; // this clears the lower part
						}
					}
					/* copy to lower part */
					for (ii = mnp; ii--; )
						for (jj = ii; jj--; )
							covprojs[ii*mnp + jj] = covprojs[jj*mnp + ii];
				}
				covprojs += covsz;
			}

			vmask[ptno*ncams + frameno] = 1;
		}

		fscanf_s(fp, "\n"); // consume trailing newline

		lineno++;
		ptno++;
	}
}



/* 
结合前面的函数，从txt文件读取初始的SfM信息
同时，也读取3D点的图像投影，方差（可选，不使用则为NULL）
后面4个参数需要动态地分配内存。

输入：
	camsfname	-->相机参数文件
	ptsfname	-->3D点及对应2D投影点的文件
	infilter	-->将四元组变为旋转矩阵的函数
	cnp,pnp,mnp	-->分别表示相机参数维度，3D点维度，以及2D点维度
	filecnp		-->指定相机参数文件中每个相机的维度
输出：
	ncams		-->相机数量
	n3Dpts		-->3D点数量
	n2Dprojs	-->2D投影点数量

*/
void readInitialSBAEstimate(char *camsfname, char *ptsfname, int cnp, int pnp, int mnp,
	void(*infilter)(dtype *pin, int nin, dtype *pout, int nout), int filecnp,
	int *ncams, int *n3Dpts, int *n2Dprojs,
	dtype **motstruct, dtype **initrot, dtype **imgpts, dtype **covimgpts, char **vmask)
{
	FILE *fpc, *fpp;
	int havecov;

	//读取相机文件和点文件
	if (fopen_s(&fpc,camsfname, "r") != 0) {
		fprintf(stderr, "cannot open file %s, exiting\n", camsfname);
		system("pause");
		exit(1);
	}
	if (fopen_s(&fpp, ptsfname, "r") != 0) {
		fprintf(stderr, "cannot open file %s, exiting\n", ptsfname);
		system("pause");
		exit(1);
	}
	//读取相机的数量
	*ncams = findNcameras(fpc);
	//读取3D点以及2D投影点的数量到n3Dpts，n2Dprojs
	readNpointsAndNprojections(fpp, n3Dpts, pnp, n2Dprojs, mnp, &havecov);

	//为相机参数和3D点分配存储器
	*motstruct = (dtype *)malloc((*ncams*cnp + *n3Dpts*pnp) * sizeof(dtype));
	if (*motstruct == NULL) {
		fprintf(stderr, "memory allocation for 'motstruct' failed in readInitialSBAEstimate()\n");
		system("pause");
		exit(1);
	}
	//为相机四元数分配存储器
	*initrot = (dtype *)malloc((*ncams*FULLQUATSZ) * sizeof(dtype)); // Note: this assumes quaternions for rotations!
	if (*initrot == NULL) {
		fprintf(stderr, "memory allocation for 'initrot' failed in readInitialSBAEstimate()\n");
		system("pause");
		exit(1);
	}
	//为所有2D点分配存储器
	*imgpts = (dtype *)malloc(*n2Dprojs*mnp * sizeof(dtype));
	if (*imgpts == NULL) {
		fprintf(stderr, "memory allocation for 'imgpts' failed in readInitialSBAEstimate()\n");
		system("pause");
		exit(1);
	}
	if (havecov) {
		*covimgpts = (dtype *)malloc(*n2Dprojs*mnp*mnp * sizeof(dtype));
		if (*covimgpts == NULL) {
			fprintf(stderr, "memory allocation for 'covimgpts' failed in readInitialSBAEstimate()\n");
			system("pause");
			exit(1);
		}
	}
	else
		*covimgpts = NULL;
	//为点掩码分配存储器，因为一个3D点不是能投影到所有图像帧上。
	*vmask = (char *)malloc(*n3Dpts * *ncams * sizeof(char));
	if (*vmask == NULL) {
		fprintf(stderr, "memory allocation for 'vmask' failed in readInitialSBAEstimate()\n");
		system("pause");
		exit(1);
	}
	memset(*vmask, 0, *n3Dpts * *ncams * sizeof(char)); /* clear vmask */

	/* prepare for re-reading files */
	rewind(fpc);
	rewind(fpp);

	//读取相机参数到 motstruct和initrot
	readCameraParams(fpc, cnp, infilter, filecnp, *motstruct, *initrot);
	//读取3D点到motstruct的3D存储部分，读取2D点及掩码到imgpts和vmask中。如有2D点的协方差读取到covimgpts中。
	readPointParamsAndProjections(fpp, *motstruct + *ncams*cnp, pnp, *imgpts, *covimgpts, havecov, mnp, *vmask, *ncams);

	fclose(fpc);
	fclose(fpp);
}