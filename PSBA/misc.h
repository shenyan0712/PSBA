#pragma once

#include "psba.h"

#define emalloc(sz)       emalloc_(__FILE__, __LINE__, sz)


void quat2vec(dtype *inp, int nin, dtype *outp, int nout);

void vec2quat(dtype *inp, int nin, dtype *outp, int nout);


void camsFormatTrans(dtype *in, int nin, int dim, int stubForTrans, dtype *out);
void quat2matrix(dtype *in, int nin, dtype *out);

void *emalloc_(char *file, int line, size_t sz);

dtype compute_L2_sq(int n, dtype *x);

dtype dotProduct(dtype *a, dtype *b, int size);

void generate_idxs(int nCams, int n3Dpts, int n2Dprojs, dtype *impts_data, char *vmask,
	int *comm3DIdx, int *comm3DIdxCnt, int *iidx, int *jidx, int *blk_idx);