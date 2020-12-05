#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <stdarg.h>
#include "../src/mb.h"
#include "helper_cuda.h"

#define MAX_GPU_COUNT      2
#define cutilCheckMsg(a) getLastCudaError(a)
#define cutilSafeCall(a) cudaError(a)
extern int	proc_id;

extern int	augmentData;
extern CLFlt	*chainCondLikes;
extern Chain	chainParams;
extern int	condLikeRowSize;
extern int	globalinvCondLikeSize;
extern int	globaln;
extern int	globalnNodes;
extern int	globalnScalerNodes;
extern int	globaloneMatSize;
extern MrBFlt	*globallnL;
extern MrBFlt	*invCondLikes;
extern ModelInfo	modelSettings[MAX_NUM_DIVS];
extern CLFlt	*nodeScalerSpace;
extern int	numCompressedChars;
extern int	numCurrentDivisions;
extern int	numLocalChains;
extern int	numLocalTaxa;
extern CLFlt	*numSitesOfPat;
extern CLFlt	*termCondLikes;
extern int	*termState;
extern int	tiProbRowSize;
extern CLFlt	*tiProbSpace;
extern CLFlt	*treeScalerSpace;

static CLFlt	*devchainCondLikes;
static MrBFlt	*devinvCondLikes;
static MrBFlt	*devlnL;
static CLFlt	*devnodeScalerSpace;
static CLFlt	*devnumSitesOfPat;

static int	*devtermState;
static CLFlt	*devtiProbSpace;
static CLFlt	*devtreeScalerSpace;

int	*modeldevlnLIdx;
int	sizeofdevlnL;
cudaStream_t	*stream;

#define inter_task 1500

__device__ CLFlt clScaler(CLFlt *s_clP)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    CLFlt scaler;
    CLFlt s0, s1;

    __shared__ CLFlt s_temp[12][4];

    s_temp[tid_z][tid_y] = 0.0f;

    int idx = 16*tid_z + 4*tid_y;

    s0 = MAX(s_clP[idx+0], s_clP[idx+1]);
    s1 = MAX(s_clP[idx+2], s_clP[idx+3]);

    s_temp[tid_z][tid_y] = MAX(s0,s1);

    s0 = MAX(s_temp[tid_z][0], s_temp[tid_z][1]);
    s1 = MAX(s_temp[tid_z][2], s_temp[tid_z][3]);

    scaler = MAX(s0, s1);

    s_clP[16*tid_z + 4 * tid_y + tid_x] /= scaler;

    scaler = (CLFlt)log(scaler);

    return scaler;
}

__device__ void likelihood_NUC4_hasPInvar_YES_e(MrBFlt like, MrBFlt *clInvar, CLFlt lnScaler, CLFlt *nSitesOfPat, MrBFlt *lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int char_idx)
{
    MrBFlt lnLike;
    MrBFlt likeI = clInvar[0]*bs_A + clInvar[1]*bs_C + clInvar[2]*bs_G + clInvar[3]*bs_T;

    like *= freq;
    likeI *= pInvar;

    if(lnScaler < -200)
    {
        if(likeI > 1E-70)
            lnLike = log(likeI);
        else
            lnLike = log(like) + lnScaler;
    }
    else
    {
        if(likeI == 0.0)
            lnLike = log(like) + lnScaler;
        else
            lnLike = log(like + (likeI/exp(lnScaler))) + lnScaler;
    }

    lnL[char_idx] = lnLike * nSitesOfPat[char_idx];
}

__global__ void gpu_down_0_0(CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *tiPL, CLFlt *tiPR, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int char_idx = 12 * blockIdx.x + tid_z;
    int clx_idx = 16 * char_idx + 4 * tid_y + tid_x;
    int sclx_idx = 16 * tid_z + 4 * tid_y + tid_x;
    int stipx_idx = sclx_idx;

    __shared__ CLFlt s_clL[192];
    __shared__ CLFlt s_clR[192];
    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];

    if(stipx_idx < 64)
    {
        s_tiPL[stipx_idx] = tiPL[stipx_idx];
        s_tiPR[stipx_idx] = tiPR[stipx_idx];
    }

    if(char_idx < modelnumChars)
    {
        s_clL[sclx_idx] = clL[clx_idx];
        s_clR[sclx_idx] = clR[clx_idx];

        __syncthreads();

        sclx_idx = 16 * tid_z + 4 * tid_y;
        stipx_idx = 16 * tid_y + 4 * tid_x;

        clP[clx_idx] =
                (s_tiPL[stipx_idx+0]*s_clL[sclx_idx+0] +
                 s_tiPL[stipx_idx+1]*s_clL[sclx_idx+1] +
                 s_tiPL[stipx_idx+2]*s_clL[sclx_idx+2] +
                 s_tiPL[stipx_idx+3]*s_clL[sclx_idx+3]) *
                (s_tiPR[stipx_idx+0]*s_clR[sclx_idx+0] +
                 s_tiPR[stipx_idx+1]*s_clR[sclx_idx+1] +
                 s_tiPR[stipx_idx+2]*s_clR[sclx_idx+2] +
                 s_tiPR[stipx_idx+3]*s_clR[sclx_idx+3]);
    }
}

__global__ void gpu_down_0_1 (CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *tiPL, CLFlt *tiPR, CLFlt *lnScaler, CLFlt *scPOld, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int char_idx = 12 * blockIdx.x + tid_z;
    int clx_idx = 16 * char_idx + 4 * tid_y + tid_x;
    int sclx_idx = 16 * tid_z + 4 * tid_y + tid_x;
    int stipx_idx = sclx_idx;

    __shared__ CLFlt s_clL[192];
    __shared__ CLFlt s_clR[192];
    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];

    if(stipx_idx < 64)
    {
        s_tiPL[stipx_idx] = tiPL[stipx_idx];
        s_tiPR[stipx_idx] = tiPR[stipx_idx];
    }

    if(char_idx < modelnumChars)
    {
        s_clL[sclx_idx] = clL[clx_idx];
        s_clR[sclx_idx] = clR[clx_idx];

        __syncthreads();

        sclx_idx = 16 * tid_z + 4 * tid_y;
        stipx_idx = 16 * tid_y + 4 * tid_x;

        clP[clx_idx] =
                (s_tiPL[stipx_idx+0]*s_clL[sclx_idx+0] +
                 s_tiPL[stipx_idx+1]*s_clL[sclx_idx+1] +
                 s_tiPL[stipx_idx+2]*s_clL[sclx_idx+2] +
                 s_tiPL[stipx_idx+3]*s_clL[sclx_idx+3]) *
                (s_tiPR[stipx_idx+0]*s_clR[sclx_idx+0] +
                 s_tiPR[stipx_idx+1]*s_clR[sclx_idx+1] +
                 s_tiPR[stipx_idx+2]*s_clR[sclx_idx+2] +
                 s_tiPR[stipx_idx+3]*s_clR[sclx_idx+3]);

        if((!tid_x)&&(!tid_y))
            lnScaler[char_idx] -= scPOld[char_idx];
    }
}

__global__ void gpu_down_0_2(CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *tiPL, CLFlt *tiPR, CLFlt *lnScaler, CLFlt *scPNew, int  modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int char_idx = 12 * blockIdx.x + tid_z;
    int clx_idx = 16 * char_idx + 4 * tid_y + tid_x;
    int sclx_idx = 16 * tid_z + 4 * tid_y + tid_x;
    int stipx_idx = sclx_idx;

    __shared__ CLFlt s_clL[192];
    __shared__ CLFlt s_clR[192];
    __shared__ CLFlt s_clP[192];
    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];

    if(stipx_idx < 64)
    {
        s_tiPL[stipx_idx] = tiPL[stipx_idx];
        s_tiPR[stipx_idx] = tiPR[stipx_idx];
    }

    if(char_idx < modelnumChars)
    {
        s_clL[sclx_idx] = clL[clx_idx];
        s_clR[sclx_idx] = clR[clx_idx];

        __syncthreads();

        int sclp_idx = sclx_idx;

        sclx_idx = 16 * tid_z + 4 * tid_y;
        stipx_idx = 16 * tid_y + 4 * tid_x;

        s_clP[sclp_idx] =
                (s_tiPL[stipx_idx+0]*s_clL[sclx_idx+0] +
                 s_tiPL[stipx_idx+1]*s_clL[sclx_idx+1] +
                 s_tiPL[stipx_idx+2]*s_clL[sclx_idx+2] +
                 s_tiPL[stipx_idx+3]*s_clL[sclx_idx+3]) *
                (s_tiPR[stipx_idx+0]*s_clR[sclx_idx+0] +
                 s_tiPR[stipx_idx+1]*s_clR[sclx_idx+1] +
                 s_tiPR[stipx_idx+2]*s_clR[sclx_idx+2] +
                 s_tiPR[stipx_idx+3]*s_clR[sclx_idx+3]);

        CLFlt new_lnScaler;
        new_lnScaler = clScaler(s_clP);

        if((!tid_x)&&(!tid_y))
        {
            scPNew[char_idx] = new_lnScaler;
            lnScaler[char_idx] += new_lnScaler;
        }

        clP[clx_idx] = s_clP[sclp_idx];
    }
}

__global__ void gpu_down_0_3(CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *tiPL, CLFlt *tiPR, CLFlt *lnScaler, CLFlt *scPOld, CLFlt *scPNew, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int char_idx = 12 * blockIdx.x + tid_z;
    int clx_idx = 16 * char_idx + 4 * tid_y + tid_x;
    int sclx_idx = 16 * tid_z + 4 * tid_y + tid_x;
    int stipx_idx = sclx_idx;

    __shared__ CLFlt s_clL[192];
    __shared__ CLFlt s_clR[192];
    __shared__ CLFlt s_clP[192];
    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];

    if(stipx_idx < 64)
    {
        s_tiPL[stipx_idx] = tiPL[stipx_idx];
        s_tiPR[stipx_idx] = tiPR[stipx_idx];
    }

    if(char_idx < modelnumChars)
    {
        s_clL[sclx_idx] = clL[clx_idx];
        s_clR[sclx_idx] = clR[clx_idx];

        __syncthreads();

        int sclp_idx = sclx_idx;

        sclx_idx = 16 * tid_z + 4 * tid_y;
        stipx_idx = 16 * tid_y + 4 * tid_x;

        s_clP[sclp_idx] =
                (s_tiPL[stipx_idx+0]*s_clL[sclx_idx+0] +
                 s_tiPL[stipx_idx+1]*s_clL[sclx_idx+1] +
                 s_tiPL[stipx_idx+2]*s_clL[sclx_idx+2] +
                 s_tiPL[stipx_idx+3]*s_clL[sclx_idx+3]) *
                (s_tiPR[stipx_idx+0]*s_clR[sclx_idx+0] +
                 s_tiPR[stipx_idx+1]*s_clR[sclx_idx+1] +
                 s_tiPR[stipx_idx+2]*s_clR[sclx_idx+2] +
                 s_tiPR[stipx_idx+3]*s_clR[sclx_idx+3]);

        CLFlt new_lnScaler;
        new_lnScaler = clScaler(s_clP);

        if((!tid_x)&&(!tid_y))
        {
            scPNew[char_idx] = new_lnScaler;
            CLFlt dif = lnScaler[char_idx];
            dif -= scPOld[char_idx];
            dif += new_lnScaler;
            lnScaler[char_idx] = dif;
        }

        clP[clx_idx] = s_clP[sclp_idx];
    }
}

__global__ void gpu_down_0_3_e(CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *tiPL, CLFlt *tiPR, CLFlt *lnScaler, CLFlt *scPOld, CLFlt *scPNew, int residue)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int char_idx = 64*bid + tid;

    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];

    s_tiPL[tid] = tiPL[tid];
    s_tiPR[tid] = tiPR[tid];

    CLFlt *m_tiPL = &s_tiPL[0];
    CLFlt *m_tiPR = &s_tiPR[0];

    __shared__ CLFlt s_clL[64*16 + 4*16];
    __shared__ CLFlt s_clR[64*16 + 4*16];

    int num=64, times=16, nthread=0;
    if(bid+1 == gridDim.x && residue != 0)
    {
        num=residue;
        int i = num*16;
        times = i/64;
        nthread = i%64;
    }

    int offset = tid>>4;
    CLFlt *m_clL = &s_clL[tid+offset];
    CLFlt *m_clR = &s_clR[tid+offset];
    CLFlt *g_clL = &clL[bid*64*16+tid];
    CLFlt *g_clR = &clR[bid*64*16+tid];
    CLFlt *g_clP = &clP[bid*64*16+tid];

    int i,j,k;
    for(i=j=k=0;i<times;i++)
    {
        m_clL[k] = g_clL[j];
        m_clR[k] = g_clR[j];
        k+=68;
        j+=64;
    }
    if(tid < nthread)
    {
        m_clL[k]=g_clL[j];
        m_clR[k]=g_clR[j];
    }

    __syncthreads();

    if(tid < num)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1, scaler = 0.0;
        int gamma;

        m_clL = &s_clL[17*tid];
        m_clR = &s_clR[17*tid];

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_clL[0]; c0 = m_clL[1]; g0 = m_clL[2]; t0 = m_clL[3];

            a1 = m_tiPL[0]*a0 + m_tiPL[1]*c0 + m_tiPL[2]*g0 + m_tiPL[3]*t0;
            c1 = m_tiPL[4]*a0 + m_tiPL[5]*c0 + m_tiPL[6]*g0 + m_tiPL[7]*t0;
            g1 = m_tiPL[8]*a0 + m_tiPL[9]*c0 + m_tiPL[10]*g0 + m_tiPL[11]*t0;
            t1 = m_tiPL[12]*a0 + m_tiPL[13]*c0 + m_tiPL[14]*g0 + m_tiPL[15]*t0;

            a0 = m_clR[0]; c0 = m_clR[1]; g0 = m_clR[2]; t0 = m_clR[3];

            a1 *= m_tiPR[0]*a0 + m_tiPR[1]*c0 + m_tiPR[2]*g0 + m_tiPR[3]*t0;
            c1 *= m_tiPR[4]*a0 + m_tiPR[5]*c0 + m_tiPR[6]*g0 + m_tiPR[7]*t0;
            g1 *= m_tiPR[8]*a0 + m_tiPR[9]*c0 + m_tiPR[10]*g0 + m_tiPR[11]*t0;
            t1 *= m_tiPR[12]*a0 + m_tiPR[13]*c0 + m_tiPR[14]*g0 + m_tiPR[15]*t0;

            scaler = MAX(scaler, a1);
            scaler = MAX(scaler, c1);
            scaler = MAX(scaler, g1);
            scaler = MAX(scaler, t1);

            m_clL[0] = a1;
            m_clL[1] = c1;
            m_clL[2] = g1;
            m_clL[3] = t1;

            m_clL += 4;
            m_clR += 4;
            m_tiPL += 16;
            m_tiPR += 16;
        }

        m_clL = &s_clL[tid*17];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            m_clL[0] /= scaler;
            m_clL[1] /= scaler;
            m_clL[2] /= scaler;
            m_clL[3] /= scaler;
            m_clL += 4;
        }

        scaler = (CLFlt)log(scaler);
        scPNew[char_idx] = scaler;
        CLFlt dif = lnScaler[char_idx];
        dif -= scPOld[char_idx];
        dif += scPNew[char_idx];
        lnScaler[char_idx] = dif;
    }

    __syncthreads();

    m_clL = &s_clL[tid+offset];
    for(i=j=k=0;i<times;i++)
    {
        g_clP[j] = m_clL[k];
        k += 68;
        j += 64;
    }
    if(tid < nthread) g_clP[j]=m_clL[k];
}

__global__ void gpu_down_0_0_e(CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *tiPL, CLFlt *tiPR, int residue)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];

    s_tiPL[tid] = tiPL[tid];
    s_tiPR[tid] = tiPR[tid];

    CLFlt *m_tiPL = &s_tiPL[0];
    CLFlt *m_tiPR = &s_tiPR[0];

    __shared__ CLFlt s_clL[64*16 + 4*16];
    __shared__ CLFlt s_clR[64*16 + 4*16];

    int num=64, times=16, nthread=0;
    if(bid+1 == gridDim.x && residue != 0)
    {
        num=residue;
        int i = num*16;
        times = i/64;
        nthread = i%64;
    }

    int offset = tid>>4;
    CLFlt *m_clL = &s_clL[tid+offset];
    CLFlt *m_clR = &s_clR[tid+offset];
    CLFlt *g_clL = &clL[bid*64*16+tid];
    CLFlt *g_clR = &clR[bid*64*16+tid];
    CLFlt *g_clP = &clP[bid*64*16+tid];

    int i,j,k;
    for(i=j=k=0;i<times;i++)
    {
        m_clL[k] = g_clL[j];
        m_clR[k] = g_clR[j];
        k+=68;
        j+=64;
    }
    if(tid < nthread)
    {
        m_clL[k]=g_clL[j];
        m_clR[k]=g_clR[j];
    }

    __syncthreads();

    if(tid < num)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1;
        int gamma;
        m_clL = &s_clL[17*tid];
        m_clR = &s_clR[17*tid];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_clL[0]; c0 = m_clL[1]; g0 = m_clL[2]; t0 = m_clL[3];

            a1 = m_tiPL[0]*a0 + m_tiPL[1]*c0 + m_tiPL[2]*g0 + m_tiPL[3]*t0;
            c1 = m_tiPL[4]*a0 + m_tiPL[5]*c0 + m_tiPL[6]*g0 + m_tiPL[7]*t0;
            g1 = m_tiPL[8]*a0 + m_tiPL[9]*c0 + m_tiPL[10]*g0 + m_tiPL[11]*t0;
            t1 = m_tiPL[12]*a0 + m_tiPL[13]*c0 + m_tiPL[14]*g0 + m_tiPL[15]*t0;

            a0 = m_clR[0]; c0 = m_clR[1]; g0 = m_clR[2]; t0 = m_clR[3];

            a1 *= m_tiPR[0]*a0 + m_tiPR[1]*c0 + m_tiPR[2]*g0 + m_tiPR[3]*t0;
            c1 *= m_tiPR[4]*a0 + m_tiPR[5]*c0 + m_tiPR[6]*g0 + m_tiPR[7]*t0;
            g1 *= m_tiPR[8]*a0 + m_tiPR[9]*c0 + m_tiPR[10]*g0 + m_tiPR[11]*t0;
            t1 *= m_tiPR[12]*a0 + m_tiPR[13]*c0 + m_tiPR[14]*g0 + m_tiPR[15]*t0;

            m_clL[0] = a1;
            m_clL[1] = c1;
            m_clL[2] = g1;
            m_clL[3] = t1;

            m_clL += 4;
            m_clR += 4;
            m_tiPL += 16;
            m_tiPR += 16;
        }
    }

    __syncthreads();

    m_clL = &s_clL[tid+offset];
    for(i=j=k=0;i<times;i++)
    {
        g_clP[j] = m_clL[k];
        k += 68;
        j += 64;
    }
    if(tid < nthread) g_clP[j]=m_clL[k];
}

static void down_0_0(int offset_clP, int offset_clL, int offset_clR, int offset_pL, int offset_pR, int modelnumChars, int modelnumGammaCats, int chain)
{
    if(modelnumChars < inter_task)
    {
        int	bz = 192 / (4 * modelnumGammaCats);
        int	gx = modelnumChars / bz;
        if (modelnumChars % bz != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, modelnumGammaCats, bz);

        gpu_down_0_0<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, modelnumChars);
        cutilCheckMsg("gpu_down_0_0 failed\n");
    }
    else
    {
        int nThread = 64;
        int gx = modelnumChars / nThread;
        int residue = modelnumChars % nThread;
        if (residue != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(64, 1, 1);

        gpu_down_0_0_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, residue);

        cutilCheckMsg("gpu_down_0_0_e failed\n");
    }
}

__global__ void gpu_down_0_1_e(CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *tiPL, CLFlt *tiPR, CLFlt *lnScaler, CLFlt *scPOld, int residue)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int char_idx = 64*bid + tid;

    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];

    s_tiPL[tid] = tiPL[tid];
    s_tiPR[tid] = tiPR[tid];

    CLFlt *m_tiPL = &s_tiPL[0];
    CLFlt *m_tiPR = &s_tiPR[0];

    __shared__ CLFlt s_clL[64*16 + 4*16];
    __shared__ CLFlt s_clR[64*16 + 4*16];

    int num=64, times=16, nthread=0;
    if(bid+1 == gridDim.x && residue != 0)
    {
        num=residue;
        int i = num*16;
        times = i/64;
        nthread = i%64;
    }

    int offset = tid>>4;
    CLFlt *m_clL = &s_clL[tid+offset];
    CLFlt *m_clR = &s_clR[tid+offset];
    CLFlt *g_clL = &clL[bid*64*16+tid];
    CLFlt *g_clR = &clR[bid*64*16+tid];
    CLFlt *g_clP = &clP[bid*64*16+tid];
    int i,j,k;
    for(i=j=k=0;i<times;i++)
    {
        m_clL[k] = g_clL[j];
        m_clR[k] = g_clR[j];
        k+=68;
        j+=64;
    }
    if(tid < nthread)
    {
        m_clL[k]=g_clL[j];
        m_clR[k]=g_clR[j];
    }

    __syncthreads();

    if(tid < num)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1;
        int gamma;
        m_clL = &s_clL[17*tid];
        m_clR = &s_clR[17*tid];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_clL[0]; c0 = m_clL[1]; g0 = m_clL[2]; t0 = m_clL[3];

            a1 = m_tiPL[0]*a0 + m_tiPL[1]*c0 + m_tiPL[2]*g0 + m_tiPL[3]*t0;
            c1 = m_tiPL[4]*a0 + m_tiPL[5]*c0 + m_tiPL[6]*g0 + m_tiPL[7]*t0;
            g1 = m_tiPL[8]*a0 + m_tiPL[9]*c0 + m_tiPL[10]*g0 + m_tiPL[11]*t0;
            t1 = m_tiPL[12]*a0 + m_tiPL[13]*c0 + m_tiPL[14]*g0 + m_tiPL[15]*t0;

            a0 = m_clR[0]; c0 = m_clR[1]; g0 = m_clR[2]; t0 = m_clR[3];

            a1 *= m_tiPR[0]*a0 + m_tiPR[1]*c0 + m_tiPR[2]*g0 + m_tiPR[3]*t0;
            c1 *= m_tiPR[4]*a0 + m_tiPR[5]*c0 + m_tiPR[6]*g0 + m_tiPR[7]*t0;
            g1 *= m_tiPR[8]*a0 + m_tiPR[9]*c0 + m_tiPR[10]*g0 + m_tiPR[11]*t0;
            t1 *= m_tiPR[12]*a0 + m_tiPR[13]*c0 + m_tiPR[14]*g0 + m_tiPR[15]*t0;

            m_clL[0] = a1;
            m_clL[1] = c1;
            m_clL[2] = g1;
            m_clL[3] = t1;

            m_clL += 4;
            m_clR += 4;
            m_tiPL += 16;
            m_tiPR += 16;
        }

        CLFlt dif = lnScaler[char_idx];
        dif -= scPOld[char_idx];
        lnScaler[char_idx] = dif;
    }

    __syncthreads();

    m_clL = &s_clL[tid+offset];
    for(i=j=k=0;i<times;i++)
    {
        g_clP[j] = m_clL[k];
        k += 68;
        j += 64;
    }
    if(tid < nthread) g_clP[j]=m_clL[k];
}

static void down_0_1(int offset_clP, int offset_clL, int offset_clR, int offset_pL, int offset_pR, int offset_lnScaler, int offset_scPOld, int modelnumChars, int modelnumGammaCats, int chain)
{
    if(modelnumChars < inter_task)
    {
        int	bz = 192 / (4 * modelnumGammaCats);
        int	gx = modelnumChars / bz;
        if (modelnumChars % bz != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, modelnumGammaCats, bz);

        gpu_down_0_1<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPOld, modelnumChars);
        cutilCheckMsg("gpu_down_0_1 failed\n");
    }
    else
    {
        int nThread = 64;
        int gx = modelnumChars / nThread;
        int residue = modelnumChars % nThread;
        if (residue != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(64, 1, 1);

        gpu_down_0_1_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtreeScalerSpace+offset_lnScaler, devnodeScalerSpace+offset_scPOld, residue);
        cutilCheckMsg("gpu_down_0_1_e failed\n");
    }
}

static void down_0_3(int offset_clP, int offset_clL, int offset_clR, int offset_pL, int offset_pR, int offset_lnScaler, int offset_scPOld, int offset_scPNew, int modelnumChars, int modelnumGammaCats, int chain)
{
    if(modelnumChars < inter_task)
    {
        int	bz = 192 / (4 * modelnumGammaCats);
        int	gx = modelnumChars / bz;
        if (modelnumChars % bz != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, modelnumGammaCats, bz);

        gpu_down_0_3<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPOld, devnodeScalerSpace+offset_scPNew, modelnumChars);
        cutilCheckMsg("gpu_down_0_3 failed\n");
    }
    else
    {
        int nThread = 64;
        int gx = modelnumChars / nThread;
        int residue = modelnumChars % nThread;
        if (residue != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(64, 1, 1);

        gpu_down_0_3_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPOld, devnodeScalerSpace+offset_scPNew, residue);
        cutilCheckMsg("gpu_down_0_3_e failed\n");
    }
}

__global__ void gpu_down_0_2_e(CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *tiPL, CLFlt *tiPR, CLFlt *lnScaler, CLFlt *scPNew, int residue)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int char_idx = 64*bid + tid;

    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];

    s_tiPL[tid] = tiPL[tid];
    s_tiPR[tid] = tiPR[tid];

    CLFlt *m_tiPL = &s_tiPL[0];
    CLFlt *m_tiPR = &s_tiPR[0];

    __shared__ CLFlt s_clL[64*16 + 4*16];
    __shared__ CLFlt s_clR[64*16 + 4*16];

    int num=64, times=16, nthread=0;
    if(bid+1 == gridDim.x && residue != 0)
    {
        num=residue;
        int i = num*16;
        times = i/64;
        nthread = i%64;
    }

    int offset = tid>>4;
    CLFlt *m_clL = &s_clL[tid+offset];
    CLFlt *m_clR = &s_clR[tid+offset];
    CLFlt *g_clL = &clL[bid*64*16+tid];
    CLFlt *g_clR = &clR[bid*64*16+tid];
    CLFlt *g_clP = &clP[bid*64*16+tid];

    int i,j,k;
    for(i=j=k=0;i<times;i++)
    {
        m_clL[k] = g_clL[j];
        m_clR[k] = g_clR[j];
        k+=68;
        j+=64;
    }
    if(tid < nthread)
    {
        m_clL[k]=g_clL[j];
        m_clR[k]=g_clR[j];
    }

    __syncthreads();

    if(tid < num)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1, scaler = 0.0;
        int gamma;
        m_clL = &s_clL[17*tid];
        m_clR = &s_clR[17*tid];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_clL[0]; c0 = m_clL[1]; g0 = m_clL[2]; t0 = m_clL[3];

            a1 = m_tiPL[0]*a0 + m_tiPL[1]*c0 + m_tiPL[2]*g0 + m_tiPL[3]*t0;
            c1 = m_tiPL[4]*a0 + m_tiPL[5]*c0 + m_tiPL[6]*g0 + m_tiPL[7]*t0;
            g1 = m_tiPL[8]*a0 + m_tiPL[9]*c0 + m_tiPL[10]*g0 + m_tiPL[11]*t0;
            t1 = m_tiPL[12]*a0 + m_tiPL[13]*c0 + m_tiPL[14]*g0 + m_tiPL[15]*t0;

            a0 = m_clR[0]; c0 = m_clR[1]; g0 = m_clR[2]; t0 = m_clR[3];

            a1 *= m_tiPR[0]*a0 + m_tiPR[1]*c0 + m_tiPR[2]*g0 + m_tiPR[3]*t0;
            c1 *= m_tiPR[4]*a0 + m_tiPR[5]*c0 + m_tiPR[6]*g0 + m_tiPR[7]*t0;
            g1 *= m_tiPR[8]*a0 + m_tiPR[9]*c0 + m_tiPR[10]*g0 + m_tiPR[11]*t0;
            t1 *= m_tiPR[12]*a0 + m_tiPR[13]*c0 + m_tiPR[14]*g0 + m_tiPR[15]*t0;

            scaler = MAX(scaler, a1);
            scaler = MAX(scaler, c1);
            scaler = MAX(scaler, g1);
            scaler = MAX(scaler, t1);

            m_clL[0] = a1;
            m_clL[1] = c1;
            m_clL[2] = g1;
            m_clL[3] = t1;

            m_clL += 4;
            m_clR += 4;
            m_tiPL += 16;
            m_tiPR += 16;
        }

        m_clL = &s_clL[tid*17];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            m_clL[0] /= scaler;
            m_clL[1] /= scaler;
            m_clL[2] /= scaler;
            m_clL[3] /= scaler;
            m_clL += 4;
        }
        scaler = (CLFlt)log(scaler);
        scPNew[char_idx] = scaler;
        CLFlt dif = lnScaler[char_idx];
        dif += scPNew[char_idx];
        lnScaler[char_idx] = dif;
    }

    __syncthreads();

    m_clL = &s_clL[tid+offset];
    for(i=j=k=0;i<times;i++)
    {
        g_clP[j] = m_clL[k];
        k += 68;
        j += 64;
    }
    if(tid < nthread) g_clP[j]=m_clL[k];
}

static void down_0_2(int offset_clP, int offset_clL, int offset_clR, int offset_pL, int offset_pR, int offset_lnScaler, int offset_scPNew, int modelnumChars, int modelnumGammaCats, int chain)
{
    if(modelnumChars < inter_task)
    {
        int	bz = 192 / (4 * modelnumGammaCats);
        int	gx = modelnumChars / bz;
        if (modelnumChars % bz != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, modelnumGammaCats, bz);

        gpu_down_0_2<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtreeScalerSpace+offset_lnScaler, devnodeScalerSpace+offset_scPNew, modelnumChars);
        cutilCheckMsg("gpu_down_0_2 failed\n");
    }
    else
    {
        int nThread = 64;
        int gx = modelnumChars / nThread;
        int residue = modelnumChars % nThread;
        if (residue != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(64, 1, 1);

        gpu_down_0_2_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtreeScalerSpace+offset_lnScaler, devnodeScalerSpace+offset_scPNew, residue);
        cutilCheckMsg("gpu_down_0_2_e failed\n");
    }
}

extern "C" void cuda_down_0 (int offset_clP, int offset_clL, int offset_clR, int offset_pL, int offset_pR,int offset_scPOld, int offset_scPNew, int offset_lnScaler, int scaler, int modelnumChars, int modelnumGammaCats, int chain)
{
    switch(scaler)
    {
        case 0:

            down_0_0(offset_clP, offset_clL, offset_clR, offset_pL, offset_pR, modelnumChars, modelnumGammaCats,chain);

            break;

        case 1:

            down_0_1(offset_clP, offset_clL, offset_clR, offset_pL, offset_pR, offset_lnScaler, offset_scPOld, modelnumChars, modelnumGammaCats, chain);

            break;

        case 2:

            down_0_2(offset_clP, offset_clL, offset_clR, offset_pL, offset_pR, offset_lnScaler, offset_scPNew, modelnumChars, modelnumGammaCats, chain);

            break;

        case 3:

            down_0_3(offset_clP, offset_clL, offset_clR, offset_pL, offset_pR, offset_lnScaler, offset_scPOld, offset_scPNew, modelnumChars, modelnumGammaCats, chain);

    }

}

__global__ void gpu_down_12_0(CLFlt *clP, CLFlt *clx, CLFlt *tiPX, int *xState,  CLFlt *ti_preLikeX, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int char_idx = 12 * blockIdx.x + tid_z;
    int clx_idx = 16 * char_idx + 4 * tid_y + tid_x;
    int sclx_idx = 16 * tid_z + 4 * tid_y + tid_x;
    int stipx_idx = sclx_idx;

    __shared__ CLFlt s_clx[192];
    __shared__ CLFlt s_tiPX[64];
    __shared__ CLFlt s_preLikeX[80];

    if(stipx_idx < 64)
    {
        s_tiPX[stipx_idx] = tiPX[stipx_idx];
    }

    if(tid_z < 4)
    {
        int tip_idx = 16*tid_z + tid_y + 4*tid_x;
        int spreLikex_idx = 20*tid_z+4*tid_y+tid_x;
        s_preLikeX[spreLikex_idx] = ti_preLikeX[tip_idx];
    }
    else if(tid_z < 5)
    {
        int spreLikex_idx = 20*tid_x + 16 + tid_y;
        s_preLikeX[spreLikex_idx] = 1.0f;
    }

    if(char_idx < modelnumChars)
    {
        s_clx[sclx_idx] = clx[clx_idx];

        __syncthreads();

        sclx_idx = 16 * tid_z + 4 * tid_y;
        stipx_idx = 16 * tid_y + 4 * tid_x;

        int xi = xState[char_idx];

        clP[clx_idx] =
                (s_tiPX[stipx_idx+0]*s_clx[sclx_idx+0] +
                 s_tiPX[stipx_idx+1]*s_clx[sclx_idx+1] +
                 s_tiPX[stipx_idx+2]*s_clx[sclx_idx+2] +
                 s_tiPX[stipx_idx+3]*s_clx[sclx_idx+3]) *
                s_preLikeX[xi + 20 * tid_y + tid_x];
    }
}

__global__ void gpu_down_12_0_e(CLFlt *clP, CLFlt *clx, CLFlt *tiPX, int *xState,  CLFlt *ti_preLikeX, int residue)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPX[64];
    __shared__ CLFlt s_preLikeX[80];
    __shared__ CLFlt s_clx[80*16 + 5*16];

    CLFlt *m_tiPX = &s_tiPX[0];

    int num=80, times=16, nthread=0;
    if(bid_x+1 == gridDim.x && residue != 0)
    {
        num=residue;
        int i = num*16;
        times = i/80;
        nthread = i%80;
    }

    if(tid_y < 4)
    {
        int tip_idx = 16 * tid_z + tid_y + 4 * tid_x;
        s_preLikeX[tid] = ti_preLikeX[tip_idx];
    }
    else
    {
        s_preLikeX[tid] = 1.0f;
    }

    if(tid < 64)
    {
        s_tiPX[tid] = tiPX[tid];
    }
    int offset = tid>>4;
    int i,j,k;
    CLFlt *m_clx = &s_clx[tid+offset];
    CLFlt *g_clx = &clx[bid_x*80*16+tid];
    CLFlt *g_clP = &clP[bid_x*80*16+tid];
    for(i=j=k=0;i<times;i++)
    {
        m_clx[k] = g_clx[j];
        k+=85;
        j+=80;
    }
    if(tid < nthread) m_clx[k]=g_clx[j];

    __syncthreads();

    if(tid < num)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1;
        int gamma, xi;
        m_clx = &s_clx[17*tid];
        xi = xState[char_idx];

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_clx[0]; c0 = m_clx[1]; g0 = m_clx[2]; t0 = m_clx[3];

            a1 = m_tiPX[0]*a0 + m_tiPX[1]*c0 + m_tiPX[2]*g0 + m_tiPX[3]*t0;
            c1 = m_tiPX[4]*a0 + m_tiPX[5]*c0 + m_tiPX[6]*g0 + m_tiPX[7]*t0;
            g1 = m_tiPX[8]*a0 + m_tiPX[9]*c0 + m_tiPX[10]*g0 + m_tiPX[11]*t0;
            t1 = m_tiPX[12]*a0 + m_tiPX[13]*c0 + m_tiPX[14]*g0 + m_tiPX[15]*t0;

            a1 *= s_preLikeX[xi];
            c1 *= s_preLikeX[xi+1];
            g1 *= s_preLikeX[xi+2];
            t1 *= s_preLikeX[xi+3];

            m_clx[0] = a1;
            m_clx[1] = c1;
            m_clx[2] = g1;
            m_clx[3] = t1;

            m_clx += 4;
            m_tiPX += 16;
            xi += 20;
        }
    }

    __syncthreads();

    m_clx = &s_clx[tid+offset];
    for(i=j=k=0;i<times;i++)
    {
        g_clP[j] = m_clx[k];
        k += 85;
        j += 80;
    }
    if(tid < nthread) g_clP[j]=m_clx[k];
}

static void  down_12_0(int offset_clP, int offset_clx, int offset_px, int offset_xState, int offset_px_preLike, int modelnumChars, int modelnumGammaCats, int chain)
{
    if(modelnumChars < inter_task)
    {
        int	bz = 192 / (4 * modelnumGammaCats);
        int	gx = modelnumChars / bz;
        if (modelnumChars % bz != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, modelnumGammaCats, bz);

        gpu_down_12_0<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clx, devtiProbSpace+offset_px, devtermState+offset_xState, devtiProbSpace+offset_px_preLike, modelnumChars);
    }
    else
    {
        int nThread = 80;
        int gx = modelnumChars / nThread;
        int residue = modelnumChars % nThread;
        if (residue != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, 5, 4);

        gpu_down_12_0_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clx, devtiProbSpace+offset_px, devtermState+offset_xState, devtiProbSpace+offset_px_preLike, residue);
    }
}

__global__ void gpu_down_12_1(CLFlt *clP, CLFlt *clx, CLFlt *tiPX, int *xState,  CLFlt *ti_preLikeX, CLFlt *lnScaler, CLFlt *scPOld, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int char_idx = 12 * blockIdx.x + tid_z;
    int clx_idx = 16 * char_idx + 4 * tid_y + tid_x;
    int sclx_idx = 16 * tid_z + 4 * tid_y + tid_x;
    int stipx_idx = sclx_idx;

    __shared__ CLFlt s_clx[192];
    __shared__ CLFlt s_tiPX[64];
    __shared__ CLFlt s_preLikeX[80];

    if(stipx_idx < 64)
    {
        s_tiPX[stipx_idx] = tiPX[stipx_idx];
    }

    if(tid_z < 4)
    {
        int tip_idx = 16*tid_z + tid_y + 4*tid_x;
        int spreLikex_idx = 20*tid_z+4*tid_y+tid_x;
        s_preLikeX[spreLikex_idx] = ti_preLikeX[tip_idx];
    }
    else if(tid_z < 5)
    {
        int spreLikex_idx = 20*tid_x + 4*tid_z + tid_y;
        s_preLikeX[spreLikex_idx] = 1.0f;
    }

    if(char_idx < modelnumChars)
    {
        s_clx[sclx_idx] = clx[clx_idx];

        __syncthreads();

        sclx_idx = 16 * tid_z + 4 * tid_y;
        stipx_idx = 16 * tid_y + 4 * tid_x;

        int xi = xState[char_idx];

        clP[clx_idx] =
                (s_tiPX[stipx_idx+0]*s_clx[sclx_idx+0] +
                 s_tiPX[stipx_idx+1]*s_clx[sclx_idx+1] +
                 s_tiPX[stipx_idx+2]*s_clx[sclx_idx+2] +
                 s_tiPX[stipx_idx+3]*s_clx[sclx_idx+3]) *
                s_preLikeX[xi + 20 * tid_y + tid_x];

        if((!tid_x)&&(!tid_y))
            lnScaler[char_idx] -= scPOld[char_idx];
    }
}

__global__ void gpu_down_12_1_e(CLFlt *clP, CLFlt *clx, CLFlt *tiPX, int *xState,  CLFlt *ti_preLikeX, CLFlt *lnScaler, CLFlt *scPOld, int residue)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPX[64];
    __shared__ CLFlt s_preLikeX[80];
    __shared__ CLFlt s_clx[80*17];

    CLFlt *m_tiPX = &s_tiPX[0];

    int num=80, times=16, nthread=0;
    if(bid_x+1 == gridDim.x && residue != 0)
    {
        num=residue;
        int i = num*16;
        times = i/80;
        nthread = i%80;
    }

    if(tid_y < 4)
    {
        int tip_idx = 16 * tid_z + tid_y + 4 * tid_x;
        s_preLikeX[tid] = ti_preLikeX[tip_idx];
    }
    else
        s_preLikeX[tid] = 1.0f;

    if(tid < 64)
    {
        s_tiPX[tid] = tiPX[tid];
    }
    int offset = tid>>4;
    int i,j,k;
    CLFlt *m_clx = &s_clx[tid+offset];
    CLFlt *g_clx = &clx[bid_x*80*16+tid];
    CLFlt *g_clP = &clP[bid_x*80*16+tid];
    for(i=j=k=0;i<times;i++)
    {
        m_clx[k] = g_clx[j];
        k+=85;
        j+=80;
    }
    if(tid < nthread) m_clx[k]=g_clx[j];

    __syncthreads();

    if(tid < num)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1;
        int gamma, xi;
        m_clx = &s_clx[17*tid];
        xi = xState[char_idx];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_clx[0]; c0 = m_clx[1]; g0 = m_clx[2]; t0 = m_clx[3];

            a1 = m_tiPX[0]*a0 + m_tiPX[1]*c0 + m_tiPX[2]*g0 + m_tiPX[3]*t0;
            c1 = m_tiPX[4]*a0 + m_tiPX[5]*c0 + m_tiPX[6]*g0 + m_tiPX[7]*t0;
            g1 = m_tiPX[8]*a0 + m_tiPX[9]*c0 + m_tiPX[10]*g0 + m_tiPX[11]*t0;
            t1 = m_tiPX[12]*a0 + m_tiPX[13]*c0 + m_tiPX[14]*g0 + m_tiPX[15]*t0;

            a1 *= s_preLikeX[xi];
            c1 *= s_preLikeX[xi+1];
            g1 *= s_preLikeX[xi+2];
            t1 *= s_preLikeX[xi+3];

            m_clx[0] = a1;
            m_clx[1] = c1;
            m_clx[2] = g1;
            m_clx[3] = t1;

            m_clx += 4;
            m_tiPX += 16;
            xi += 20;
        }

        CLFlt dif = lnScaler[char_idx];
        dif -= scPOld[char_idx];
        lnScaler[char_idx] = dif;
    }

    __syncthreads();

    m_clx = &s_clx[tid+offset];
    for(i=j=k=0;i<times;i++)
    {
        g_clP[j] = m_clx[k];
        k += 85;
        j += 80;
    }
    if(tid < nthread) g_clP[j]=m_clx[k];
}

static void  down_12_1(int offset_clP, int offset_clx, int offset_px, int offset_xState, int offset_px_preLike, int offset_lnScaler, int offset_scPOld, int modelnumChars, int modelnumGammaCats, int chain)
{
    if(modelnumChars < inter_task)
    {
        int	bz = 192 / (4 * modelnumGammaCats);
        int	gx = modelnumChars / bz;
        if (modelnumChars % bz != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, modelnumGammaCats, bz);

        gpu_down_12_1<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clx, devtiProbSpace+offset_px, devtermState+offset_xState, devtiProbSpace+offset_px_preLike, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPOld, modelnumChars);
    }
    else
    {
        int nThread = 80;
        int gx = modelnumChars / nThread;
        int residue = modelnumChars % nThread;
        if (residue != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, 5, 4);

        gpu_down_12_1_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clx, devtiProbSpace+offset_px, devtermState+offset_xState, devtiProbSpace+offset_px_preLike, devtreeScalerSpace+offset_lnScaler, devnodeScalerSpace+offset_scPOld, residue);
    }
}

__global__ void gpu_down_12_2 (CLFlt *clP, CLFlt *clx, CLFlt *tiPX, int *xState,  CLFlt *ti_preLikeX, CLFlt *lnScaler, CLFlt *scPNew, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int char_idx = 12 * blockIdx.x + tid_z;
    int clx_idx = 16 * char_idx + 4 * tid_y + tid_x;
    int sclx_idx = 16 * tid_z + 4 * tid_y + tid_x;
    int stipx_idx = sclx_idx;

    __shared__ CLFlt s_clx[192];
    __shared__ CLFlt s_clP[192];
    __shared__ CLFlt s_tiPX[64];
    __shared__ CLFlt s_preLikeX[80];

    if(stipx_idx < 64)
    {
        s_tiPX[stipx_idx] = tiPX[stipx_idx];
    }

    if(tid_z < 4)
    {
        int tip_idx = 16*tid_z + tid_y + 4*tid_x;
        int spreLikex_idx = 20*tid_z+4*tid_y+tid_x;
        s_preLikeX[spreLikex_idx] = ti_preLikeX[tip_idx];
    }
    else if(tid_z < 5)
    {
        int spreLikex_idx = 20*tid_x + 16 + tid_y;
        s_preLikeX[spreLikex_idx] = 1.0f;
    }

    if(char_idx < modelnumChars)
    {
        s_clx[sclx_idx] = clx[clx_idx];

        __syncthreads();

        int sclp_idx = sclx_idx;
        sclx_idx = 16 * tid_z + 4 * tid_y;
        stipx_idx = 16 * tid_y + 4 * tid_x;
        int xi = xState[char_idx];

        s_clP[sclp_idx] =
                (s_tiPX[stipx_idx+0]*s_clx[sclx_idx+0] +
                 s_tiPX[stipx_idx+1]*s_clx[sclx_idx+1] +
                 s_tiPX[stipx_idx+2]*s_clx[sclx_idx+2] +
                 s_tiPX[stipx_idx+3]*s_clx[sclx_idx+3]) *
                s_preLikeX[xi + 20 * tid_y + tid_x];

        CLFlt new_lnScaler;
        new_lnScaler = clScaler(s_clP);

        if((!tid_x)&&(!tid_y))
        {
            scPNew[char_idx] = new_lnScaler;
            lnScaler[char_idx] += new_lnScaler;
        }

        clP[clx_idx] = s_clP[sclp_idx];
    }
}

__global__ void gpu_down_12_2_e(CLFlt *clP, CLFlt *clx, CLFlt *tiPX, int *xState,  CLFlt *ti_preLikeX, CLFlt *lnScaler, CLFlt *scPNew, int modelnumChars, int residue)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPX[64];
    __shared__ CLFlt s_preLikeX[80];
    __shared__ CLFlt s_clx[80*17];

    CLFlt *m_tiPX = &s_tiPX[0];

    int num=80, times=16, nthread=0;
    if(bid_x+1 == gridDim.x && residue != 0)
    {
        num=residue;
        int i = num*16;
        times = i/80;
        nthread = i%80;
    }

    if(tid_y < 4)
    {
        int tip_idx = 16 * tid_z + tid_y + 4 * tid_x;
        s_preLikeX[tid] = ti_preLikeX[tip_idx];
    }
    else
        s_preLikeX[tid] = 1.0f;

    if(tid < 64)
    {
        s_tiPX[tid] = tiPX[tid];
    }
    int offset = tid>>4;
    int i,j,k;
    CLFlt *m_clx = &s_clx[tid+offset];
    CLFlt *g_clx = &clx[bid_x*80*16+tid];
    CLFlt *g_clP = &clP[bid_x*80*16+tid];
    for(i=j=k=0;i<times;i++)
    {
        m_clx[k] = g_clx[j];
        k+=85;
        j+=80;
    }
    if(tid < nthread) m_clx[k]=g_clx[j];

    __syncthreads();

    if(tid < num)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1, scaler = 0.0;
        int gamma, xi;

        m_clx = &s_clx[17*tid];
        xi = xState[char_idx];

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_clx[0]; c0 = m_clx[1]; g0 = m_clx[2]; t0 = m_clx[3];

            a1 = m_tiPX[0]*a0 + m_tiPX[1]*c0 + m_tiPX[2]*g0 + m_tiPX[3]*t0;
            c1 = m_tiPX[4]*a0 + m_tiPX[5]*c0 + m_tiPX[6]*g0 + m_tiPX[7]*t0;
            g1 = m_tiPX[8]*a0 + m_tiPX[9]*c0 + m_tiPX[10]*g0 + m_tiPX[11]*t0;
            t1 = m_tiPX[12]*a0 + m_tiPX[13]*c0 + m_tiPX[14]*g0 + m_tiPX[15]*t0;

            a1 *= s_preLikeX[xi];
            c1 *= s_preLikeX[xi+1];
            g1 *= s_preLikeX[xi+2];
            t1 *= s_preLikeX[xi+3];

            scaler = MAX(scaler, a1);
            scaler = MAX(scaler, c1);
            scaler = MAX(scaler, g1);
            scaler = MAX(scaler, t1);

            m_clx[0] = a1;
            m_clx[1] = c1;
            m_clx[2] = g1;
            m_clx[3] = t1;

            m_clx += 4;
            m_tiPX += 16;
            xi += 20;
        }

        m_clx = &s_clx[tid*17];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            m_clx[0] /= scaler;
            m_clx[1] /= scaler;
            m_clx[2] /= scaler;
            m_clx[3] /= scaler;
            m_clx += 4;
        }

        scaler = (CLFlt)log(scaler);
        scPNew[char_idx] = scaler;
        CLFlt dif = lnScaler[char_idx];
        dif += scaler;
        lnScaler[char_idx] = dif;
    }

    __syncthreads();

    m_clx = &s_clx[tid+offset];
    for(i=j=k=0;i<times;i++)
    {
        g_clP[j] = m_clx[k];
        k += 85;
        j += 80;
    }
    if(tid < nthread) g_clP[j]=m_clx[k];
}

static void down_12_2(int offset_clP, int offset_clx, int offset_px, int offset_xState, int offset_px_preLike,int  offset_lnScaler,int  offset_scPNew, int modelnumChars, int modelnumGammaCats, int chain)
{
    if(modelnumChars < inter_task)
    {
        int	bz = 192 / (4 * modelnumGammaCats);
        int	gx = modelnumChars / bz;
        if (modelnumChars % bz != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, modelnumGammaCats, bz);

        gpu_down_12_2<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clx, devtiProbSpace+offset_px, devtermState+offset_xState, devtiProbSpace+offset_px_preLike, devtreeScalerSpace+offset_lnScaler, devnodeScalerSpace+offset_scPNew, modelnumChars);
    }
    else
    {
        int nThread = 80;
        int gx = modelnumChars / nThread;
        int residue = modelnumChars % nThread;
        if (residue != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, 5, 4);

        gpu_down_12_2_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clx, devtiProbSpace+offset_px, devtermState+offset_xState, devtiProbSpace+offset_px_preLike, devtreeScalerSpace+offset_lnScaler, devnodeScalerSpace+offset_scPNew, modelnumChars, residue);
    }
}

__global__ void gpu_down_12_3(CLFlt *clP, CLFlt *clx, CLFlt *tiPX, int *xState,  CLFlt *ti_preLikeX, CLFlt *lnScaler, CLFlt *scPOld, CLFlt *scPNew, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int char_idx = 12 * blockIdx.x + tid_z;
    int clx_idx = 16 * char_idx + 4 * tid_y + tid_x;
    int sclx_idx = 16 * tid_z + 4 * tid_y + tid_x;
    int stipx_idx = sclx_idx;

    __shared__ CLFlt s_clx[192];
    __shared__ CLFlt s_clP[192];
    __shared__ CLFlt s_tiPX[64];
    __shared__ CLFlt s_preLikeX[80];

    if(stipx_idx < 64)
    {
        s_tiPX[stipx_idx] = tiPX[stipx_idx];
    }

    if(tid_z < 4)
    {
        int tip_idx = 16*tid_z + tid_y + 4*tid_x;
        int spreLikex_idx = 20*tid_z+4*tid_y+tid_x;
        s_preLikeX[spreLikex_idx] = ti_preLikeX[tip_idx];
    }
    else if(tid_z < 5)
    {
        int spreLikex_idx = 20*tid_x + 16 + tid_y;
        s_preLikeX[spreLikex_idx] = 1.0f;
    }

    if(char_idx < modelnumChars)
    {
        s_clx[sclx_idx] = clx[clx_idx];

        __syncthreads();

        int sclp_idx = sclx_idx;

        sclx_idx = 16 * tid_z + 4 * tid_y;
        stipx_idx = 16 * tid_y + 4 * tid_x;

        int xi = xState[char_idx];

        s_clP[sclp_idx] =
                (s_tiPX[stipx_idx+0]*s_clx[sclx_idx+0] +
                 s_tiPX[stipx_idx+1]*s_clx[sclx_idx+1] +
                 s_tiPX[stipx_idx+2]*s_clx[sclx_idx+2] +
                 s_tiPX[stipx_idx+3]*s_clx[sclx_idx+3]) *
                s_preLikeX[xi + 20 * tid_y + tid_x];

        CLFlt new_lnScaler;
        new_lnScaler = clScaler(s_clP);

        if((!tid_x)&&(!tid_y))
        {
            scPNew[char_idx] = new_lnScaler;
            CLFlt dif = lnScaler[char_idx];
            dif -= scPOld[char_idx];
            dif += new_lnScaler;
            lnScaler[char_idx] = dif;
        }

        clP[clx_idx] = s_clP[sclp_idx];
    }
}

__global__ void gpu_down_12_3_e(CLFlt *clP, CLFlt *clx, CLFlt *tiPX, int *xState,  CLFlt *ti_preLikeX, CLFlt *lnScaler, CLFlt *scPOld, CLFlt *scPNew, int residue)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPX[64];
    __shared__ CLFlt s_preLikeX[80];
    __shared__ CLFlt s_clx[80*17];

    CLFlt *m_tiPX = &s_tiPX[0];

    int num=80, times=16, nthread=0;
    if(bid_x+1 == gridDim.x && residue != 0)
    {
        num=residue;
        int i = num*16;
        times = i/80;
        nthread = i%80;
    }

    if(tid_y < 4)
    {
        int tip_idx = 16 * tid_z + tid_y + 4 * tid_x;
        s_preLikeX[tid] = ti_preLikeX[tip_idx];
    }
    else
        s_preLikeX[tid] = 1.0f;

    if(tid < 64)
    {
        s_tiPX[tid] = tiPX[tid];
    }
    int offset = tid>>4;
    int i,j,k;
    CLFlt *m_clx = &s_clx[tid+offset];
    CLFlt *g_clx = &clx[bid_x*80*16+tid];
    CLFlt *g_clP = &clP[bid_x*80*16+tid];
    for(i=j=k=0;i<times;i++)
    {
        m_clx[k] = g_clx[j];
        k+=85;
        j+=80;
    }
    if(tid < nthread) m_clx[k]=g_clx[j];

    __syncthreads();

    if(tid < num)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1, scaler = 0.0;
        int gamma, xi;

        m_clx = &s_clx[17*tid];
        xi = xState[char_idx];

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_clx[0]; c0 = m_clx[1]; g0 = m_clx[2]; t0 = m_clx[3];

            a1 = m_tiPX[0]*a0 + m_tiPX[1]*c0 + m_tiPX[2]*g0 + m_tiPX[3]*t0;
            c1 = m_tiPX[4]*a0 + m_tiPX[5]*c0 + m_tiPX[6]*g0 + m_tiPX[7]*t0;
            g1 = m_tiPX[8]*a0 + m_tiPX[9]*c0 + m_tiPX[10]*g0 + m_tiPX[11]*t0;
            t1 = m_tiPX[12]*a0 + m_tiPX[13]*c0 + m_tiPX[14]*g0 + m_tiPX[15]*t0;

            a1 *= s_preLikeX[xi];
            c1 *= s_preLikeX[xi+1];
            g1 *= s_preLikeX[xi+2];
            t1 *= s_preLikeX[xi+3];

            scaler = MAX(scaler, a1);
            scaler = MAX(scaler, c1);
            scaler = MAX(scaler, g1);
            scaler = MAX(scaler, t1);

            m_clx[0] = a1;
            m_clx[1] = c1;
            m_clx[2] = g1;
            m_clx[3] = t1;

            m_clx += 4;
            m_tiPX += 16;
            xi += 20;
        }

        m_clx = &s_clx[tid*17];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            m_clx[0] /= scaler;
            m_clx[1] /= scaler;
            m_clx[2] /= scaler;
            m_clx[3] /= scaler;
            m_clx += 4;
        }

        scaler = (CLFlt)log(scaler);
        scPNew[char_idx] = scaler;
        CLFlt dif = lnScaler[char_idx];
        dif -= scPOld[char_idx];
        dif += scaler;
        lnScaler[char_idx] = dif;
    }

    __syncthreads();

    m_clx = &s_clx[tid+offset];
    for(i=j=k=0;i<times;i++)
    {
        g_clP[j] = m_clx[k];
        k += 85;
        j += 80;
    }
    if(tid < nthread) g_clP[j]=m_clx[k];
}

static void down_12_3(int offset_clP, int offset_clx, int offset_px, int offset_xState, int offset_px_preLike, int offset_lnScaler, int offset_scPOld, int offset_scPNew, int modelnumChars, int modelnumGammaCats, int chain)
{
    if(modelnumChars < inter_task)
    {
        int	bz = 192 / (modelnumGammaCats * 4);
        int	gx = modelnumChars / bz;
        if (modelnumChars % bz != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, modelnumGammaCats, bz);

        gpu_down_12_3<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clx, devtiProbSpace+offset_px, devtermState+offset_xState, devtiProbSpace+offset_px_preLike, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPOld, devnodeScalerSpace+offset_scPNew, modelnumChars);
    }
    else
    {
        int nThread = 80;
        int gx = modelnumChars / nThread;
        int residue = modelnumChars % nThread;
        if (residue != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, 5, 4);

        gpu_down_12_3_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clx, devtiProbSpace+offset_px, devtermState+offset_xState, devtiProbSpace+offset_px_preLike, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPOld, devnodeScalerSpace+offset_scPNew, residue);
    }
}

extern "C" void cuda_down_12 (int offset_clP, int offset_clx, int offset_px, int offset_xState, int offset_px_preLike, int offset_scPOld, int offset_scPNew, int offset_lnScaler, int scaler, int modelnumChars, int modelnumGammaCats, int chain)
{
    switch(scaler)
    {
        case 0:

            down_12_0(offset_clP, offset_clx, offset_px, offset_xState, offset_px_preLike, modelnumChars, modelnumGammaCats, chain);

            break;

        case 1:

            down_12_1(offset_clP, offset_clx, offset_px, offset_xState, offset_px_preLike, offset_lnScaler,offset_scPOld, modelnumChars, modelnumGammaCats, chain);

            break;

        case 2:
            down_12_2(offset_clP, offset_clx, offset_px, offset_xState, offset_px_preLike, offset_lnScaler, offset_scPNew, modelnumChars, modelnumGammaCats, chain);

            break;

        case 3:

            down_12_3(offset_clP, offset_clx, offset_px, offset_xState, offset_px_preLike, offset_lnScaler, offset_scPOld, offset_scPNew, modelnumChars, modelnumGammaCats, chain);

    }
}

__global__ void gpu_down_3_0 (CLFlt *clP, int *lState, int *rState, CLFlt *tiPL, CLFlt *tiPR, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int char_idx = 12 * blockIdx.x + tid_z;
    int clx_idx = 16 * char_idx + 4 * tid_y + tid_x;

    __shared__ CLFlt s_preLikeL[80];
    __shared__ CLFlt s_preLikeR[80];

    if(tid_z < 4)
    {
        int tip_idx = 16*tid_z + tid_y + 4*tid_x;
        int spreLikex_idx = 20*tid_z+4*tid_y+tid_x;
        s_preLikeL[spreLikex_idx] = tiPL[tip_idx];
        s_preLikeR[spreLikex_idx] = tiPR[tip_idx];
    }
    else if(tid_z < 5)
    {
        int spreLikex_idx = 20*tid_x + 16 + tid_y;
        s_preLikeL[spreLikex_idx] = 1.0f;
        s_preLikeR[spreLikex_idx] = 1.0f;
    }

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        int li = lState[char_idx];
        int ri = rState[char_idx];

        clP[clx_idx] =
                s_preLikeL[li + 20 * tid_y + tid_x] *
                s_preLikeR[ri + 20 * tid_y + tid_x];
    }
}


static void down_3_0(int offset_clP, int offset_lState, int offset_rState, int offset_pL, int offset_pR, int modelnumChars, int modelnumGammaCats, int chain)
{

    int	bz = 192 / (4 * modelnumGammaCats);
    int	gx = modelnumChars / bz;
    if (modelnumChars % bz != 0)
        gx += 1;

    dim3	dimGrid(gx, 1, 1);
    dim3	dimBlock(4, modelnumGammaCats, bz);

    gpu_down_3_0<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devtermState+offset_lState, devtermState+offset_rState, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, modelnumChars);
}

__global__ void gpu_down_3_1 (CLFlt *clP, int *lState, int *rState, CLFlt *tiPL, CLFlt *tiPR, CLFlt *lnScaler, CLFlt *scPOld, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int char_idx = 12 * blockIdx.x + tid_z;
    int clx_idx = 16 * char_idx + 4 * tid_y + tid_x;

    __shared__ CLFlt s_preLikeL[80];
    __shared__ CLFlt s_preLikeR[80];

    if(tid_z < 4)
    {
        int tip_idx = 16*tid_z + tid_y + 4*tid_x;
        int spreLikex_idx = 20*tid_z+4*tid_y+tid_x;
        s_preLikeL[spreLikex_idx] = tiPL[tip_idx];
        s_preLikeR[spreLikex_idx] = tiPR[tip_idx];
    }
    else if(tid_z < 5)
    {
        int spreLikex_idx = 20*tid_x + 16 + tid_y;
        s_preLikeL[spreLikex_idx] = 1.0f;
        s_preLikeR[spreLikex_idx] = 1.0f;
    }

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        int i = lState[char_idx];
        int j = rState[char_idx];

        clP[clx_idx] =
                s_preLikeL[i + 20 * tid_y + tid_x] *
                s_preLikeR[j + 20 * tid_y + tid_x];

        if((!tid_x)&&(!tid_y))
            lnScaler[char_idx] -= scPOld[char_idx];
    }
}

static void down_3_1(int offset_clP, int offset_lState, int offset_rState, int offset_pL, int offset_pR, int offset_lnScaler, int offset_scPOld, int modelnumChars, int modelnumGammaCats, int chain)
{
    int	bz = 192 / (4 * modelnumGammaCats);
    int	gx = modelnumChars / bz;
    if (modelnumChars % bz != 0)
        gx += 1;

    dim3	dimGrid(gx, 1, 1);
    dim3	dimBlock(4, modelnumGammaCats, bz);

    gpu_down_3_1<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devtermState+offset_lState, devtermState+offset_rState, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPOld, modelnumChars);
}

__global__ void gpu_down_3_2 (CLFlt *clP, int *lState, int *rState, CLFlt *tiPL, CLFlt *tiPR, CLFlt *lnScaler, CLFlt *scPNew, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int char_idx = 12 * blockIdx.x + tid_z;
    int clx_idx = 16 * char_idx + 4 * tid_y + tid_x;
    int spreLikex_idx = 16 * tid_z + 4 * tid_y + tid_x;

    __shared__ CLFlt s_preLikeL[80];
    __shared__ CLFlt s_preLikeR[80];
    __shared__ CLFlt s_clP[192];

    if(tid_z < 4)
    {
        int tip_idx = 16*tid_z + tid_y + 4*tid_x;
        int idx = 20*tid_z+4*tid_y+tid_x;
        s_preLikeL[idx] = tiPL[tip_idx];
        s_preLikeR[idx] = tiPR[tip_idx];
    }
    else if(tid_z < 5)
    {
        int idx = 20*tid_x + 16 + tid_y;
        s_preLikeL[idx] = 1.0f;
        s_preLikeR[idx] = 1.0f;
    }

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        int li = lState[char_idx];
        int ri = rState[char_idx];

        s_clP[spreLikex_idx] =
                s_preLikeL[li + 20 * tid_y + tid_x] *
                s_preLikeR[ri + 20 * tid_y + tid_x];

        CLFlt new_lnScaler;
        new_lnScaler = clScaler(s_clP);

        if((!tid_x)&&(!tid_y))
        {
            scPNew[char_idx] = new_lnScaler;
            lnScaler[char_idx] += new_lnScaler;
        }

        clP[clx_idx] = s_clP[spreLikex_idx];
    }
}

__global__ void gpu_down_3_2_e(CLFlt *clP, int *lState, int *rState, CLFlt *tiPL, CLFlt *tiPR, CLFlt *lnScaler, CLFlt *scPNew, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_preLikeL[80];
    __shared__ CLFlt s_preLikeR[80];
    __shared__ CLFlt s_clP[80*17];

    if(tid_y < 4)
    {
        int tiPX_idx = 16 * tid_z + tid_y + 4 * tid_x;
        s_preLikeL[tid] = tiPL[tiPX_idx];
        s_preLikeR[tid] = tiPR[tiPX_idx];
    }
    else
    {
        s_preLikeL[tid] = 1.0f;
        s_preLikeR[tid] = 1.0f;
    }

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        CLFlt a0, c0, g0, t0, scaler=0.0f;
        int gamma, li, ri;
        CLFlt *g_clP = &clP[16 * char_idx];
        CLFlt *m_clP = &s_clP[tid * 17];
        li = lState[char_idx];
        ri = rState[char_idx];

        for(gamma=0; gamma < 4; gamma ++)
        {
            a0 = s_preLikeL[li]*s_preLikeR[ri];
            c0 = s_preLikeL[li+1]*s_preLikeR[ri+1];
            g0 = s_preLikeL[li+2]*s_preLikeR[ri+2];
            t0 = s_preLikeL[li+3]*s_preLikeR[ri+3];

            scaler = MAX(scaler, a0);
            scaler = MAX(scaler, c0);
            scaler = MAX(scaler, g0);
            scaler = MAX(scaler, t0);

            m_clP[0] = a0;
            m_clP[1] = c0;
            m_clP[2] = g0;
            m_clP[3] = t0;
            m_clP += 4;

            li += 20;
            ri += 20;
        }

        m_clP = &s_clP[tid * 17];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_clP[0]; g_clP[0] = a0/scaler;
            c0 = m_clP[1]; g_clP[1] = c0/scaler;
            g0 = m_clP[2]; g_clP[2] = g0/scaler;
            t0 = m_clP[3]; g_clP[3] = t0/scaler;
            m_clP += 4;
            g_clP += 4;
        }

        scaler = (CLFlt)log(scaler);
        scPNew[char_idx] = scaler;
        CLFlt dif = lnScaler[char_idx];
        dif += scaler;
        lnScaler[char_idx] = dif;
    }
}

static void down_3_2(int offset_clP, int offset_lState, int offset_rState, int offset_pL, int offset_pR, int offset_lnScaler, int offset_scPNew, int modelnumChars, int modelnumGammaCats, int chain)
{
    if(modelnumChars < inter_task)
    {
        int	bz = 192 / (4 * modelnumGammaCats);
        int	gx = modelnumChars / bz;
        if (modelnumChars % bz != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, modelnumGammaCats, bz);

        gpu_down_3_2<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devtermState+offset_lState, devtermState+offset_rState, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPNew, modelnumChars);
    }
    else
    {
        int nThread = 80;
        int gx = modelnumChars / nThread;
        int residue = modelnumChars % nThread;
        if (residue != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, 5, 4);

        gpu_down_3_2_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devtermState+offset_lState, devtermState+offset_rState, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPNew, modelnumChars);
    }
}

__global__ void gpu_down_3_3(CLFlt *clP, int *lState, int *rState, CLFlt *tiPL, CLFlt *tiPR, CLFlt *lnScaler, CLFlt *scPOld, CLFlt *scPNew, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int char_idx = 12 * blockIdx.x + tid_z;
    int clx_idx = 16 * char_idx + 4 * tid_y + tid_x;
    int spreLikex_idx = 16 * tid_z + 4 * tid_y + tid_x;

    __shared__ CLFlt s_preLikeL[80];
    __shared__ CLFlt s_preLikeR[80];
    __shared__ CLFlt s_clP[192];

    if(tid_z < 4)
    {
        int tip_idx = 16*tid_z + tid_y + 4*tid_x;
        int idx = 20*tid_z+4*tid_y+tid_x;
        s_preLikeL[idx] = tiPL[tip_idx];
        s_preLikeR[idx] = tiPR[tip_idx];
    }
    else if(tid_z < 5)
    {
        int idx = 20*tid_x + 16 + tid_y;
        s_preLikeL[idx] = 1.0f;
        s_preLikeR[idx] = 1.0f;
    }

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        int li = lState[char_idx];
        int ri = rState[char_idx];

        s_clP[spreLikex_idx] =
                s_preLikeL[li + 20 * tid_y + tid_x] *
                s_preLikeR[ri + 20 * tid_y + tid_x];

        CLFlt new_lnScaler;
        new_lnScaler = clScaler(s_clP);

        if((!tid_x)&&(!tid_y))
        {
            scPNew[char_idx] = new_lnScaler;
            CLFlt dif = lnScaler[char_idx];
            dif -= scPOld[char_idx];
            dif += new_lnScaler;
            lnScaler[char_idx] = dif;
        }

        clP[clx_idx] = s_clP[spreLikex_idx];
    }
}

__global__ void gpu_down_3_3_e(CLFlt *clP, int *lState, int *rState, CLFlt *tiPL, CLFlt *tiPR, CLFlt *lnScaler, CLFlt *scPOld, CLFlt *scPNew, int residue)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_preLikeL[80];
    __shared__ CLFlt s_preLikeR[80];
    __shared__ CLFlt s_clP[80*17];

    int num=80, times=16, nthread=0;
    if(bid_x+1 == gridDim.x && residue != 0)
    {
        num=residue;
        times=(num*16)/80;
        nthread=(num*16)%80;
    }

    if(tid_y < 4)
    {
        int tiPX_idx = 16 * tid_z + tid_y + 4 * tid_x;
        s_preLikeL[tid] = tiPL[tiPX_idx];
        s_preLikeR[tid] = tiPR[tiPX_idx];
    }
    else
    {
        s_preLikeL[tid] = 1.0f;
        s_preLikeR[tid] = 1.0f;
    }

    int offset = tid>>4;
    CLFlt *m_clP = &s_clP[17*tid];
    CLFlt *g_clP = &clP[bid_x*80*16+tid];

    __syncthreads();

    if(tid < num)
    {
        CLFlt a0, c0, g0, t0, scaler=0.0f;
        int gamma, li, ri;

        li = lState[char_idx];
        ri = rState[char_idx];

        for(gamma=0; gamma < 4; gamma ++)
        {
            a0 = s_preLikeL[li]*s_preLikeR[ri];
            c0 = s_preLikeL[li+1]*s_preLikeR[ri+1];
            g0 = s_preLikeL[li+2]*s_preLikeR[ri+2];
            t0 = s_preLikeL[li+3]*s_preLikeR[ri+3];

            scaler = MAX(scaler, a0);
            scaler = MAX(scaler, c0);
            scaler = MAX(scaler, g0);
            scaler = MAX(scaler, t0);

            m_clP[0] = a0;
            m_clP[1] = c0;
            m_clP[2] = g0;
            m_clP[3] = t0;
            m_clP += 4;

            li += 20;
            ri += 20;
        }

        m_clP = &s_clP[tid * 17];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            m_clP[0] /= scaler;
            m_clP[1] /= scaler;
            m_clP[2] /= scaler;
            m_clP[3] /= scaler;
            m_clP += 4;
        }

        scaler = (CLFlt)log(scaler);
        scPNew[char_idx] = scaler;
        CLFlt dif = lnScaler[char_idx];
        dif -= scPOld[char_idx];
        dif += scaler;
        lnScaler[char_idx] = dif;
    }

    __syncthreads();

    m_clP = &s_clP[tid+offset];
    int i,j,k;
    for(i=j=k=0;i<times;i++)
    {
        g_clP[j] = m_clP[k];
        k += 85;
        j += 80;
    }
    if(tid < nthread) g_clP[j]=m_clP[k];
}

static void down_3_3(int offset_clP, int offset_lState, int offset_rState, int offset_pL, int offset_pR, int offset_lnScaler, int offset_scPOld, int offset_scPNew, int modelnumChars, int modelnumGammaCats, int chain)
{
    if(modelnumChars < inter_task)
    {
        int	bz = 192 / (4 * modelnumGammaCats);
        int	gx = modelnumChars / bz;
        if (modelnumChars % bz != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, modelnumGammaCats, bz);

        gpu_down_3_3<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devtermState+offset_lState, devtermState+offset_rState, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPOld, devnodeScalerSpace+offset_scPNew, modelnumChars);
    }
    else
    {
        int nThread = 80;
        int gx = modelnumChars / nThread;
        int residue = modelnumChars % nThread;
        if (residue != 0)
            gx += 1;

        dim3	dimGrid(gx, 1, 1);
        dim3	dimBlock(4, 5, 4);

        gpu_down_3_3_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devtermState+offset_lState, devtermState+offset_rState, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPOld, devnodeScalerSpace+offset_scPNew, residue);
    }
}

extern "C" void cuda_down_3 (int offset_clP, int offset_lState, int offset_rState, int offset_pL, int offset_pR, int offset_scPOld, int offset_scPNew, int offset_lnScaler, int scaler, int modelnumChars, int modelnumGammaCats, int chain)
{
    switch(scaler)
    {
        case 0:
            down_3_0(offset_clP, offset_lState, offset_rState, offset_pL, offset_pR, modelnumChars, modelnumGammaCats, chain);
            break;

        case 1:

            down_3_1(offset_clP, offset_lState, offset_rState, offset_pL, offset_pR, offset_lnScaler, offset_scPOld, modelnumChars, modelnumGammaCats, chain);

            break;

        case 2:

            down_3_2(offset_clP, offset_lState, offset_rState, offset_pL, offset_pR, offset_lnScaler, offset_scPNew, modelnumChars, modelnumGammaCats, chain);

            break;

        case 3:

            down_3_3(offset_clP, offset_lState, offset_rState, offset_pL, offset_pR, offset_lnScaler, offset_scPOld, offset_scPNew, modelnumChars, modelnumGammaCats, chain);
    }
}

__global__ void gpu_root_4_0_e(CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *clA, CLFlt *tiPL, CLFlt *tiPR, CLFlt *tiPA, CLFlt *lnScaler, MrBFlt *clInvar, CLFlt *nSitesOfPat, MrBFlt *lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];
    __shared__ CLFlt s_tiPA[64];

    if(tid < 64)
    {
        s_tiPL[tid] = tiPL[tid];
        s_tiPR[tid] = tiPR[tid];
        s_tiPA[tid] = tiPA[tid];
    }

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1;
        CLFlt like_a=0.0, like_c=0.0, like_g=0.0, like_t=0.0;
        CLFlt *m_tiPL = &s_tiPL[0];
        CLFlt *m_tiPR = &s_tiPR[0];
        CLFlt *m_tiPA = &s_tiPA[0];
        int clx_offset = bid_x*80*16 + 16 * tid;
        CLFlt *g_clL = &clL[clx_offset];
        CLFlt *g_clR = &clR[clx_offset];
        CLFlt *g_clA = &clA[clx_offset];
        CLFlt *g_clP = &clP[clx_offset];
        int gamma;

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = g_clL[0]; c0 = g_clL[1]; g0 = g_clL[2]; t0 = g_clL[3];

            a1 = m_tiPL[0]*a0 + m_tiPL[1]*c0 + m_tiPL[2]*g0 + m_tiPL[3]*t0;
            c1 = m_tiPL[4]*a0 + m_tiPL[5]*c0 + m_tiPL[6]*g0 + m_tiPL[7]*t0;
            g1 = m_tiPL[8]*a0 + m_tiPL[9]*c0 + m_tiPL[10]*g0 + m_tiPL[11]*t0;
            t1 = m_tiPL[12]*a0 + m_tiPL[13]*c0 + m_tiPL[14]*g0 + m_tiPL[15]*t0;

            a0 = g_clR[0]; c0 = g_clR[1]; g0 = g_clR[2]; t0 = g_clR[3];

            a1 *= m_tiPR[0]*a0 + m_tiPR[1]*c0 + m_tiPR[2]*g0 + m_tiPR[3]*t0;
            c1 *= m_tiPR[4]*a0 + m_tiPR[5]*c0 + m_tiPR[6]*g0 + m_tiPR[7]*t0;
            g1 *= m_tiPR[8]*a0 + m_tiPR[9]*c0 + m_tiPR[10]*g0 + m_tiPR[11]*t0;
            t1 *= m_tiPR[12]*a0 + m_tiPR[13]*c0 + m_tiPR[14]*g0 + m_tiPR[15]*t0;

            a0 = g_clA[0]; c0 = g_clA[1]; g0 = g_clA[2]; t0 = g_clA[3];

            a1 *= m_tiPA[0]*a0 + m_tiPA[1]*c0 + m_tiPA[2]*g0 + m_tiPA[3]*t0;
            c1 *= m_tiPA[4]*a0 + m_tiPA[5]*c0 + m_tiPA[6]*g0 + m_tiPA[7]*t0;
            g1 *= m_tiPA[8]*a0 + m_tiPA[9]*c0 + m_tiPA[10]*g0 + m_tiPA[11]*t0;
            t1 *= m_tiPA[12]*a0 + m_tiPA[13]*c0 + m_tiPA[14]*g0 + m_tiPA[15]*t0;

            like_a += a1;
            like_c += c1;
            like_g += g1;
            like_t += t1;

            g_clP[0] = a1;
            g_clP[1] = c1;
            g_clP[2] = g1;
            g_clP[3] = t1;
            g_clP += 4;

            g_clL += 4;
            g_clR += 4;
            g_clA += 4;
            m_tiPL += 16;
            m_tiPR += 16;
            m_tiPA += 16;
        }

        MrBFlt like = like_a * bs_A + like_c * bs_C + like_g * bs_G + like_t * bs_T;
        MrBFlt *g_clInvar = &clInvar[bid_x*80*4 + 4*tid];
        likelihood_NUC4_hasPInvar_YES_e(like, g_clInvar, lnScaler[char_idx], nSitesOfPat, lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, char_idx);
    }
}

__global__ void gpu_root_4_1_e(CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *clA, CLFlt *tiPL, CLFlt *tiPR, CLFlt *tiPA, CLFlt *lnScaler, CLFlt *scPOld, MrBFlt *clInvar, CLFlt *nSitesOfPat, MrBFlt *lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];
    __shared__ CLFlt s_tiPA[64];

    if(tid < 64)
    {
        s_tiPL[tid] = tiPL[tid];
        s_tiPR[tid] = tiPR[tid];
        s_tiPA[tid] = tiPA[tid];
    }

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1;
        CLFlt like_a=0.0, like_c=0.0, like_g=0.0, like_t=0.0;
        CLFlt *m_tiPL = &s_tiPL[0];
        CLFlt *m_tiPR = &s_tiPR[0];
        CLFlt *m_tiPA = &s_tiPA[0];
        int clx_offset = bid_x*80*16 + 16*tid;
        CLFlt *g_clL = &clL[clx_offset];
        CLFlt *g_clR = &clR[clx_offset];
        CLFlt *g_clA = &clA[clx_offset];
        CLFlt *g_clP = &clP[clx_offset];
        int gamma;

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = g_clL[0]; c0 = g_clL[1]; g0 = g_clL[2]; t0 = g_clL[3];

            a1 = m_tiPL[0]*a0 + m_tiPL[1]*c0 + m_tiPL[2]*g0 + m_tiPL[3]*t0;
            c1 = m_tiPL[4]*a0 + m_tiPL[5]*c0 + m_tiPL[6]*g0 + m_tiPL[7]*t0;
            g1 = m_tiPL[8]*a0 + m_tiPL[9]*c0 + m_tiPL[10]*g0 + m_tiPL[11]*t0;
            t1 = m_tiPL[12]*a0 + m_tiPL[13]*c0 + m_tiPL[14]*g0 + m_tiPL[15]*t0;

            a0 = g_clR[0]; c0 = g_clR[1]; g0 = g_clR[2]; t0 = g_clR[3];

            a1 *= m_tiPR[0]*a0 + m_tiPR[1]*c0 + m_tiPR[2]*g0 + m_tiPR[3]*t0;
            c1 *= m_tiPR[4]*a0 + m_tiPR[5]*c0 + m_tiPR[6]*g0 + m_tiPR[7]*t0;
            g1 *= m_tiPR[8]*a0 + m_tiPR[9]*c0 + m_tiPR[10]*g0 + m_tiPR[11]*t0;
            t1 *= m_tiPR[12]*a0 + m_tiPR[13]*c0 + m_tiPR[14]*g0 + m_tiPR[15]*t0;

            a0 = g_clA[0]; c0 = g_clA[1]; g0 = g_clA[2]; t0 = g_clA[3];

            a1 *= m_tiPA[0]*a0 + m_tiPA[1]*c0 + m_tiPA[2]*g0 + m_tiPA[3]*t0;
            c1 *= m_tiPA[4]*a0 + m_tiPA[5]*c0 + m_tiPA[6]*g0 + m_tiPA[7]*t0;
            g1 *= m_tiPA[8]*a0 + m_tiPA[9]*c0 + m_tiPA[10]*g0 + m_tiPA[11]*t0;
            t1 *= m_tiPA[12]*a0 + m_tiPA[13]*c0 + m_tiPA[14]*g0 + m_tiPA[15]*t0;

            like_a += a1;
            like_c += c1;
            like_g += g1;
            like_t += t1;

            g_clP[0] = a1;
            g_clP[1] = c1;
            g_clP[2] = g1;
            g_clP[3] = t1;
            g_clP += 4;

            g_clL += 4;
            g_clR += 4;
            g_clA += 4;
            m_tiPL += 16;
            m_tiPR += 16;
            m_tiPA += 16;
        }

        CLFlt dif = lnScaler[char_idx];
        dif -= scPOld[char_idx];

        MrBFlt like = like_a * bs_A + like_c * bs_C + like_g * bs_G + like_t * bs_T;
        MrBFlt *g_clInvar = &clInvar[bid_x*80*4 + 4*tid];
        likelihood_NUC4_hasPInvar_YES_e(like, g_clInvar, dif, nSitesOfPat, lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, char_idx);
        lnScaler[char_idx] = dif;
    }
}

__global__ void gpu_root_4_2_e(CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *clA, CLFlt *tiPL, CLFlt *tiPR, CLFlt *tiPA, CLFlt *lnScaler, CLFlt *scPNew, MrBFlt *clInvar, CLFlt *nSitesOfPat, MrBFlt *lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];
    __shared__ CLFlt s_tiPA[64];

    if(tid < 64)
    {
        s_tiPL[tid] = tiPL[tid];
        s_tiPR[tid] = tiPR[tid];
        s_tiPA[tid] = tiPA[tid];
    }

    __shared__ CLFlt s_like[80*17];
    CLFlt *m_like = &s_like[17*tid];

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1, scaler = 0.0;
        CLFlt *m_tiPL = &s_tiPL[0];
        CLFlt *m_tiPR = &s_tiPR[0];
        CLFlt *m_tiPA = &s_tiPA[0];
        int clx_offset = bid_x*80*16 + 16*tid;
        CLFlt *g_clL = &clL[clx_offset];
        CLFlt *g_clR = &clR[clx_offset];
        CLFlt *g_clA = &clA[clx_offset];
        CLFlt *g_clP = &clP[clx_offset];
        int gamma;

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = g_clL[0]; c0 = g_clL[1]; g0 = g_clL[2]; t0 = g_clL[3];

            a1 = m_tiPL[0]*a0 + m_tiPL[1]*c0 + m_tiPL[2]*g0 + m_tiPL[3]*t0;
            c1 = m_tiPL[4]*a0 + m_tiPL[5]*c0 + m_tiPL[6]*g0 + m_tiPL[7]*t0;
            g1 = m_tiPL[8]*a0 + m_tiPL[9]*c0 + m_tiPL[10]*g0 + m_tiPL[11]*t0;
            t1 = m_tiPL[12]*a0 + m_tiPL[13]*c0 + m_tiPL[14]*g0 + m_tiPL[15]*t0;

            a0 = g_clR[0]; c0 = g_clR[1]; g0 = g_clR[2]; t0 = g_clR[3];

            a1 *= m_tiPR[0]*a0 + m_tiPR[1]*c0 + m_tiPR[2]*g0 + m_tiPR[3]*t0;
            c1 *= m_tiPR[4]*a0 + m_tiPR[5]*c0 + m_tiPR[6]*g0 + m_tiPR[7]*t0;
            g1 *= m_tiPR[8]*a0 + m_tiPR[9]*c0 + m_tiPR[10]*g0 + m_tiPR[11]*t0;
            t1 *= m_tiPR[12]*a0 + m_tiPR[13]*c0 + m_tiPR[14]*g0 + m_tiPR[15]*t0;

            a0 = g_clA[0]; c0 = g_clA[1]; g0 = g_clA[2]; t0 = g_clA[3];

            a1 *= m_tiPA[0]*a0 + m_tiPA[1]*c0 + m_tiPA[2]*g0 + m_tiPA[3]*t0;
            c1 *= m_tiPA[4]*a0 + m_tiPA[5]*c0 + m_tiPA[6]*g0 + m_tiPA[7]*t0;
            g1 *= m_tiPA[8]*a0 + m_tiPA[9]*c0 + m_tiPA[10]*g0 + m_tiPA[11]*t0;
            t1 *= m_tiPA[12]*a0 + m_tiPA[13]*c0 + m_tiPA[14]*g0 + m_tiPA[15]*t0;

            scaler = MAX(scaler, a1);
            scaler = MAX(scaler, c1);
            scaler = MAX(scaler, g1);
            scaler = MAX(scaler, t1);

            m_like[0] = a1;
            m_like[1] = c1;
            m_like[2] = g1;
            m_like[3] = t1;

            g_clL += 4;
            g_clR += 4;
            g_clA += 4;

            m_tiPL += 16;
            m_tiPR += 16;
            m_tiPA += 16;
            m_like += 4;
        }

        m_like = &s_like[tid*17];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            m_like[0] /= scaler;
            m_like[1] /= scaler;
            m_like[2] /= scaler;
            m_like[3] /= scaler;
            m_like += 4;
        }

        scaler = (CLFlt)log(scaler);
        scPNew[char_idx] = scaler;
        CLFlt dif = lnScaler[char_idx];
        dif += scaler;

        m_like = &s_like[tid*17];
        a1=0.0; c1=0.0; g1=0.0; t1=0.0;
        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_like[0]; a1 += a0; g_clP[0] = a0;
            c0 = m_like[1]; c1 += c0; g_clP[1] = c0;
            g0 = m_like[2]; g1 += g0; g_clP[2] = g0;
            t0 = m_like[3]; t1 += t0; g_clP[3] = t0;
            m_like += 4;
            g_clP += 4;
        }

        MrBFlt like = a1 * bs_A + c1 * bs_C + g1 * bs_G + t1 * bs_T;
        MrBFlt *g_clInvar = &clInvar[bid_x*80*4 + 4*tid];
        likelihood_NUC4_hasPInvar_YES_e(like, g_clInvar, dif, nSitesOfPat, lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, char_idx);
        lnScaler[char_idx] = dif;
    }
}

__global__ void gpu_root_4_3_e(CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *clA, CLFlt *tiPL, CLFlt *tiPR, CLFlt *tiPA, CLFlt *lnScaler, CLFlt *scPOld, CLFlt *scPNew, MrBFlt *clInvar, CLFlt *nSitesOfPat, MrBFlt *lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];
    __shared__ CLFlt s_tiPA[64];

    if(tid < 64)
    {
        s_tiPL[tid] = tiPL[tid];
        s_tiPR[tid] = tiPR[tid];
        s_tiPA[tid] = tiPA[tid];
    }

    __shared__ CLFlt s_like[80*17];
    CLFlt *m_like = &s_like[17*tid];

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1, scaler = 0.0;
        CLFlt *m_tiPL = &s_tiPL[0];
        CLFlt *m_tiPR = &s_tiPR[0];
        CLFlt *m_tiPA = &s_tiPA[0];
        int clx_offset = bid_x*80*16 + 16*tid;
        CLFlt *g_clL = &clL[clx_offset];
        CLFlt *g_clR = &clR[clx_offset];
        CLFlt *g_clA = &clA[clx_offset];
        CLFlt *g_clP = &clP[clx_offset];
        int gamma;

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = g_clL[0]; c0 = g_clL[1]; g0 = g_clL[2]; t0 = g_clL[3];

            a1 = m_tiPL[0]*a0 + m_tiPL[1]*c0 + m_tiPL[2]*g0 + m_tiPL[3]*t0;
            c1 = m_tiPL[4]*a0 + m_tiPL[5]*c0 + m_tiPL[6]*g0 + m_tiPL[7]*t0;
            g1 = m_tiPL[8]*a0 + m_tiPL[9]*c0 + m_tiPL[10]*g0 + m_tiPL[11]*t0;
            t1 = m_tiPL[12]*a0 + m_tiPL[13]*c0 + m_tiPL[14]*g0 + m_tiPL[15]*t0;

            a0 = g_clR[0]; c0 = g_clR[1]; g0 = g_clR[2]; t0 = g_clR[3];

            a1 *= m_tiPR[0]*a0 + m_tiPR[1]*c0 + m_tiPR[2]*g0 + m_tiPR[3]*t0;
            c1 *= m_tiPR[4]*a0 + m_tiPR[5]*c0 + m_tiPR[6]*g0 + m_tiPR[7]*t0;
            g1 *= m_tiPR[8]*a0 + m_tiPR[9]*c0 + m_tiPR[10]*g0 + m_tiPR[11]*t0;
            t1 *= m_tiPR[12]*a0 + m_tiPR[13]*c0 + m_tiPR[14]*g0 + m_tiPR[15]*t0;

            a0 = g_clA[0]; c0 = g_clA[1]; g0 = g_clA[2]; t0 = g_clA[3];

            a1 *= m_tiPA[0]*a0 + m_tiPA[1]*c0 + m_tiPA[2]*g0 + m_tiPA[3]*t0;
            c1 *= m_tiPA[4]*a0 + m_tiPA[5]*c0 + m_tiPA[6]*g0 + m_tiPA[7]*t0;
            g1 *= m_tiPA[8]*a0 + m_tiPA[9]*c0 + m_tiPA[10]*g0 + m_tiPA[11]*t0;
            t1 *= m_tiPA[12]*a0 + m_tiPA[13]*c0 + m_tiPA[14]*g0 + m_tiPA[15]*t0;

            scaler = MAX(scaler, a1);
            scaler = MAX(scaler, c1);
            scaler = MAX(scaler, g1);
            scaler = MAX(scaler, t1);

            m_like[0] = a1;
            m_like[1] = c1;
            m_like[2] = g1;
            m_like[3] = t1;

            g_clL += 4;
            g_clR += 4;
            g_clA += 4;

            m_tiPL += 16;
            m_tiPR += 16;
            m_tiPA += 16;
            m_like += 4;
        }

        m_like = &s_like[tid*17];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            m_like[0] /= scaler;
            m_like[1] /= scaler;
            m_like[2] /= scaler;
            m_like[3] /= scaler;
            m_like += 4;
        }

        scaler = (CLFlt)log(scaler);
        scPNew[char_idx] = scaler;
        CLFlt dif = lnScaler[char_idx];
        dif -= scPOld[char_idx];
        dif += scaler;

        m_like = &s_like[tid*17];
        a1=0.0; c1=0.0; g1=0.0; t1=0.0;
        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_like[0]; a1 += a0; g_clP[0] = a0;
            c0 = m_like[1]; c1 += c0; g_clP[1] = c0;
            g0 = m_like[2]; g1 += g0; g_clP[2] = g0;
            t0 = m_like[3]; t1 += t0; g_clP[3] = t0;
            m_like += 4;
            g_clP += 4;
        }

        MrBFlt like = a1 * bs_A + c1 * bs_C + g1 * bs_G + t1 * bs_T;
        MrBFlt *g_clInvar = &clInvar[bid_x*80*4 + 4*tid];
        likelihood_NUC4_hasPInvar_YES_e(like, g_clInvar, dif, nSitesOfPat, lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, char_idx);
        lnScaler[char_idx] = dif;
    }
}

static void root_4_0(int offset_clP, int offset_clL, int offset_clR, int offset_clA, int offset_pL, int offset_pR, int offset_pA, int offset_lnScaler, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int  modelnumChars, int modelnumGammaCats, int chain)
{
    int nThread = 80;
    int gx = modelnumChars / nThread;
    int residue = modelnumChars % nThread;
    if (residue != 0)
        gx += 1;

    dim3	dimGrid(gx, 1, 1);
    dim3	dimBlock(4, 5, 4);

    gpu_root_4_0_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devchainCondLikes+offset_clA, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtiProbSpace+offset_pA, devtreeScalerSpace+offset_lnScaler, devinvCondLikes+offset_clInvar, devnumSitesOfPat+offset_nSitesOfPat, devlnL+offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars);

    cutilCheckMsg("gpu_root_4_0_e failed\n");
}

static void root_4_1(int offset_clP, int offset_clL, int offset_clR, int offset_clA, int offset_pL, int offset_pR, int offset_pA, int offset_lnScaler, int offset_scPOld, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int  modelnumChars, int modelnumGammaCats, int chain)
{
    int nThread = 80;
    int gx = modelnumChars / nThread;
    int residue = modelnumChars % nThread;
    if (residue != 0)
        gx += 1;

    dim3	dimGrid(gx, 1, 1);
    dim3	dimBlock(4, 5, 4);

    gpu_root_4_1_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devchainCondLikes+offset_clA, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtiProbSpace+offset_pA, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPOld, devinvCondLikes+offset_clInvar, devnumSitesOfPat+offset_nSitesOfPat, devlnL+offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars);

    cutilCheckMsg("gpu_root_4_1_e failed\n");
}


static void root_4_2(int offset_clP, int offset_clL, int offset_clR, int offset_clA, int offset_pL, int offset_pR, int offset_pA, int offset_lnScaler, int offset_scPNew, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int  modelnumChars, int modelnumGammaCats, int chain)
{
    int nThread = 80;
    int gx = modelnumChars / nThread;
    int residue = modelnumChars % nThread;
    if (residue != 0)
        gx += 1;

    dim3	dimGrid(gx, 1, 1);
    dim3	dimBlock(4, 5, 4);

    gpu_root_4_2_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devchainCondLikes+offset_clA, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtiProbSpace+offset_pA, devtreeScalerSpace+offset_lnScaler, devnodeScalerSpace+offset_scPNew, devinvCondLikes+offset_clInvar, devnumSitesOfPat+offset_nSitesOfPat, devlnL+offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars);

    cutilCheckMsg("gpu_root_4_2_e failed\n");
}

static void root_4_3(int offset_clP, int offset_clL, int offset_clR, int offset_clA, int offset_pL, int offset_pR, int offset_pA, int offset_lnScaler, int offset_scPOld, int offset_scPNew, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int  modelnumChars, int modelnumGammaCats, int chain)
{
    int nThread = 80;
    int gx = modelnumChars / nThread;
    int residue = modelnumChars % nThread;
    if (residue != 0)
        gx += 1;

    dim3	dimGrid(gx, 1, 1);
    dim3	dimBlock(4, 5, 4);

    gpu_root_4_3_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devchainCondLikes+offset_clA, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtiProbSpace+offset_pA, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPOld, devnodeScalerSpace+offset_scPNew, devinvCondLikes+offset_clInvar, devnumSitesOfPat+offset_nSitesOfPat, devlnL+offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars);

    cutilCheckMsg("gpu_root_4_3_e failed\n");
}


extern "C" void cuda_root_4(int offset_clP, int offset_clL, int offset_clR, int offset_clA, int offset_pL, int offset_pR, int offset_pA,  int offset_scPOld, int offset_scPNew,  int offset_lnScaler, int scaler, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int hasPInvar, int modelnumChars, int modelnumGammaCats, int chain)
{
    if(hasPInvar == NO)
    {
        fprintf(stderr, "tgMC3 was designed for data having invariable sites\n");
        exit(1);
    }

    switch(scaler)
    {
        case 0:

            root_4_0(offset_clP, offset_clL, offset_clR, offset_clA, offset_pL, offset_pR, offset_pA, offset_lnScaler, offset_clInvar, offset_nSitesOfPat, offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars, modelnumGammaCats, chain);

            break;

        case 1:

            root_4_1(offset_clP, offset_clL, offset_clR, offset_clA, offset_pL, offset_pR, offset_pA, offset_lnScaler, offset_scPOld, offset_clInvar, offset_nSitesOfPat, offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars, modelnumGammaCats, chain);

            break;

        case 2:

            root_4_2(offset_clP, offset_clL, offset_clR, offset_clA, offset_pL, offset_pR, offset_pA, offset_lnScaler, offset_scPNew, offset_clInvar, offset_nSitesOfPat, offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars, modelnumGammaCats, chain);

            break;

        case 3:

            root_4_3(offset_clP, offset_clL, offset_clR, offset_clA, offset_pL, offset_pR, offset_pA, offset_lnScaler, offset_scPOld, offset_scPNew, offset_clInvar, offset_nSitesOfPat, offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars, modelnumGammaCats, chain);

    }
}

__global__ void gpu_root_03_0_e(CLFlt*clP, CLFlt *clL, CLFlt *clR, CLFlt *tiPL, CLFlt *tiPR, int *aState, CLFlt *tiPA, CLFlt *lnScaler, MrBFlt *clInvar, CLFlt *nSitesOfPat, MrBFlt *lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];

    if(tid < 64)
    {
        s_tiPL[tid] = tiPL[tid];
        s_tiPR[tid] = tiPR[tid];
    }

    __shared__ CLFlt s_preLikeA[80];

    if(tid_y < 4)
    {
        int tip_idx = 16 * tid_z + tid_y + 4 * tid_x;
        s_preLikeA[tid] = tiPA[tip_idx];
    }
    else
    {
        s_preLikeA[tid] = 1.0f;
    }

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1;
        CLFlt like_a=0.0, like_c=0.0, like_g=0.0, like_t=0.0;
        int clx_offset = bid_x*80*16 + 16*tid;
        CLFlt *g_clL = &clL[clx_offset];
        CLFlt *g_clR = &clR[clx_offset];
        CLFlt *g_clP = &clP[clx_offset];
        CLFlt *m_tiPL = &s_tiPL[0];
        CLFlt *m_tiPR = &s_tiPR[0];
        MrBFlt like = 0.0;
        int gamma, ai = aState[char_idx];

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = g_clL[0]; c0 = g_clL[1]; g0 = g_clL[2]; t0 = g_clL[3];

            a1 = (m_tiPL[0]*a0 + m_tiPL[1]*c0 + m_tiPL[2]*g0 + m_tiPL[3]*t0);
            c1 = (m_tiPL[4]*a0 + m_tiPL[5]*c0 + m_tiPL[6]*g0 + m_tiPL[7]*t0);
            g1 = (m_tiPL[8]*a0 + m_tiPL[9]*c0 + m_tiPL[10]*g0 + m_tiPL[11]*t0);
            t1 = (m_tiPL[12]*a0 + m_tiPL[13]*c0 + m_tiPL[14]*g0 + m_tiPL[15]*t0);

            a0 = g_clR[0]; c0 = g_clR[1]; g0 = g_clR[2]; t0 = g_clR[3];

            a1 *= (m_tiPR[0]*a0 + m_tiPR[1]*c0 + m_tiPR[2]*g0 + m_tiPR[3]*t0);
            c1 *= (m_tiPR[4]*a0 + m_tiPR[5]*c0 + m_tiPR[6]*g0 + m_tiPR[7]*t0);
            g1 *= (m_tiPR[8]*a0 + m_tiPR[9]*c0 + m_tiPR[10]*g0 + m_tiPR[11]*t0);
            t1 *= (m_tiPR[12]*a0 + m_tiPR[13]*c0 + m_tiPR[14]*g0 + m_tiPR[15]*t0);

            a1 *= s_preLikeA[ai];
            c1 *= s_preLikeA[ai+1];
            g1 *= s_preLikeA[ai+2];
            t1 *= s_preLikeA[ai+3];

            like_a += a1;
            like_c += c1;
            like_g += g1;
            like_t += t1;

            g_clP[0] = a1;
            g_clP[1] = c1;
            g_clP[2] = g1;
            g_clP[3] = t1;
            g_clP += 4;

            g_clL += 4;
            g_clR += 4;
            m_tiPL += 16;
            m_tiPR += 16;
            ai += 20;
        }

        like = like_a * bs_A + like_c * bs_C + like_g * bs_G + like_t * bs_T;

        MrBFlt *g_clInvar = &clInvar[bid_x*80*4 + 4*tid];

        likelihood_NUC4_hasPInvar_YES_e(like, g_clInvar, lnScaler[char_idx], nSitesOfPat, lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, char_idx);
    }
}

__global__ void gpu_root_03_1_e(CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *tiPL, CLFlt *tiPR, int *aState, CLFlt *tiPA, CLFlt *lnScaler, CLFlt *scPOld, MrBFlt *clInvar, CLFlt *nSitesOfPat, MrBFlt *lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];

    if(tid < 64)
    {
        s_tiPL[tid] = tiPL[tid];
        s_tiPR[tid] = tiPR[tid];
    }

    __shared__ CLFlt s_preLikeA[80];

    if(tid_y < 4)
    {
        int tip_idx = 16 * tid_z + tid_y + 4 * tid_x;
        s_preLikeA[tid] = tiPA[tip_idx];
    }
    else
    {
        s_preLikeA[tid] = 1.0f;
    }

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1;
        CLFlt like_a=0.0, like_c=0.0, like_g=0.0, like_t=0.0;
        int clx_offset = bid_x*80*16 + 16*tid;
        CLFlt *g_clL = &clL[clx_offset];
        CLFlt *g_clR = &clR[clx_offset];
        CLFlt *g_clP = &clP[clx_offset];
        CLFlt *m_tiPL = &s_tiPL[0];
        CLFlt *m_tiPR = &s_tiPR[0];
        int gamma, ai = aState[char_idx];

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = g_clL[0]; c0 = g_clL[1]; g0 = g_clL[2]; t0 = g_clL[3];

            a1 = (m_tiPL[0]*a0 + m_tiPL[1]*c0 + m_tiPL[2]*g0 + m_tiPL[3]*t0);
            c1 = (m_tiPL[4]*a0 + m_tiPL[5]*c0 + m_tiPL[6]*g0 + m_tiPL[7]*t0);
            g1 = (m_tiPL[8]*a0 + m_tiPL[9]*c0 + m_tiPL[10]*g0 + m_tiPL[11]*t0);
            t1 = (m_tiPL[12]*a0 + m_tiPL[13]*c0 + m_tiPL[14]*g0 + m_tiPL[15]*t0);

            a0 = g_clR[0]; c0 = g_clR[1]; g0 = g_clR[2]; t0 = g_clR[3];

            a1 *= (m_tiPR[0]*a0 + m_tiPR[1]*c0 + m_tiPR[2]*g0 + m_tiPR[3]*t0);
            c1 *= (m_tiPR[4]*a0 + m_tiPR[5]*c0 + m_tiPR[6]*g0 + m_tiPR[7]*t0);
            g1 *= (m_tiPR[8]*a0 + m_tiPR[9]*c0 + m_tiPR[10]*g0 + m_tiPR[11]*t0);
            t1 *= (m_tiPR[12]*a0 + m_tiPR[13]*c0 + m_tiPR[14]*g0 + m_tiPR[15]*t0);

            a1 *= s_preLikeA[ai];
            c1 *= s_preLikeA[ai+1];
            g1 *= s_preLikeA[ai+2];
            t1 *= s_preLikeA[ai+3];

            like_a += a1;
            like_c += c1;
            like_g += g1;
            like_t += t1;

            g_clP[0] = a1;
            g_clP[1] = c1;
            g_clP[2] = g1;
            g_clP[3] = t1;
            g_clP += 4;

            g_clL += 4;
            g_clR += 4;
            m_tiPL += 16;
            m_tiPR += 16;
            ai += 20;
        }

        CLFlt dif = lnScaler[char_idx];
        dif -= scPOld[char_idx];

        MrBFlt like = like_a * bs_A + like_c * bs_C + like_g * bs_G + like_t * bs_T;
        MrBFlt *g_clInvar = &clInvar[bid_x*80*4 + 4*tid];
        likelihood_NUC4_hasPInvar_YES_e(like, g_clInvar, dif, nSitesOfPat, lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, char_idx);
        lnScaler[char_idx] = dif;
    }
}

__global__ void gpu_root_03_2_e(CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *tiPL, CLFlt *tiPR, int *aState, CLFlt *tiPA, CLFlt *lnScaler, CLFlt *scPNew, MrBFlt *clInvar, CLFlt *nSitesOfPat, MrBFlt *lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];

    if(tid < 64)
    {
        s_tiPL[tid] = tiPL[tid];
        s_tiPR[tid] = tiPR[tid];
    }

    __shared__ CLFlt s_preLikeA[80];

    if(tid_y < 4)
    {
        int tip_idx = 16 * tid_z + tid_y + 4 * tid_x;
        s_preLikeA[tid] = tiPA[tip_idx];
    }
    else
    {
        s_preLikeA[tid] = 1.0f;
    }

    __shared__ CLFlt s_like[80*17];
    CLFlt *m_like = &s_like[17*tid];

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1, scaler;
        int clx_offset = bid_x*80*16 + 16*tid;
        CLFlt *g_clL = &clL[clx_offset];
        CLFlt *g_clR = &clR[clx_offset];
        CLFlt *g_clP = &clP[clx_offset];
        CLFlt *m_tiPL = &s_tiPL[0];
        CLFlt *m_tiPR = &s_tiPR[0];
        int gamma, ai = aState[char_idx];

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = g_clL[0]; c0 = g_clL[1]; g0 = g_clL[2]; t0 = g_clL[3];

            a1 = (m_tiPL[0]*a0 + m_tiPL[1]*c0 + m_tiPL[2]*g0 + m_tiPL[3]*t0);
            c1 = (m_tiPL[4]*a0 + m_tiPL[5]*c0 + m_tiPL[6]*g0 + m_tiPL[7]*t0);
            g1 = (m_tiPL[8]*a0 + m_tiPL[9]*c0 + m_tiPL[10]*g0 + m_tiPL[11]*t0);
            t1 = (m_tiPL[12]*a0 + m_tiPL[13]*c0 + m_tiPL[14]*g0 + m_tiPL[15]*t0);

            a0 = g_clR[0]; c0 = g_clR[1]; g0 = g_clR[2]; t0 = g_clR[3];

            a1 *= (m_tiPR[0]*a0 + m_tiPR[1]*c0 + m_tiPR[2]*g0 + m_tiPR[3]*t0);
            c1 *= (m_tiPR[4]*a0 + m_tiPR[5]*c0 + m_tiPR[6]*g0 + m_tiPR[7]*t0);
            g1 *= (m_tiPR[8]*a0 + m_tiPR[9]*c0 + m_tiPR[10]*g0 + m_tiPR[11]*t0);
            t1 *= (m_tiPR[12]*a0 + m_tiPR[13]*c0 + m_tiPR[14]*g0 + m_tiPR[15]*t0);

            a1 *= s_preLikeA[ai];
            c1 *= s_preLikeA[ai+1];
            g1 *= s_preLikeA[ai+2];
            t1 *= s_preLikeA[ai+3];

            scaler = MAX(scaler, a1);
            scaler = MAX(scaler, c1);
            scaler = MAX(scaler, g1);
            scaler = MAX(scaler, t1);

            m_like[0] = a1;
            m_like[1] = c1;
            m_like[2] = g1;
            m_like[3] = t1;

            g_clL += 4;
            g_clR += 4;
            m_tiPL += 16;
            m_tiPR += 16;
            m_like += 4;
            ai += 20;
        }

        m_like = &s_like[tid*17];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            m_like[0] /= scaler;
            m_like[1] /= scaler;
            m_like[2] /= scaler;
            m_like[3] /= scaler;
            m_like += 4;
        }

        scaler = (CLFlt)log(scaler);
        scPNew[char_idx] = scaler;
        CLFlt dif = lnScaler[char_idx];
        dif += scaler;

        m_like = &s_like[tid*17];
        a1=0.0; c1=0.0; g1=0.0; t1=0.0;
        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_like[0]; a1 += a0; g_clP[0] = a0;
            c0 = m_like[1]; c1 += c0; g_clP[1] = c0;
            g0 = m_like[2]; g1 += g0; g_clP[2] = g0;
            t0 = m_like[3]; t1 += t0; g_clP[3] = t0;
            m_like += 4;
            g_clP += 4;
        }

        MrBFlt like = a1 * bs_A + c1 * bs_C + g1 * bs_G + t1 * bs_T;
        MrBFlt *g_clInvar = &clInvar[bid_x*80*4 + 4*tid];
        likelihood_NUC4_hasPInvar_YES_e(like, g_clInvar, dif, nSitesOfPat, lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, char_idx);
        lnScaler[char_idx] = dif;
    }
}

__global__ void gpu_root_03_3_e(CLFlt *clP, CLFlt *clL, CLFlt *clR, CLFlt *tiPL, CLFlt *tiPR, int *aState, CLFlt *tiPA, CLFlt *lnScaler, CLFlt *scPOld, CLFlt *scPNew, MrBFlt *clInvar, CLFlt *nSitesOfPat, MrBFlt *lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPL[64];
    __shared__ CLFlt s_tiPR[64];

    if(tid < 64)
    {
        s_tiPL[tid] = tiPL[tid];
        s_tiPR[tid] = tiPR[tid];
    }

    __shared__ CLFlt s_preLikeA[80];

    if(tid_y < 4)
    {
        int tip_idx = 16 * tid_z + tid_y + 4 * tid_x;
        s_preLikeA[tid] = tiPA[tip_idx];
    }
    else
    {
        s_preLikeA[tid] = 1.0f;
    }

    __shared__ CLFlt s_like[80*17];
    CLFlt *m_like = &s_like[17*tid];

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1, scaler = 0.0;
        int clx_offset = bid_x*80*16 + 16*tid;
        CLFlt *g_clL = &clL[clx_offset];
        CLFlt *g_clR = &clR[clx_offset];
        CLFlt *g_clP = &clP[clx_offset];
        CLFlt *m_tiPL = &s_tiPL[0];
        CLFlt *m_tiPR = &s_tiPR[0];
        int gamma, ai = aState[char_idx];

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = g_clL[0]; c0 = g_clL[1]; g0 = g_clL[2]; t0 = g_clL[3];

            a1 = (m_tiPL[0]*a0 + m_tiPL[1]*c0 + m_tiPL[2]*g0 + m_tiPL[3]*t0);
            c1 = (m_tiPL[4]*a0 + m_tiPL[5]*c0 + m_tiPL[6]*g0 + m_tiPL[7]*t0);
            g1 = (m_tiPL[8]*a0 + m_tiPL[9]*c0 + m_tiPL[10]*g0 + m_tiPL[11]*t0);
            t1 = (m_tiPL[12]*a0 + m_tiPL[13]*c0 + m_tiPL[14]*g0 + m_tiPL[15]*t0);

            a0 = g_clR[0]; c0 = g_clR[1]; g0 = g_clR[2]; t0 = g_clR[3];

            a1 *= (m_tiPR[0]*a0 + m_tiPR[1]*c0 + m_tiPR[2]*g0 + m_tiPR[3]*t0);
            c1 *= (m_tiPR[4]*a0 + m_tiPR[5]*c0 + m_tiPR[6]*g0 + m_tiPR[7]*t0);
            g1 *= (m_tiPR[8]*a0 + m_tiPR[9]*c0 + m_tiPR[10]*g0 + m_tiPR[11]*t0);
            t1 *= (m_tiPR[12]*a0 + m_tiPR[13]*c0 + m_tiPR[14]*g0 + m_tiPR[15]*t0);

            a1 *= s_preLikeA[ai];
            c1 *= s_preLikeA[ai+1];
            g1 *= s_preLikeA[ai+2];
            t1 *= s_preLikeA[ai+3];

            scaler = MAX(scaler, a1);
            scaler = MAX(scaler, c1);
            scaler = MAX(scaler, g1);
            scaler = MAX(scaler, t1);

            m_like[0] = a1;
            m_like[1] = c1;
            m_like[2] = g1;
            m_like[3] = t1;

            g_clL += 4;
            g_clR += 4;
            m_tiPL += 16;
            m_tiPR += 16;
            m_like += 4;
            ai += 20;
        }

        m_like = &s_like[tid*17];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            m_like[0] /= scaler;
            m_like[1] /= scaler;
            m_like[2] /= scaler;
            m_like[3] /= scaler;
            m_like += 4;
        }

        scaler = (CLFlt)log(scaler);
        scPNew[char_idx] = scaler;
        CLFlt dif = lnScaler[char_idx];
        dif -= scPOld[char_idx];
        dif += scaler;

        m_like = &s_like[tid*17];
        a1=0.0; c1=0.0; g1=0.0; t1=0.0;
        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_like[0]; a1 += a0; g_clP[0] = a0;
            c0 = m_like[1]; c1 += c0; g_clP[1] = c0;
            g0 = m_like[2]; g1 += g0; g_clP[2] = g0;
            t0 = m_like[3]; t1 += t0; g_clP[3] = t0;
            m_like += 4;
            g_clP += 4;
        }

        MrBFlt like = a1 * bs_A + c1 * bs_C + g1 * bs_G + t1 * bs_T;
        MrBFlt *g_clInvar = &clInvar[bid_x*80*4 + 4*tid];
        likelihood_NUC4_hasPInvar_YES_e(like, g_clInvar, dif, nSitesOfPat, lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, char_idx);
        lnScaler[char_idx] = dif;
    }
}

static void root_03_0(int offset_clP, int offset_clL, int offset_clR, int offset_pL, int offset_pR, int offset_aState, int offset_pA, int offset_lnScaler, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars, int modelnumGammaCats, int chain)
{
    int nThread = 80;
    int gx = modelnumChars / nThread;
    int residue = modelnumChars % nThread;
    if (residue != 0)
        gx += 1;

    dim3	dimGrid(gx, 1, 1);
    dim3	dimBlock(4, 5, 4);

    gpu_root_03_0_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtermState+offset_aState, devtiProbSpace+offset_pA, devtreeScalerSpace+offset_lnScaler, devinvCondLikes+offset_clInvar, devnumSitesOfPat+offset_nSitesOfPat, devlnL+offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars);

    cutilCheckMsg("gpu_root_03_0_e failed\n");
}

static void root_03_1(int offset_clP, int offset_clL, int offset_clR, int offset_pL, int offset_pR, int offset_aState, int offset_pA, int offset_lnScaler, int offset_scPOld, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars, int modelnumGammaCats, int chain)
{
    int nThread = 80;
    int gx = modelnumChars / nThread;
    int residue = modelnumChars % nThread;
    if (residue != 0)
        gx += 1;

    dim3	dimGrid(gx, 1, 1);
    dim3	dimBlock(4, 5, 4);

    gpu_root_03_1_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtermState+offset_aState, devtiProbSpace+offset_pA, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPOld, devinvCondLikes+offset_clInvar, devnumSitesOfPat+offset_nSitesOfPat, devlnL+offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars);

    cutilCheckMsg("gpu_root_03_1_e failed\n");
}

static void root_03_2(int offset_clP, int offset_clL, int offset_clR, int offset_pL, int offset_pR, int offset_aState, int offset_pA, int offset_lnScaler, int offset_scPNew, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars, int modelnumGammaCats, int chain)
{
    int nThread = 80;
    int gx = modelnumChars / nThread;
    int residue = modelnumChars % nThread;
    if (residue != 0)
        gx += 1;

    dim3	dimGrid(gx, 1, 1);
    dim3	dimBlock(4, 5, 4);

    gpu_root_03_2_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtermState+offset_aState, devtiProbSpace+offset_pA, devtreeScalerSpace+offset_lnScaler, devnodeScalerSpace+offset_scPNew, devinvCondLikes+offset_clInvar, devnumSitesOfPat+offset_nSitesOfPat, devlnL+offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars);

    cutilCheckMsg("gpu_root_03_2_e failed\n");
}

static void root_03_3(int offset_clP, int offset_clL, int offset_clR, int offset_pL, int offset_pR, int offset_aState, int offset_pA, int offset_lnScaler, int offset_scPOld, int offset_scPNew, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars, int modelnumGammaCats, int chain)
{
    int nThread = 80;
    int gx = modelnumChars / nThread;
    int residue = modelnumChars % nThread;
    if (residue != 0)
        gx += 1;

    dim3	dimGrid(gx, 1, 1);
    dim3	dimBlock(4, 5, 4);

    gpu_root_03_3_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clL, devchainCondLikes+offset_clR, devtiProbSpace+offset_pL, devtiProbSpace+offset_pR, devtermState+offset_aState, devtiProbSpace+offset_pA, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPOld, devnodeScalerSpace+offset_scPNew, devinvCondLikes+offset_clInvar, devnumSitesOfPat+offset_nSitesOfPat, devlnL+offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars);

    cutilCheckMsg("gpu_root_03_3_e failed\n");
}

extern "C" void cuda_root_03 (int offset_clP, int offset_clL, int offset_clR, int offset_pL, int offset_pR, int offset_aState, int offset_pA,  int offset_scPOld, int offset_scPNew, int offset_lnScaler, int scaler, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A,MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int hasPInvar, int modelnumChars, int modelnumGammaCats, int chain)
{
    if(hasPInvar == NO)
    {
        fprintf(stderr, "tgMC3 was designed for data having invariable sites\n");
        exit(1);
    }

    switch(scaler)
    {
        case 0:

            root_03_0(offset_clP, offset_clL, offset_clR, offset_pL, offset_pR, offset_aState, offset_pA, offset_lnScaler, offset_clInvar, offset_nSitesOfPat, offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars, modelnumGammaCats, chain);

            break;

        case 1:

            root_03_1(offset_clP, offset_clL, offset_clR, offset_pL, offset_pR, offset_aState, offset_pA, offset_lnScaler, offset_scPOld, offset_clInvar, offset_nSitesOfPat, offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars, modelnumGammaCats, chain);

            break;

        case 2:

            root_03_2(offset_clP, offset_clL, offset_clR, offset_pL, offset_pR, offset_aState, offset_pA, offset_lnScaler, offset_scPNew, offset_clInvar, offset_nSitesOfPat, offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars, modelnumGammaCats, chain);

            break;

        case 3:

            root_03_3(offset_clP, offset_clL, offset_clR, offset_pL, offset_pR, offset_aState, offset_pA, offset_lnScaler, offset_scPOld, offset_scPNew, offset_clInvar, offset_nSitesOfPat, offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars, modelnumGammaCats, chain);

    }
}

__global__ void gpu_root_12_0_e(CLFlt *clP, CLFlt *clx, CLFlt *tiPX, int *xState, int *aState, CLFlt *ti_preLikeX, CLFlt *tiPA, CLFlt *lnScaler, MrBFlt *clInvar, CLFlt *nSitesOfPat, MrBFlt *lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPX[64];

    if(tid < 64)
    {
        s_tiPX[tid] = tiPX[tid];
    }

    __shared__ CLFlt s_preLikeX[80];
    __shared__ CLFlt s_preLikeA[80];

    if(tid_y < 4)
    {
        int tip_idx = 16 * tid_z + tid_y + 4 * tid_x;
        s_preLikeA[tid] = tiPA[tip_idx];
        s_preLikeX[tid] = ti_preLikeX[tip_idx];
    }
    else
    {
        s_preLikeA[tid] = 1.0f;
        s_preLikeX[tid] = 1.0f;
    }

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1;
        CLFlt like_a=0.0, like_c=0.0, like_g=0.0, like_t=0.0;
        int xi, ai, gamma;
        xi = xState[char_idx];
        ai = aState[char_idx];
        int clx_offset = bid_x*80*16 + 16 * tid;
        CLFlt *g_clx = &clx[clx_offset];
        CLFlt *g_clP = &clP[clx_offset];
        CLFlt *m_tiPX = &s_tiPX[0];

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = g_clx[0]; c0 = g_clx[1]; g0 = g_clx[2]; t0 = g_clx[3];

            a1 = m_tiPX[0]*a0 + m_tiPX[1]*c0 + m_tiPX[2]*g0 + m_tiPX[3]*t0;
            c1 = m_tiPX[4]*a0 + m_tiPX[5]*c0 + m_tiPX[6]*g0 + m_tiPX[7]*t0;
            g1 = m_tiPX[8]*a0 + m_tiPX[9]*c0 + m_tiPX[10]*g0 + m_tiPX[11]*t0;
            t1 = m_tiPX[12]*a0 + m_tiPX[13]*c0 + m_tiPX[14]*g0 + m_tiPX[15]*t0;

            a1 *= s_preLikeX[xi] * s_preLikeA[ai];
            c1 *= s_preLikeX[xi+1] * s_preLikeA[ai+1];
            g1 *= s_preLikeX[xi+2] * s_preLikeA[ai+2];
            t1 *= s_preLikeX[xi+3] * s_preLikeA[ai+3];

            like_a += a1;
            like_c += c1;
            like_g += g1;
            like_t += t1;

            g_clP[0] = a1;
            g_clP[1] = c1;
            g_clP[2] = g1;
            g_clP[3] = t1;
            g_clP += 4;

            g_clx += 4;
            m_tiPX += 16;
            xi += 20;
            ai += 20;
        }

        MrBFlt like = like_a * bs_A + like_c * bs_C + like_g * bs_G + like_t * bs_T;
        MrBFlt *g_clInvar = &clInvar[bid_x*80*4 + 4*tid];
        likelihood_NUC4_hasPInvar_YES_e(like, g_clInvar, lnScaler[char_idx], nSitesOfPat, lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, char_idx);
    }
}

__global__ void gpu_root_12_1_e(CLFlt *clP, CLFlt *clx, CLFlt *tiPX, int *xState, int *aState, CLFlt *ti_preLikeX, CLFlt *tiPA, CLFlt *lnScaler, CLFlt *scPOld, MrBFlt *clInvar, CLFlt *nSitesOfPat, MrBFlt *lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPX[64];

    if(tid < 64)
    {
        s_tiPX[tid] = tiPX[tid];
    }

    __shared__ CLFlt s_preLikeX[80];
    __shared__ CLFlt s_preLikeA[80];

    if(tid_y < 4)
    {
        int tip_idx = 16 * tid_z + tid_y + 4 * tid_x;
        s_preLikeA[tid] = tiPA[tip_idx];
        s_preLikeX[tid] = ti_preLikeX[tip_idx];
    }
    else
    {
        s_preLikeA[tid] = 1.0f;
        s_preLikeX[tid] = 1.0f;
    }

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1;
        CLFlt like_a=0.0, like_c=0.0, like_g=0.0, like_t=0.0;
        int xi, ai, gamma;
        xi = xState[char_idx];
        ai = aState[char_idx];
        int clx_offset = bid_x*80*16 + 16 * tid;
        CLFlt *g_clx = &clx[clx_offset];
        CLFlt *g_clP = &clP[clx_offset];
        CLFlt *m_tiPX = &s_tiPX[0];

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = g_clx[0]; c0 = g_clx[1]; g0 = g_clx[2]; t0 = g_clx[3];

            a1 = m_tiPX[0]*a0 + m_tiPX[1]*c0 + m_tiPX[2]*g0 + m_tiPX[3]*t0;
            c1 = m_tiPX[4]*a0 + m_tiPX[5]*c0 + m_tiPX[6]*g0 + m_tiPX[7]*t0;
            g1 = m_tiPX[8]*a0 + m_tiPX[9]*c0 + m_tiPX[10]*g0 + m_tiPX[11]*t0;
            t1 = m_tiPX[12]*a0 + m_tiPX[13]*c0 + m_tiPX[14]*g0 + m_tiPX[15]*t0;

            a1 *= s_preLikeX[xi] * s_preLikeA[ai];
            c1 *= s_preLikeX[xi+1] * s_preLikeA[ai+1];
            g1 *= s_preLikeX[xi+2] * s_preLikeA[ai+2];
            t1 *= s_preLikeX[xi+3] * s_preLikeA[ai+3];

            like_a += a1;
            like_c += c1;
            like_g += g1;
            like_t += t1;

            g_clP[0] = a1;
            g_clP[1] = c1;
            g_clP[2] = g1;
            g_clP[3] = t1;
            g_clP += 4;

            g_clx += 4;
            m_tiPX += 16;
            xi += 20;
            ai += 20;
        }

        CLFlt dif = lnScaler[char_idx];
        dif -= scPOld[char_idx];
        MrBFlt like = like_a * bs_A + like_c * bs_C + like_g * bs_G + like_t * bs_T;
        MrBFlt *g_clInvar = &clInvar[bid_x*80*4 + 4*tid];
        likelihood_NUC4_hasPInvar_YES_e(like, g_clInvar, dif, nSitesOfPat, lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, char_idx);

        lnScaler[char_idx] = dif;
    }
}

__global__ void gpu_root_12_2_e(CLFlt *clP, CLFlt *clx, CLFlt *tiPX, int *xState, int *aState, CLFlt *ti_preLikeX, CLFlt *tiPA, CLFlt *lnScaler, CLFlt *scPNew, MrBFlt *clInvar, CLFlt *nSitesOfPat, MrBFlt *lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPX[64];

    if(tid < 64)
    {
        s_tiPX[tid] = tiPX[tid];
    }

    __shared__ CLFlt s_preLikeX[80];
    __shared__ CLFlt s_preLikeA[80];

    if(tid_y < 4)
    {
        int tip_idx = 16 * tid_z + tid_y + 4 * tid_x;
        s_preLikeA[tid] = tiPA[tip_idx];
        s_preLikeX[tid] = ti_preLikeX[tip_idx];
    }
    else
    {
        s_preLikeA[tid] = 1.0f;
        s_preLikeX[tid] = 1.0f;
    }

    __shared__ CLFlt s_like[80*17];
    CLFlt *m_like = &s_like[17*tid];

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1, scaler = 0.0;
        int xi, ai, gamma;
        xi = xState[char_idx];
        ai = aState[char_idx];
        int clx_offset = bid_x*80*16 + 16 * tid;
        CLFlt *g_clx = &clx[clx_offset];
        CLFlt *g_clP = &clP[clx_offset];
        CLFlt *m_tiPX = &s_tiPX[0];
        MrBFlt like = 0.0;

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = g_clx[0]; c0 = g_clx[1]; g0 = g_clx[2]; t0 = g_clx[3];

            a1 = m_tiPX[0]*a0 + m_tiPX[1]*c0 + m_tiPX[2]*g0 + m_tiPX[3]*t0;
            c1 = m_tiPX[4]*a0 + m_tiPX[5]*c0 + m_tiPX[6]*g0 + m_tiPX[7]*t0;
            g1 = m_tiPX[8]*a0 + m_tiPX[9]*c0 + m_tiPX[10]*g0 + m_tiPX[11]*t0;
            t1 = m_tiPX[12]*a0 + m_tiPX[13]*c0 + m_tiPX[14]*g0 + m_tiPX[15]*t0;

            a1 *= s_preLikeX[xi] * s_preLikeA[ai];
            c1 *= s_preLikeX[xi+1] * s_preLikeA[ai+1];
            g1 *= s_preLikeX[xi+2] * s_preLikeA[ai+2];
            t1 *= s_preLikeX[xi+3] * s_preLikeA[ai+3];

            scaler = MAX(scaler, a1);
            scaler = MAX(scaler, c1);
            scaler = MAX(scaler, g1);
            scaler = MAX(scaler, t1);

            m_like[0] = a1;
            m_like[1] = c1;
            m_like[2] = g1;
            m_like[3] = t1;

            g_clx += 4;
            m_tiPX += 16;
            xi += 20;
            ai += 20;
            m_like += 4;
        }

        m_like = &s_like[tid*17];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            m_like[0] /= scaler;
            m_like[1] /= scaler;
            m_like[2] /= scaler;
            m_like[3] /= scaler;
            m_like += 4;
        }

        scaler = (CLFlt)log(scaler);
        scPNew[char_idx] = scaler;
        CLFlt dif = lnScaler[char_idx];
        dif += scaler;

        m_like = &s_like[tid*17];
        a1=0.0; c1=0.0; g1=0.0; t1=0.0;
        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_like[0]; a1 += a0; g_clP[0] = a0;
            c0 = m_like[1]; c1 += c0; g_clP[1] = c0;
            g0 = m_like[2]; g1 += g0; g_clP[2] = g0;
            t0 = m_like[3]; t1 += t0; g_clP[3] = t0;
            m_like += 4;
            g_clP += 4;
        }

        like = a1 * bs_A + c1 * bs_C + g1 * bs_G + t1 * bs_T;

        MrBFlt *g_clInvar = &clInvar[bid_x*80*4 + 4*tid];
        likelihood_NUC4_hasPInvar_YES_e(like, g_clInvar, dif, nSitesOfPat, lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, char_idx);

        lnScaler[char_idx] = dif;
    }
}

__global__ void gpu_root_12_3_e(CLFlt *clP, CLFlt *clx, CLFlt *tiPX, int *xState, int *aState, CLFlt *ti_preLikeX, CLFlt *tiPA, CLFlt *lnScaler, CLFlt *scPOld, CLFlt *scPNew, MrBFlt *clInvar, CLFlt *nSitesOfPat, MrBFlt *lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    int bid_x = blockIdx.x;

    int tid = blockDim.x * blockDim.y * tid_z + blockDim.x * tid_y + tid_x;
    int char_idx = 80 * bid_x + tid;

    __shared__ CLFlt s_tiPX[64];

    if(tid < 64)
    {
        s_tiPX[tid] = tiPX[tid];
    }

    __shared__ CLFlt s_preLikeX[80];
    __shared__ CLFlt s_preLikeA[80];

    if(tid_y < 4)
    {
        int tip_idx = 16 * tid_z + tid_y + 4 * tid_x;
        s_preLikeA[tid] = tiPA[tip_idx];
        s_preLikeX[tid] = ti_preLikeX[tip_idx];
    }
    else
    {
        s_preLikeA[tid] = 1.0f;
        s_preLikeX[tid] = 1.0f;
    }

    __shared__ CLFlt s_like[80*17];
    CLFlt *m_like = &s_like[17*tid];

    __syncthreads();

    if(char_idx < modelnumChars)
    {
        CLFlt a0, c0, g0, t0;
        CLFlt a1, c1, g1, t1, scaler = 0.0;
        int xi, ai, gamma;
        xi = xState[char_idx];
        ai = aState[char_idx];
        int clx_offset = bid_x*80*16 + 16*tid;
        CLFlt *g_clP = &clP[clx_offset];
        CLFlt *g_clx = &clx[clx_offset];
        CLFlt *m_tiPX = &s_tiPX[0];

        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = g_clx[0]; c0 = g_clx[1]; g0 = g_clx[2]; t0 = g_clx[3];

            a1 = m_tiPX[0]*a0 + m_tiPX[1]*c0 + m_tiPX[2]*g0 + m_tiPX[3]*t0;
            c1 = m_tiPX[4]*a0 + m_tiPX[5]*c0 + m_tiPX[6]*g0 + m_tiPX[7]*t0;
            g1 = m_tiPX[8]*a0 + m_tiPX[9]*c0 + m_tiPX[10]*g0 + m_tiPX[11]*t0;
            t1 = m_tiPX[12]*a0 + m_tiPX[13]*c0 + m_tiPX[14]*g0 + m_tiPX[15]*t0;

            a1 *= s_preLikeX[xi] * s_preLikeA[ai];
            c1 *= s_preLikeX[xi+1] * s_preLikeA[ai+1];
            g1 *= s_preLikeX[xi+2] * s_preLikeA[ai+2];
            t1 *= s_preLikeX[xi+3] * s_preLikeA[ai+3];

            scaler = MAX(scaler, a1);
            scaler = MAX(scaler, c1);
            scaler = MAX(scaler, g1);
            scaler = MAX(scaler, t1);

            m_like[0] = a1;
            m_like[1] = c1;
            m_like[2] = g1;
            m_like[3] = t1;

            g_clx += 4;
            m_tiPX += 16;
            xi += 20;
            ai += 20;
            m_like += 4;
        }

        m_like = &s_like[tid*17];
        for(gamma = 0; gamma < 4; gamma ++)
        {
            m_like[0] /= scaler;
            m_like[1] /= scaler;
            m_like[2] /= scaler;
            m_like[3] /= scaler;
            m_like += 4;
        }

        scaler = (CLFlt)log(scaler);
        scPNew[char_idx] = scaler;
        CLFlt dif = lnScaler[char_idx];
        dif -= scPOld[char_idx];
        dif += scaler;

        m_like = &s_like[tid*17];
        a1=0.0; c1=0.0; g1=0.0; t1=0.0;
        for(gamma = 0; gamma < 4; gamma ++)
        {
            a0 = m_like[0]; a1 += a0; g_clP[0] = a0;
            c0 = m_like[1]; c1 += c0; g_clP[1] = c0;
            g0 = m_like[2]; g1 += g0; g_clP[2] = g0;
            t0 = m_like[3]; t1 += t0; g_clP[3] = t0;
            m_like += 4;
            g_clP += 4;
        }

        MrBFlt like = a1 * bs_A + c1 * bs_C + g1 * bs_G + t1 * bs_T;
        MrBFlt *g_clInvar = &clInvar[bid_x*80*4 + 4*tid];
        likelihood_NUC4_hasPInvar_YES_e(like, g_clInvar, dif, nSitesOfPat, lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, char_idx);
        lnScaler[char_idx] = dif;
    }
}

static void root_12_0(int offset_clP, int offset_clx, int offset_px, int offset_xState, int offset_aState, int offset_px_preLike, int offset_pA, int offset_lnScaler, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars, int modelnumGammaCats, int chain)
{
    int nthread = 80;
    int gx = modelnumChars / nthread;
    int residue = modelnumChars % nthread;

    if(residue != 0)
        gx += 1;

    dim3	dimGrid(gx, 1, 1);
    dim3	dimBlock(4, 5, 4);

    gpu_root_12_0_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clx, devtiProbSpace+offset_px, devtermState+offset_xState, devtermState+offset_aState, devtiProbSpace + offset_px_preLike, devtiProbSpace+offset_pA, devtreeScalerSpace+offset_lnScaler, devinvCondLikes+offset_clInvar, devnumSitesOfPat+offset_nSitesOfPat, devlnL+offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars);

    cutilCheckMsg("gpu_root_12_0_e failed\n");
}

static void root_12_1(int offset_clP, int offset_clx, int offset_px, int offset_xState, int offset_aState, int offset_px_preLike, int offset_pA, int offset_lnScaler, int offset_scPOld, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars, int modelnumGammaCats, int chain)
{
    int nThread = 80;
    int gx = modelnumChars / nThread;
    int residue = modelnumChars % nThread;
    if (residue != 0)
        gx += 1;

    dim3	dimGrid(gx, 1, 1);
    dim3	dimBlock(4, 5, 4);

    gpu_root_12_1_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clx, devtiProbSpace+offset_px, devtermState+offset_xState, devtermState+offset_aState, devtiProbSpace + offset_px_preLike, devtiProbSpace+offset_pA, devtreeScalerSpace+offset_lnScaler, devnodeScalerSpace+offset_scPOld, devinvCondLikes+offset_clInvar, devnumSitesOfPat+offset_nSitesOfPat, devlnL+offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars);

    cutilCheckMsg("gpu_root_12_1_e failed\n");
}

static void root_12_2(int offset_clP, int offset_clx, int offset_px, int offset_xState, int offset_aState, int offset_px_preLike, int offset_pA, int offset_lnScaler, int offset_scPNew, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars, int modelnumGammaCats, int chain)
{
    int nThread = 80;
    int gx = modelnumChars / nThread;
    int residue = modelnumChars % nThread;
    if (residue != 0)
        gx += 1;

    dim3	dimGrid(gx, 1, 1);
    dim3	dimBlock(4, 5, 4);

    gpu_root_12_2_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clx, devtiProbSpace+offset_px, devtermState+offset_xState, devtermState+offset_aState, devtiProbSpace + offset_px_preLike, devtiProbSpace+offset_pA, devtreeScalerSpace+offset_lnScaler, devnodeScalerSpace+offset_scPNew, devinvCondLikes+offset_clInvar, devnumSitesOfPat+offset_nSitesOfPat, devlnL+offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars);

    cutilCheckMsg("gpu_root_12_2_e failed\n");
}

static void root_12_3(int offset_clP, int offset_clx, int offset_px, int offset_xState, int offset_aState, int offset_px_preLike, int offset_pA, int offset_lnScaler, int offset_scPOld, int offset_scPNew, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A, MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int modelnumChars, int modelnumGammaCats, int chain)
{
    int nThread = 80;
    int gx = modelnumChars / nThread;
    int residue = modelnumChars % nThread;
    if (residue != 0)
        gx += 1;

    dim3	dimGrid(gx, 1, 1);
    dim3	dimBlock(4, 5, 4);

    gpu_root_12_3_e<<<dimGrid, dimBlock, 0, stream[chain]>>>(devchainCondLikes+offset_clP, devchainCondLikes+offset_clx, devtiProbSpace+offset_px, devtermState+offset_xState, devtermState+offset_aState, devtiProbSpace + offset_px_preLike, devtiProbSpace+offset_pA, devtreeScalerSpace+offset_lnScaler,devnodeScalerSpace+offset_scPOld, devnodeScalerSpace+offset_scPNew, devinvCondLikes+offset_clInvar, devnumSitesOfPat+offset_nSitesOfPat, devlnL+offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars);

    cutilCheckMsg("gpu_root_12_3_e failed\n");
}

extern "C" void cuda_root_12(int offset_clP, int offset_clx, int offset_px, int offset_xState, int offset_aState, int offset_px_preLike, int offset_pA, int offset_scPOld, int offset_scPNew, int offset_lnScaler, int scaler, int offset_clInvar, int offset_nSitesOfPat, int offset_lnL, MrBFlt bs_A,MrBFlt bs_C, MrBFlt bs_G, MrBFlt bs_T, MrBFlt freq, MrBFlt pInvar, int hasPInvar, int modelnumChars, int modelnumGammaCats, int chain)
{
    if(hasPInvar == NO)
    {
        fprintf(stderr, "tgMC3 was designed for data having invariable sites\n");
        exit(1);
    }

    switch(scaler)
    {
        case 0:

            root_12_0(offset_clP, offset_clx, offset_px, offset_xState, offset_aState, offset_px_preLike, offset_pA, offset_lnScaler, offset_clInvar, offset_nSitesOfPat, offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars, modelnumGammaCats, chain);

            break;

        case 1:
            root_12_1(offset_clP, offset_clx, offset_px, offset_xState, offset_aState, offset_px_preLike, offset_pA, offset_lnScaler, offset_scPOld, offset_clInvar, offset_nSitesOfPat, offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars, modelnumGammaCats, chain);
            break;

        case 2:

            root_12_2(offset_clP, offset_clx, offset_px, offset_xState, offset_aState, offset_px_preLike, offset_pA, offset_lnScaler, offset_scPNew, offset_clInvar, offset_nSitesOfPat, offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars, modelnumGammaCats, chain);

            break;

        case 3:

            root_12_3(offset_clP, offset_clx, offset_px, offset_xState, offset_aState, offset_px_preLike, offset_pA, offset_lnScaler, offset_scPOld, offset_scPNew, offset_clInvar, offset_nSitesOfPat, offset_lnL, bs_A, bs_C, bs_G, bs_T, freq, pInvar, modelnumChars, modelnumGammaCats, chain);
    }
}

extern "C" void cudaFreeAll (void)
{
    int	i;

    for (i=0; i<numLocalChains; i++)
    {
        cutilSafeCall(cudaStreamDestroy(stream[i]));
        cutilCheckMsg("cudaStreamDestroy failed");
    }

    if (augmentData == YES)
    {
        cutilSafeCall(cudaMemcpy((void *)chainCondLikes, (const void *)devchainCondLikes, numLocalChains*globaloneMatSize*sizeof(CLFlt), cudaMemcpyDeviceToHost));
        cutilCheckMsg("cudaMemcpy chainCondLikes failed");
    }
    else
    {
        for (i=0; i<numLocalChains; i++)
        {
            cutilSafeCall(cudaMemcpy((void *)(chainCondLikes+i*(globaloneMatSize-numLocalTaxa*condLikeRowSize)), (const void *)(devchainCondLikes+i*globaloneMatSize+numLocalTaxa*condLikeRowSize), (globaloneMatSize-numLocalTaxa*condLikeRowSize)*sizeof(CLFlt), cudaMemcpyDeviceToHost));
            cutilCheckMsg("cudaMemcpy chainCondLikeschainCondLikes failed");
        }
    }
    cutilSafeCall(cudaFree((void *)devchainCondLikes));
    cutilCheckMsg("cudaFreedevchainCondLikes failed");

    cutilSafeCall(cudaFree((void *)devinvCondLikes));
    cutilCheckMsg("cudaFreedevinvCondLikes failed");

    cutilSafeCall(cudaFree((void *)devlnL));
    cutilCheckMsg("cudaFree devlnL failed");

    cutilSafeCall(cudaMemcpy((void *)nodeScalerSpace, (const void *)devnodeScalerSpace, 2*numLocalChains*globalnScalerNodes*numCompressedChars*sizeof(CLFlt), cudaMemcpyDeviceToHost));
    cutilCheckMsg("cudaMemcpy nodeScalerSpace failed");
    cutilSafeCall(cudaFree((void *)devnodeScalerSpace));
    cutilCheckMsg("cudaFreedevnodeScalerSpace failed");

    cutilSafeCall(cudaFree((void *)devnumSitesOfPat));
    cutilCheckMsg("cudaFreedevnumSitesOfPat failed");

    cutilSafeCall(cudaFree((void *)devtermState));
    cutilCheckMsg("cudaFreedevtermState failed");

    cutilSafeCall(cudaFree((void *)devtiProbSpace));
    cutilCheckMsg("cudaFreedevtiProbSpace failed");

    cutilSafeCall(cudaMemcpy((void *)treeScalerSpace, (const void *)devtreeScalerSpace, 2*numLocalChains*numCompressedChars*sizeof(CLFlt), cudaMemcpyDeviceToHost));
    cutilCheckMsg("cudaMemcpy treeScalerSpace failed");
    cutilSafeCall(cudaFree((void *)devtreeScalerSpace));
    cutilCheckMsg("cudaFreedevtreeScalerSpace failed");

    cutilSafeCall(cudaFreeHost((void *)globallnL));
    cutilCheckMsg("cudaFreeHostgloballnL failed");
}

extern "C" void cudaFreeHosttiProbSpace (void)
{
    cutilSafeCall(cudaFreeHost((void *)tiProbSpace));
    cutilCheckMsg("cudaFreeHosttiProbSpace failed");
}

extern "C" void cudaHostAllocWriteCombinedtiProbSpace (size_t size)
{
    cutilSafeCall(cudaHostAlloc((void **)&tiProbSpace, size, cudaHostAllocWriteCombined));
    cutilCheckMsg("cudaHostAllocWriteCombinedtiProbSpace failed");
}

extern "C" void cudaMallocAll (void)
{
    int i;
    cutilSafeCall(cudaMalloc((void **)&devchainCondLikes, numLocalChains*globaloneMatSize*sizeof(CLFlt)));
    cutilCheckMsg("cudaMalloc devchainCondLikes failed");
    if (augmentData == YES)
    {
        cutilSafeCall(cudaMemcpy((void *)devchainCondLikes, (const void *)chainCondLikes, numLocalChains*globaloneMatSize*sizeof(CLFlt), cudaMemcpyHostToDevice));
        cutilCheckMsg("cudaMemcpy devchainCondLikes failed");
    }
    else
    {
        for (i=0; i<numLocalChains; i++)
        {
            cutilSafeCall(cudaMemcpy((void *)(devchainCondLikes+i*globaloneMatSize), (const void *)termCondLikes, numLocalTaxa*condLikeRowSize*sizeof(CLFlt), cudaMemcpyHostToDevice));
            cutilCheckMsg("cudaMemcpy devchainCondLikestermCondLikes failed");

            cutilSafeCall(cudaMemcpy((void *)(devchainCondLikes+i*globaloneMatSize+numLocalTaxa*condLikeRowSize), (const void *)(chainCondLikes+i*(globaloneMatSize-numLocalTaxa*condLikeRowSize)), (globaloneMatSize-numLocalTaxa*condLikeRowSize)*sizeof(CLFlt), cudaMemcpyHostToDevice));
            cutilCheckMsg("cudaMemcpy devchainCondLikeschainCondLikes failed");
        }
    }

    cutilSafeCall(cudaMalloc((void **)&devinvCondLikes, globalinvCondLikeSize*sizeof(MrBFlt)));
    cutilCheckMsg("cudaMallocdevinvCondLikes failed");
    cutilSafeCall(cudaMemcpy((void *)devinvCondLikes, (const void *)invCondLikes, globalinvCondLikeSize*sizeof(MrBFlt), cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpydevinvCondLikes failed");

    modeldevlnLIdx = (int *) calloc (numCurrentDivisions, sizeof(int));
    sizeofdevlnL = 0;
    for (i=0; i<numCurrentDivisions; i++)
    {
        modeldevlnLIdx[i] = sizeofdevlnL;
        sizeofdevlnL += modelSettings[i].numChars;
    }
    cutilSafeCall(cudaMalloc((void **)&devlnL, numLocalChains * sizeofdevlnL * sizeof(MrBFlt)));
    cutilCheckMsg("cudaMalloc devlnL failed");

    cutilSafeCall(cudaMalloc((void **)&devnodeScalerSpace, 2*numLocalChains*globalnScalerNodes*numCompressedChars*sizeof(CLFlt)));
    cutilCheckMsg("cudaMallocdevnodeScalerSpace failed");
    cutilSafeCall(cudaMemcpy((void *)devnodeScalerSpace, (const void *)nodeScalerSpace, 2*numLocalChains*globalnScalerNodes*numCompressedChars*sizeof(CLFlt), cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpydevnodeScalerSpace failed");

    cutilSafeCall(cudaMalloc((void **)&devnumSitesOfPat, numCompressedChars*chainParams.numChains*sizeof(MrBFlt)));
    cutilCheckMsg("cudaMallocdevnumSitesOfPat failed");
    cutilSafeCall(cudaMemcpy((void *)devnumSitesOfPat, (const void *)numSitesOfPat, numCompressedChars*chainParams.numChains*sizeof(MrBFlt), cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpydevnumSitesOfPat failed");

    cutilSafeCall(cudaMalloc((void **)&devtermState, numLocalTaxa*numCompressedChars*sizeof(int)));
    cutilCheckMsg("cudaMallocdevtermState failed");
    cutilSafeCall(cudaMemcpy((void *)devtermState, (const void *)termState, numLocalTaxa*numCompressedChars*sizeof(int), cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpydevtermState failed");

    cutilSafeCall(cudaMalloc((void **)&devtiProbSpace, 2*numLocalChains*globalnNodes*tiProbRowSize *sizeof(CLFlt)));
    cutilCheckMsg("cudaMallocdevtiProbSpace failed");

    cutilSafeCall(cudaMalloc((void **)&devtreeScalerSpace, 2*numLocalChains*numCompressedChars*sizeof(CLFlt)));
    cutilCheckMsg("cudaMallocdevtreeScalerSpace failed");
    cutilSafeCall(cudaMemcpy((void *)devtreeScalerSpace, (const void *)treeScalerSpace, 2*numLocalChains*numCompressedChars*sizeof(CLFlt), cudaMemcpyHostToDevice));
    cutilCheckMsg("cudaMemcpydevtreeScalerSpace failed");

    cutilSafeCall(cudaMallocHost((void **)&globallnL, numLocalChains * sizeofdevlnL * sizeof(MrBFlt)));
    cutilCheckMsg("cudaMallocHost globallnL failed");

    stream = (cudaStream_t *) calloc (numLocalChains, sizeof(cudaStream_t));

    for (i=0; i<numLocalChains; i++)
    {
        cutilSafeCall(cudaStreamCreate(&stream[i]));
        cutilCheckMsg("cudaStreamCreate failed");
    }
}

extern "C" void cudaMemcpyAsyncgloballnL (int chain)
{
    cutilSafeCall(cudaMemcpyAsync((void *)(globallnL + chain * sizeofdevlnL), (const void *)(devlnL + chain * sizeofdevlnL), sizeofdevlnL * sizeof(MrBFlt), cudaMemcpyDeviceToHost, stream[chain]));
    cutilCheckMsg("cudaMemcpyAsync globallnL failed");
}

extern "C" void cudaMemcpyAsynctiProbSpace (int chain)
{
    cutilSafeCall(cudaMemcpyAsync((void *)(devtiProbSpace+chain*2*globalnNodes*tiProbRowSize), (const void *)(tiProbSpace+chain*2*globalnNodes*tiProbRowSize), 2*globalnNodes*tiProbRowSize*sizeof(CLFlt), cudaMemcpyHostToDevice, stream[chain]));
    cutilCheckMsg("cudaMemcpyAsyncdevtiProbSpace failed");
}

extern "C" void cudaMemcpyAsynctreeScalerSpace (int offset_fromState, int offset_toState, int chain)
{
    cutilSafeCall(cudaMemcpyAsync((void *)(devtreeScalerSpace + offset_toState), (const void *)(devtreeScalerSpace + offset_fromState), numCompressedChars*sizeof(CLFlt), cudaMemcpyDeviceToDevice, stream[chain]));
    cutilCheckMsg("cudaMemcpyAsynctreeScalerSpace failed");
}

extern "C" void cudaStreamSync (int chain)
{
    cutilSafeCall(cudaStreamSynchronize(stream[chain]));
    cutilCheckMsg("cudaStreamSynchronize failed");
    if (cudaStreamQuery(stream[chain]) != cudaSuccess)
    {
        fprintf(stderr, "All operations in stream %d have not completed.\n", chain);
    }
}

extern "C" int InitCudaEnvironment (void)
{
    int dev, devCount;

    cutilSafeCall(cudaGetDeviceCount(&devCount));

    devCount = min(devCount, MAX_GPU_COUNT);

    if (devCount == 0)
    {
        printf("no CUDA-capable devices found.\n");
        return ERROR;
    }

    for (dev = 0; dev < devCount; ++dev)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0)
        {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                printf("There is no device supporting CUDA.\n");
            else if (devCount == 1)
                printf("There is 1 device supporting CUDA\n");
            else
                printf("There are %d devices supporting CUDA\n", devCount);
        }
    }

    cutilSafeCall(cudaSetDevice(0));
    cutilCheckMsg("cudaSetDevice failed");
    return 0;
}
