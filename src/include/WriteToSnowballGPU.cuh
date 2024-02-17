#include <stdio.h>

__global__ void WriteToSnowballGPU(
	float *px, float *py, float *pz,
	int *pq,
	float *px_snow, float *py_snow, float *pz_snow,
	int *pq_snow,
	int *pnum,
	int nB, int snowball, int np_branches)
{

	int b = threadIdx.x + blockDim.x * blockIdx.x;		// Branch number

	//	int  writeFlag = 1;
	while (b < nB){
		for (int n = 0; n < pnum[b]; n++){
			int na = b + n*nB;
			int na_snow = b + n*nB + snowball*np_branches;

			px_snow[na_snow] = px[na];
			py_snow[na_snow] = py[na];
			pz_snow[na_snow] = pz[na];
			pq_snow[na_snow] = pq[na];
			/*if (b == 17){
				printf("writing px[%i] to px_snow[%i] = %1.10f, pq = %i, pqsnow = %i\n", na, na_snow, px_snow[na_snow], pq[na], pq_snow[na_snow]);
			}*/
		}
		b += blockDim.x*gridDim.x;
	}
}
