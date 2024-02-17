#include <math.h>

#define h			6.62607E-34
#define	c			299792458
#define pi			3.141592653589793
#define eps0		8.85418782E-12				// Permittivity of free space
#define mu0			1.2566370614E-6				// Vacuum permeability; N-s2/C2

__global__ void ThrustCalculationGPU(
	float *px, float *py, float *pz,
	float *vx, float *vy, float *vz,
	int *pq,
	float *E_realx,
	float *E_imagx,
	float *B_realx,
	float *B_imagx,
	float *E_realy,
	float *E_imagy,
	float *B_realy,
	float *B_imagy,
	float *E_realz,
	float *E_imagz,
	float *B_realz,
	float *B_imagz,
	int nB,
	float freq, float cosphase, float sinphase,
	int *pnum,
	int ts,
	float dt
	){

	float Ex, Ey, Ez, Bx, By, Bz;

	int b = threadIdx.x + blockDim.x * blockIdx.x;		// Branch number
	//printf("transfer launched\n");
	while (b < nB){
		Ex = E_realx[b] * cosphase + E_imagx[b] * sinphase;
		Ey = E_realy[b] * cosphase + E_imagy[b] * sinphase;
		Ez = E_realz[b] * cosphase + E_imagz[b] * sinphase;
		Ex = E_realx[b] * cosphase + E_imagx[b] * sinphase;
		Ey = E_realy[b] * cosphase + E_imagy[b] * sinphase;
		Ez = E_realz[b] * cosphase + E_imagz[b] * sinphase;

		Bx = B_realx[b] * cosphase + B_imagx[b] * sinphase;
		By = B_realy[b] * cosphase + B_imagy[b] * sinphase;
		Bz = B_realz[b] * cosphase + B_imagz[b] * sinphase;
		Bx = B_realx[b] * cosphase + B_imagx[b] * sinphase;
		By = B_realy[b] * cosphase + B_imagy[b] * sinphase;
		Bz = B_realz[b] * cosphase + B_imagz[b] * sinphase;
		float rhoE = (1/(2*mu0))*(Bx*Bx + By*By + Bz*Bz);
		float rhoB = 0.5*eps0*(Ex*Ex + Ey*Ey + Ez*Ez);
		float lambda = h*freq / (rhoE + rhoB);
		rhoVac[b] = (h*c*pi*pi) / (2 * pi * 3 * c*c * 240 * pow(lambda, 1.3333333));

		float cvx_total = 0;
		float cvy_total = 0;
		float cvz_total = 0;

		for (int i = 0; i < pnum[b]; i++){
			int na = b + pnum[b] * nB;
			cvx_total += fabs(vx[na]);
			cvy_total += fabs(vy[na]);
			cvz_total += fabs(vz[na]);
		}
		if (pnum[b]>0){
			vx_avg[b] = cvx_total / pnum[b];
			vy_avg[b] = cvy_total / pnum[b];
			vz_avg[b] = cvz_total / pnum[b];
		}
		b += blockDim.x*gridDim.x;
	}
}
