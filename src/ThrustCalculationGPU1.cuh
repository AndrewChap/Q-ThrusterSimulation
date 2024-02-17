#include <math.h>

#define planck			6.62607E-34
#define	speedoflight	299792458
#define pi				3.141592653589793
#define eps0			8.85418782E-12				// Permittivity of free space
#define mu0				1.2566370614E-6				// Vacuum permeability; N-s2/C2

__global__ void ThrustCalculationGPU1(
	float *px, float *py, float *pz,
	float *vx, float *vy, float *vz,
	float *ax, float *ay, float *az,
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
	float *pressureX, float *pressureY, float *pressureZ,
	float *velX_p, float *velY_p, float *velZ_p,
	float *velX_e, float *velY_e, float *velZ_e,
	float *accelX1, float *accelY1, float *accelZ1,
	float *accelX2, float *accelY2, float *accelZ2,
	int nB,
	float freq, float cosphase, float sinphase,
	int *pnum,
	int ts,
	float dt, float *vz_print, float *rhovac
	){

	float Ex, Ey, Ez, Bx, By, Bz;
	//printf("insideTCkernel1\b");
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
		float rhoE = (1 / (2 * mu0))*(Bx*Bx + By*By + Bz*Bz);
		float rhoB = 0.5*eps0*(Ex*Ex + Ey*Ey + Ez*Ez);
		float lambda = planck*freq / (rhoE + rhoB);
		float rhoVac = (planck*speedoflight*pi*pi) / (2 * pi * 3 * speedoflight*speedoflight * 240 * powf(lambda, 1.3333333f));

		float vx_avg_abs = 0;
		float vy_avg_abs = 0;
		float vz_avg_abs = 0;
		velX_p[b] = 0;
		velY_p[b] = 0;
		velZ_p[b] = 0;
		velX_e[b] = 0;
		velY_e[b] = 0;
		velZ_e[b] = 0;
		float ax_avg = 0;
		float ay_avg = 0;
		float az_avg = 0;
		vz_print[b] = 0;
		int num_pos = 0;
		int num_neg = 0;
		int num = 0;
		float ax_avg_pos = 0;
		float ay_avg_pos = 0;
		float az_avg_pos = 0;
		float ax_avg_neg = 0;
		float ay_avg_neg = 0;
		float az_avg_neg = 0;
		accelX1[b] = 0;
		accelY1[b] = 0;
		accelZ1[b] = 0;
		accelX2[b] = 0;
		accelY2[b] = 0;
		accelZ2[b] = 0;
		for (int n = 0; n < pnum[b]; n++){
			int na = b + n * nB;
			vx_avg_abs += fabsf(vx[na]);
			vy_avg_abs += fabsf(vy[na]);
			vz_avg_abs += fabsf(vz[na]);
			if (pq[na]>0){
				ax_avg_pos = (ax[na] + ax_avg_pos*num_pos) / (num_pos + 1);
				ay_avg_pos = (ay[na] + ay_avg_pos*num_pos) / (num_pos + 1);
				az_avg_pos = (az[na] + az_avg_pos*num_pos) / (num_pos + 1);
				velX_p[b] += (vx[na] + velX_p[b] * num_pos) / (num_pos + 1);
				velY_p[b] += (vy[na] + velY_p[b] * num_pos) / (num_pos + 1);
				velZ_p[b] += (vz[na] + velZ_p[b] * num_pos) / (num_pos + 1);
				num_pos++;
			}
			if (pq[na]<0){
				ax_avg_neg = (ax[na] + ax_avg_neg*num_neg) / (num_neg + 1);
				ay_avg_neg = (ay[na] + ay_avg_neg*num_neg) / (num_neg + 1);
				az_avg_neg = (az[na] + az_avg_neg*num_neg) / (num_neg + 1);
				velX_e[b] += (vx[na] + velX_p[b] * num_neg) / (num_neg + 1);
				velY_e[b] += (vy[na] + velY_p[b] * num_neg) / (num_neg + 1);
				velZ_e[b] += (vz[na] + velZ_p[b] * num_neg) / (num_neg + 1);
				num_neg++;
			}
			ax_avg = (ax[na] + ax_avg*num) / (num + 1);
			ay_avg = (ay[na] + ay_avg*num) / (num + 1);
			az_avg = (az[na] + az_avg*num) / (num + 1);
			num++;
			//ax_avg += ax[na];
			//ay_avg += ay[na];
			//az_avg += az[na];
		}
		if (pnum[b]>0){
			vx_avg_abs /= pnum[b];
			vy_avg_abs /= pnum[b];
			vz_avg_abs /= pnum[b];
			//ax_avg /= pnum[b];
			//ay_avg /= pnum[b];
			//az_avg /= pnum[b];
			//printf("vz_avg[%i] = %e\n", b, vz_avg);
			//vz_print[b] += vz_avg;
		}
		if (num_pos > 0 && num_neg > 0){
			accelX1[b] = ax_avg;
			accelY1[b] = ay_avg;
			accelZ1[b] = az_avg;
			accelX2[b] = 0.5*(ax_avg_pos + ax_avg_neg);
			accelY2[b] = 0.5*(ay_avg_pos + ay_avg_neg);
			accelZ2[b] = 0.5*(az_avg_pos + az_avg_neg);
		}

		pressureX[b] = rhoVac*vx_avg_abs*vx_avg_abs;
		pressureY[b] = rhoVac*vy_avg_abs*vy_avg_abs;
		pressureZ[b] = rhoVac*vz_avg_abs*vz_avg_abs;
		rhovac[b] = rhoVac;

		b += blockDim.x*gridDim.x;
	}
}