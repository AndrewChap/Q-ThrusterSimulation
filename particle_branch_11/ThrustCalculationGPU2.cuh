#include <math.h>

__global__ void ThrustCalculationGPU2(
	float *pressureX, float *pressureY, float *pressureZ,
	float *accelX1, float *accelY1, float *accelZ1,
	float *accelX2, float *accelY2, float *accelZ2,
	float *forceX1, float *forceY1, float *forceZ1,
	float *forceX2, float *forceY2, float *forceZ2,
	float *forceX3, float *forceY3, float *forceZ3,
	float *rhovac,
	float dxB, float dyB, float dzB,
	int *Bxm, int *Bxp, int *Bym, int *Byp, int *Bzm, int *Bzp,
	int nB, int substep, int subflag
	){

		
	int b = threadIdx.x + blockDim.x * blockIdx.x;		// Branch number
	float Pxm, Pxp, Pym, Pyp, Pzm, Pzp;
	while (b < nB){
		if (subflag == 1){
			forceX1[b] = 0;
			forceY1[b] = 0;
			forceZ1[b] = 0;
			forceX2[b] = 0;
			forceY2[b] = 0;
			forceZ2[b] = 0;
			forceX3[b] = 0;
			forceY3[b] = 0;
			forceZ3[b] = 0;
		}
		/*Pxm = (Bxm != -1)*PressureX[Bxm*(Bxm != -1)];
		Pxp = (Bxp != -1)*PressureX[Bxp*(Bxp != -1)];
		Pym = (Bym != -1)*PressureY[Bym*(Bym != -1)];
		Pyp = (Byp != -1)*PressureY[Byp*(Byp != -1)];
		Pzm = (Bzm != -1)*PressureZ[Bzm*(Bzm != -1)];
		Pzp = (Bzp != -1)*PressureZ[Bzp*(Bzp != -1)];*/
		
		if (Bxm[b] != -1) Pxm = pressureX[Bxm[b]]; else Pxm = 0;
		if (Bxp[b] != -1) Pxp = pressureX[Bxp[b]]; else Pxp = 0;
		if (Bym[b] != -1) Pym = pressureY[Bym[b]]; else Pym = 0;
		if (Byp[b] != -1) Pyp = pressureY[Byp[b]]; else Pyp = 0;
		if (Bzm[b] != -1) Pzm = pressureZ[Bzm[b]]; else Pzm = 0;
		if (Bzp[b] != -1) Pzp = pressureZ[Bzp[b]]; else Pzp = 0;

		forceX1[b] = (rhovac[b] * accelX1[b] + forceX1[b] * substep) / (substep + 1);
		forceY1[b] = (rhovac[b] * accelY1[b] + forceY1[b] * substep) / (substep + 1);
		forceZ1[b] = (rhovac[b] * accelZ1[b] + forceZ1[b] * substep) / (substep + 1);

		forceX2[b] = (rhovac[b] * accelX2[b] + forceX2[b] * substep) / (substep + 1);
		forceY2[b] = (rhovac[b] * accelY2[b] + forceY2[b] * substep) / (substep + 1);
		forceZ2[b] = (rhovac[b] * accelZ2[b] + forceZ2[b] * substep) / (substep + 1);

		forceX3[b] = ((-0.5*(Pxp - Pxm) / dxB) + forceX3[b] * substep) / (substep + 1);	// running average of force density
		forceY3[b] = ((-0.5*(Pyp - Pym) / dyB) + forceY3[b] * substep) / (substep + 1);	// running average of force density*/
		forceZ3[b] = ((-0.5*(Pzp - Pzm) / dzB) + forceZ3[b] * substep) / (substep + 1);	// running average of force density
		/*forceX[b] = -0.5*(Pxp - Pxm) / dxB;
		forceY[b] = -0.5*(Pyp - Pym) / dyB;
		forceZ[b] = -0.5*(Pzp - Pzm) / dzB;*/
		//forceZ[b] += -0.5*(Pzp - Pzm) / dzB;
		//forceZ[b] += Pzp;
		b += blockDim.x*gridDim.x;
	}
}