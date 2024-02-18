#include <math.h>
#include <stdio.h>

#define pi			3.141592653589793			// pi
#define eps0		8.85418782E-12				// Permittivity of free space
#define mu0			1.2566370614E-6				// Vacuum permeability; N-s2/C2
#define mu0o4pi		1E-7						// mu0/(4*pi)
#define Oo4piEps0	8.9875517873681764e9		// 1/(4*pi*eps0)
#define QE			1.602176565E-19				// elementary charge (C)
#define ME			9.10938215E-31				// electron rest mass (kg)

__global__ void ParticleMoverGPU(
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
	float *pxBuffer, float *pyBuffer, float *pzBuffer,
	float *vxBuffer, float *vyBuffer, float *vzBuffer,
	float *axBuffer, float *ayBuffer, float *azBuffer,
	int *pqBuffer, int *IDBuffer,
	int *DestinationsBuffer, int *TransferIndex, int *KillIndex,
	int *TransferFlag, int *KillFlag,
	int *NumTransfer, int *NumKill,
	int *Bxm, int *Bxp, int *Bym, int *Byp, int *Bzm, int *Bzp,
	float *Bxmin, float *Bxmax, float *Bymin, float *Bymax, float *Bzmin, float *Bzmax,
	int *pnum, int nB, float dt, float qom,
	float LX, float LY,
	float cosphase, float sinphase,
	float L, float R1, float R2, float inv_thresh,
	float qRPM, int INTERACTION, int ts, int *ID)
{
	//printf("particle mover launched\n");

	/*extern __shared__ int psx[];
	extern __shared__ int psy[];
	extern __shared__ int psz[];
	extern __shared__ int vsx[];
	extern __shared__ int vsy[];
	extern __shared__ int vsz[];*/

	int BC = 828;
	int BCt = 1215;

	int b = threadIdx.x + blockDim.x * blockIdx.x;		// Branch number
	int n = b;											// Particle address (starts out as branch number)
	float vminusx, tvecx, vprimex, vplusx;
	float vminusy, tvecy, vprimey, vplusy;
	float vminusz, tvecz, vprimez, vplusz;
	float Ex, Ey, Ez, Bx, By, Bz;
	float tvec_fact;
	float drx, dry, drz, rij2i, rij_si, rij_x, rij_y, rij_z;
	int na, na2;
	int index;
//	int  writeFlag = 1;
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
		NumTransfer[b] = 0;
		NumKill[b] = 0;
		//if (pnum[b] != 0){ printf("mover pnum[%i] = %i\n", b, pnum[b]); }
		/*if (b == BC){
			for (int n = 0; n < pnum[b]; n++){
				na = b + n*nB;
				printf("ts = %i, pnum[%i] = %i, pq[%i]-ID[%4i] = %i\n",ts, b, pnum[b], na,ID[na], pq[na]);
			}
		}*/


		for (int n = 0; n < pnum[b]; n++){
			na = b + n*nB;
			//if (b == BC && pq[na] == 0){
			//	printf("ERR1: ts = %i, pq[%i] = 0, n = %i, b = %i\n",ts, na, n, b);
			//}
			if (INTERACTION){
				float pEx = 0, pEy = 0, pEz = 0;
				float pBx = 0, pBy = 0, pBz = 0;
				float pX = px[na], pY = py[na], pZ = pz[na];
				for (int nn = 0; nn < pnum[b]; nn++){
					//first calculate relative position vector for electrostatic force
					na2 = b + nn*nB;
					drx = pX - px[na2];
					dry = pY - py[na2];
					drz = pZ - pz[na2];

					rij2i = 1 / (drx*drx + dry*dry + drz*drz);	// inverse square

					rij_si = sqrtf(rij2i);					// inverse sqrt so we can use it for multiplication rather than division below
					if (rij_si > inv_thresh) { continue; }		// too close to infinite interaction, so skip this particle
					rij_x = drx * rij_si;						// unit vector direction x
					rij_y = dry * rij_si;						// unit vector direction y
					rij_z = drz * rij_si;						// unit vector direction z

					// Calculate acceleration from electric field
					pEx += qRPM*rij_x*rij2i;
					pEy += qRPM*rij_y*rij2i;
					pEz += qRPM*rij_z*rij2i;

					//Calculate acceleration from magnetic field
					pBx += (vy[na2] * drz - vz[na2] * dry) * rij2i;
					pBy += (vz[na2] * drx - vx[na2] * drz) * rij2i;
					pBz += (vx[na2] * dry - vy[na2] * drx) * rij2i;

				}
				Ex += pEx*Oo4piEps0*qRPM;
				Ey += pEy*Oo4piEps0*qRPM;
				Ez += pEz*Oo4piEps0*qRPM;
				Bx += pBx*mu0o4pi*qRPM;
				By += pBy*mu0o4pi*qRPM;
				Bz += pBz*mu0o4pi*qRPM;
			}

			//if (writeFlag == 1 || writeFlag == 0){ printf("A1: b = %i, n = %i, pz[%i] = %f, vz[%i] = %2.0f, q=%i\n", b, n, na, pz[na], na, vz[na], pq[na]); }

			vminusx = vx[na] + qom*pq[na] * Ex*0.5*dt;
			vminusy = vy[na] + qom*pq[na] * Ey*0.5*dt;
			vminusz = vz[na] + qom*pq[na] * Ez*0.5*dt;
			tvecx = qom*pq[na] * Bx*0.5*dt;
			tvecy = qom*pq[na] * By*0.5*dt;
			tvecz = qom*pq[na] * Bz*0.5*dt;
			tvec_fact = 2 / (1 + tvecx*tvecx + tvecy*tvecy + tvecz*tvecz);
			vprimex = vminusx + vminusy*tvecz - vminusz*tvecy;
			vprimey = vminusy + vminusz*tvecx - vminusx*tvecz;
			vprimez = vminusz + vminusx*tvecy - vminusy*tvecx;
			vplusx = vminusx + (vprimey*tvecz - vprimez*tvecy)*tvec_fact;
			vplusy = vminusy + (vprimez*tvecx - vprimex*tvecz)*tvec_fact;
			vplusz = vminusz + (vprimex*tvecy - vprimey*tvecx)*tvec_fact;
			vx[na] = vplusx + qom*pq[na] * Ex*0.5*dt;
			vy[na] = vplusy + qom*pq[na] * Ey*0.5*dt;
			vz[na] = vplusz + qom*pq[na] * Ez*0.5*dt;
			//if (writeFlag == 1 || writeFlag == 0){ printf("B1: b = %i, n = %i, pz[%i] = %f, vz[%i] = %2.0f, q=%i\n", b, n, na, pz[na], na, vz[na], pq[na]); writeFlag = 0; }
			// ------ Update Particle positions -------------- //
			px[na] += vx[na] * dt;
			py[na] += vy[na] * dt;
			pz[na] += vz[na] * dt;
			//printf("Bx = %e, By = %e, Bz = %e\n", Bx, By, Bz);
			//printf("tx = %e, ty = %e, tz = %e\n", tvecx, tvecy, tvecz);
			//printf("Ex = %e, Ey = %e, Ez = %e, dt = %e\n", Ex, Ey, Ez, dt);
			//printf("vx[%4i] = %e, E = %e, a[%i] = %e\n", na, vx[na], Ex, na, qom*Ex);
			ax[na] = qom*pq[na] * (Ex + py[na] * Bz - pz[na] * By);
			ay[na] = qom*pq[na] * (Ey + pz[na] * Bx - px[na] * Bz);
			az[na] = qom*pq[na] * (Ez + px[na] * By - py[na] * Bx);


			float rbound = pz[na] * (R2 - R1) / L + R1;
			if (pz[na] > L || pz[na] < 0 || px[na] * px[na] + py[na] * py[na] > rbound*rbound){
				int indexK = b + NumKill[b] * nB;
				KillIndex[indexK] = na;
				KillFlag[na] = 1;
				NumKill[b]++;
				pq[na] = 0;
			}else{
				KillFlag[na] = 0;
				// if we aren't killing the particle then we check if the particle is being transferred to another cell

				//if (writeFlag == 1){ printf("A2: b = %i, n = %i, pz[%i] = %f\n", b, n, na, pz[na]); writeFlag = 0; }

				//if (NumKill[b] > 0){
				//printf("NumKill[%i] =  %i, px[%i] = %f, py[%i] = %f, pz[%i] = %f\n", b, NumKill[b],na,px[na],na,py[na],na,pz[na]);
				//}

				if (px[na] < Bxmin[b] && Bxm[b] != -1 && KillFlag[na] == 0){
					index = b + NumTransfer[b] * nB;		// destination index of this particle in the buffer arrays (pxBuffer, and any other array of size np_buffer)
					DestinationsBuffer[index] = Bxm[b];		// the index of the branch to which this particle is being transferred
					//TransferIndex[NumTransfer[b]] = na;		// the index in np_branches stored into an array of size np_buffer, storing the original address for the values of arrays such as pxBuffer etc.  We use this so we dont have to loop through every particle during FillGaps to check for transfers
					TransferIndex[index] = na;
					TransferFlag[na] = 1;					// Transfer flag is of size np_branches (same size as px etc.) and just lets us know that this particle is being transferred to another cell and so should not be used to fill another gap
					pxBuffer[index] = px[na];
					pyBuffer[index] = py[na];
					pzBuffer[index] = pz[na];
					vxBuffer[index] = vx[na];
					vyBuffer[index] = vy[na];
					vzBuffer[index] = vz[na];
					pqBuffer[index] = pq[na];
					IDBuffer[index] = ID[na];
					//pq[na] = 0;	// clear out particle charge so we don't think it's there later (there should be a better way of marking non-existent particles...)
					NumTransfer[b]++;
					//printf("px[%i] = %f, Bxmin[%i] = %f\n", na, px[na], b, Bxmin[b]);
					//printf("particle %i in branch %i Destination %i\n", na, b, DestinationsBuffer[index]);
				}
				else if (px[na] > Bxmax[b] && Bxp[b] != -1 && KillFlag[na] == 0){
					index = b + NumTransfer[b] * nB;
					DestinationsBuffer[index] = Bxp[b];
					//TransferIndex[NumTransfer[b]] = na;
					TransferIndex[index] = na;
					TransferFlag[na] = 1;
					pxBuffer[index] = px[na];
					pyBuffer[index] = py[na];
					pzBuffer[index] = pz[na];
					vxBuffer[index] = vx[na];
					vyBuffer[index] = vy[na];
					vzBuffer[index] = vz[na];
					pqBuffer[index] = pq[na];
					IDBuffer[index] = ID[na];
					//pq[na] = 0;	// clear out particle charge so we don't think it's there later (there should be a better way of marking non-existent particles...)
					NumTransfer[b]++;
					//printf("px[%i] = %f, Bxmax[%i] = %f\n", na, px[na], b, Bxmax[b]);
					//printf("particle %i in branch %i Destination %i\n", na, b, DestinationsBuffer[index]);

				}
				else if (py[na] < Bymin[b] && Bym[b] != -1 && KillFlag[na] == 0){
					index = b + NumTransfer[b] * nB;
					DestinationsBuffer[index] = Bym[b];
					//TransferIndex[NumTransfer[b]] = na;
					TransferIndex[index] = na;
					TransferFlag[na] = 1;
					pxBuffer[index] = px[na];
					pyBuffer[index] = py[na];
					pzBuffer[index] = pz[na];
					vxBuffer[index] = vx[na];
					vyBuffer[index] = vy[na];
					vzBuffer[index] = vz[na];
					pqBuffer[index] = pq[na];
					IDBuffer[index] = ID[na];
					//pq[na] = 0;	// clear out particle charge so we don't think it's there later (there should be a better way of marking non-existent particles...)
					NumTransfer[b]++;
					//printf("py[%i] = %f, Bymin[%i] = %f\n", na, py[na], b, Bymin[b]);
					//printf("particle %i in branch %i Destination %i\n", na, b, DestinationsBuffer[index]);

				}
				else if (py[na] > Bymax[b] && Byp[b] != -1 && KillFlag[na] == 0){
					//printf("py[%i] = %e, Bymax[%i] = %e, Byp[%i] = %i\n", na, py[na], b, Bymax[b], b, Byp[b]);
					//printf("NumTransfer[%i] = %i\n", b,NumTransfer[b]);
					index = b + NumTransfer[b] * nB;
					DestinationsBuffer[index] = Byp[b];
					//TransferIndex[NumTransfer[b]] = na;
					TransferIndex[index] = na;
					TransferFlag[na] = 1;
					pxBuffer[index] = px[na];
					pyBuffer[index] = py[na];
					pzBuffer[index] = pz[na];
					vxBuffer[index] = vx[na];
					vyBuffer[index] = vy[na];
					vzBuffer[index] = vz[na];
					pqBuffer[index] = pq[na];
					IDBuffer[index] = ID[na];
					//pq[na] = 0;	// clear out particle charge so we don't think it's there later (there should be a better way of marking non-existent particles...)
					NumTransfer[b]++;
					//printf("py[%i] = %f, Bymax[%i] = %f\n", na, py[na], b, Bymax[b]);
					//printf("particle %i in branch %i Destination %i\n", na, b, DestinationsBuffer[index]);

				}
				else if (pz[na] < Bzmin[b] && Bzm[b] != -1 && KillFlag[na] == 0){
					index = b + NumTransfer[b] * nB;
					DestinationsBuffer[index] = Bzm[b];
					//TransferIndex[NumTransfer[b]] = na;
					TransferIndex[index] = na;
					TransferFlag[na] = 1;
					pxBuffer[index] = px[na];
					pyBuffer[index] = py[na];
					pzBuffer[index] = pz[na];
					vxBuffer[index] = vx[na];
					vyBuffer[index] = vy[na];
					vzBuffer[index] = vz[na];
					pqBuffer[index] = pq[na];
					IDBuffer[index] = ID[na];
					//pq[na] = 0;	// clear out particle charge so we don't think it's there later (there should be a better way of marking non-existent particles...)
					NumTransfer[b]++;
					//printf("pz[%i] = %f, Bzmin[%i] = %f\n", na, pz[na], b, Bzmin[b]);
					//printf("particle %i in branch %i Destination %i\n", na, b, DestinationsBuffer[index]);
				}
				else if (pz[na] > Bzmax[b] && Bzp[b] != -1 && KillFlag[na] == 0){
					index = b + NumTransfer[b] * nB;		// destination index of this particle in the buffer arrays (pxBuffer, and any other array of size np_buffer)
					DestinationsBuffer[index] = Bzp[b];		// the index of the branch to which this particle is being transferred
					//TransferIndex[NumTransfer[b]] = na;		// the index in np_branches stored into an array of size np_buffer, storing the original address for the values of arrays such as pxBuffer etc.  We use this so we dont have to loop through every particle during FillGaps to check for transfers
					TransferIndex[index] = na;
					TransferFlag[na] = 1;					// Transfer flag is of size np_branches (same size as px etc.) and just lets us know that this particle is being transferred to another cell and so should not be used to fill another gap
					pxBuffer[index] = px[na];
					pyBuffer[index] = py[na];
					pzBuffer[index] = pz[na];
					vxBuffer[index] = vx[na];
					vyBuffer[index] = vy[na];
					vzBuffer[index] = vz[na];
					pqBuffer[index] = pq[na];
					IDBuffer[index] = ID[na];

					/*if (b == BC){
						//printf("ts = %i, PM: particle[%i] index [%i] charge [%i] from %i to %i, px=%1.4f\n",ts, na, index, pq[na], b, Bzp[b],px[na]);
						printf("ts = %i, PM: p[%4i]-ID[%4i] index [%i] charge [%i] from %i to %i, px=%1.4f\n", ts, na, ID[na], index, pq[na], b, Bzp[b], px[na]);
					}*/
					//pq[na] = 0;	// clear out particle charge so we don't think it's there later (there should be a better way of marking non-existent particles...)
					NumTransfer[b]++;

					//printf("pz[%i] = %f, Bzmax[%i] = %f\n", na, pz[na], b, Bzmax[b]);
					//printf("particle %i in branch %i Destination %i\n", na, b, DestinationsBuffer[index]);

				}
				else{
					TransferFlag[na] = 0;
					//printf("particle %i in branch %i no transfer\n", na, b);
					index = -1;
				}
				/*if (index != -1 && DestinationsBuffer[index] == BC){
					printf("ts = %i, pqBuffer[%i]-ID[%i] = %i, na=%i, n=%i of %i\n", ts, index, IDBuffer[index], pqBuffer[index], na,n,pnum[b] );
				}*/
				if (index != -1){
					pq[na] = 0;
				}
			} // end if/else kill statement
			//if (writeFlag == 1 || writeFlag == 0){ printf("A: buffAddress = %i, pz[%i] = %f\n", index, index, pzBuffer[index]); }
		}	// end loop through particles
		b += blockDim.x*gridDim.x;
	}	// end branch while
}	// end function
