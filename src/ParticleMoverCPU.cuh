#include <math.h>

#define pi			3.141592653589793			// pi
#define eps0		8.85418782E-12				// Permittivity of free space
#define mu0			1.2566370614E-6				// Vacuum permeability; N-s2/C2
#define mu0o4pi		1E-7						// mu0/(4*pi)
#define Oo4piEps0	8.9875517873681764e9		// 1/(4*pi*eps0)
#define QE			1.602176565E-19				// elementary charge (C)
#define ME			9.10938215E-31				// electron rest mass (kg)

void ParticleMoverCPU(
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
	float *pxBuffer, float *pyBuffer, float *pzBuffer,
	float *vxBuffer, float *vyBuffer, float *vzBuffer,
	int *pqBuffer,
	int *DestinationsBuffer, int *TransferIndex, int *KillIndex,
	int *TransferFlag, int *KillFlag,
	int *NumTransfer, int *NumKill,
	int *Bxm, int *Bxp, int *Bym, int *Byp, int *Bzm, int *Bzp,
	float *Bxmin, float *Bxmax, float *Bymin, float *Bymax, float *Bzmin, float *Bzmax,
	int *pnum, int nB, float dt, float qom,
	float LX, float LY,
	float cosphase, float sinphase,
	float L, float R1, float R2, float inv_thresh,
	float qRPM, int INTERACTION)

{
	float vminusx, tvecx, vprimex, vplusx;
	float vminusy, tvecy, vprimey, vplusy;
	float vminusz, tvecz, vprimez, vplusz;
	float Ex, Ey, Ez, Bx, By, Bz;
	float tvec_fact;
	float drx, dry, drz, rij2i, rij_si, rij_x, rij_y, rij_z;
	int na, na2;
	int index;

	for (int b = 0; b < nB; b++){
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

		NumTransfer[b] = 0;	// initialize to zero each time step
		NumKill[b] = 0;// initialize to zero each time step

		for (int n = 0; n < pnum[b]; n++){
			na = b + n*nB;
			if (INTERACTION){
				for (int nn = 0; nn < pnum[b]; nn++){
					//first calculate relative position vector for electrostatic force
					na2 = b + nn*nB;
					drx = px[na] - px[na2];
					dry = py[na] - py[na2];
					drz = pz[na] - pz[na2];
					rij2i = 1/(drx*drx + dry*dry + drz*drz);	// inverse square

					rij_si = pow(rij2i, 0.5);					// inverse sqrt so we can use it for multiplication rather than division below
					if (rij_si > inv_thresh) { continue; }		// too close to infinite interaction, so skip this particle 
					rij_x = drx * rij_si;						// unit vector direction x
					rij_y = dry * rij_si;						// unit vector direction y
					rij_z = drz * rij_si;						// unit vector direction z

					//second calculate relative velocity for Lorentz force
					/*dvx = vx[na] - vx[na2];
					dvy = vx[na] - vx[na2];
					dvz = vx[na] - vx[na2];*/

					// Calculate acceleration from electric field
					Ex += Oo4piEps0*qRPM*rij_x*rij2i;
					Ey += Oo4piEps0*qRPM*rij_y*rij2i;
					Ez += Oo4piEps0*qRPM*rij_z*rij2i;

					//Calculate acceleration from magnetic field
					Bx += mu0o4pi*qRPM*(vy[na2] * drz - vz[na2] * dry) * rij2i;
					By += mu0o4pi*qRPM*(vz[na2] * drx - vx[na2] * drz) * rij2i;
					Bz += mu0o4pi*qRPM*(vx[na2] * dry - vy[na2] * drx) * rij2i;

					/*prt[n].aB.x += N_qom*(dvy*Bjz - dvz*Bjy);
					prt[n].aB.y += N_qom*(dvz*Bjx - dvx*Bjz);
					prt[n].aB.z += N_qom*(dvx*Bjy - dvy*Bjx);

					prt[n].aBp.x += p_qom*(dvy*Bjz - dvz*Bjy);
					prt[n].aBp.y += p_qom*(dvz*Bjx - dvx*Bjz);
					prt[n].aBp.z += p_qom*(dvx*Bjy - dvy*Bjx);*/
				}
			}

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

			// ------ Update Particle positions -------------- //
			px[na] += vx[na] * dt;
			py[na] += vy[na] * dt;
			pz[na] += vz[na] * dt;

			// ------ Check if Particle has crossed a boundary -------- //

			/*
			int index = b + NumTransfer[b] * nB;

			int Dest =
			Bxm[b] * (px[na] < Bxmin[b]) + Bxp[b] * (px[na] > Bxmax[b]) +
			Bym[b] * (py[na] < Bymin[b]) + Byp[b] * (py[na] > Bymax[b]) +
			Bzm[b] * (pz[na] < Bzmin[b]) + Bzp[b] * (pz[na] > Bzmax[b]);

			DestinationsBuffer[index] = Dest;
			NumTransfer[b] += (Dest > 0);
			pxBuffer[index] = px[na];
			pyBuffer[index] = py[na];
			pzBuffer[index] = pz[na];
			vxBuffer[index] = vx[na];
			vyBuffer[index] = vy[na];
			vzBuffer[index] = vz[na];
			*/


			float rbound = pz[na] * (R2 - R1) / L + R1;
			if (pz[na] > L || pz[na] < 0 || px[na] * px[na] + py[na] * py[na] > rbound*rbound){
				KillIndex[NumKill[b]] = na;
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
					index = b + NumTransfer[b] * nB;	// index in the buffer arrays
					DestinationsBuffer[index] = Bxm[b];
					TransferIndex[NumTransfer[b]] = na;
					TransferFlag[na] = 1;
					pxBuffer[index] = px[na];
					pyBuffer[index] = py[na];
					pzBuffer[index] = pz[na];
					vxBuffer[index] = vx[na];
					vyBuffer[index] = vy[na];
					vzBuffer[index] = vz[na];
					pqBuffer[index] = pq[na];
					pq[na] = 0;	// clear out particle charge so we don't think it's there later (there should be a better way of marking non-existent particles...)
					NumTransfer[b]++;
					//printf("px[%i] = %f, Bxmin[%i] = %f\n", na, px[na], b, Bxmin[b]);
					//printf("particle %i in branch %i Destination %i\n", na, b, DestinationsBuffer[index]);
				}
				else if (px[na] > Bxmax[b] && Bxp[b] != -1 && KillFlag[na] == 0){
					index = b + NumTransfer[b] * nB;
					DestinationsBuffer[index] = Bxp[b];
					TransferIndex[NumTransfer[b]] = na;
					TransferFlag[na] = 1;
					pxBuffer[index] = px[na];
					pyBuffer[index] = py[na];
					pzBuffer[index] = pz[na];
					vxBuffer[index] = vx[na];
					vyBuffer[index] = vy[na];
					vzBuffer[index] = vz[na];
					pqBuffer[index] = pq[na];
					pq[na] = 0;	// clear out particle charge so we don't think it's there later (there should be a better way of marking non-existent particles...)
					NumTransfer[b]++;
					//printf("px[%i] = %f, Bxmax[%i] = %f\n", na, px[na], b, Bxmax[b]);
					//printf("particle %i in branch %i Destination %i\n", na, b, DestinationsBuffer[index]);

				}
				else if (py[na] < Bymin[b] && Bym[b] != -1 && KillFlag[na] == 0){
					index = b + NumTransfer[b] * nB;
					DestinationsBuffer[index] = Bym[b];
					TransferIndex[NumTransfer[b]] = na;
					TransferFlag[na] = 1;
					pxBuffer[index] = px[na];
					pyBuffer[index] = py[na];
					pzBuffer[index] = pz[na];
					vxBuffer[index] = vx[na];
					vyBuffer[index] = vy[na];
					vzBuffer[index] = vz[na];
					pqBuffer[index] = pq[na];
					pq[na] = 0;	// clear out particle charge so we don't think it's there later (there should be a better way of marking non-existent particles...)
					NumTransfer[b]++;
					//printf("py[%i] = %f, Bymin[%i] = %f\n", na, py[na], b, Bymin[b]);
					//printf("particle %i in branch %i Destination %i\n", na, b, DestinationsBuffer[index]);

				}
				else if (py[na] > Bymax[b] && Byp[b] != -1 && KillFlag[na] == 0){
					//printf("py[%i] = %e, Bymax[%i] = %e, Byp[%i] = %i\n", na, py[na], b, Bymax[b], b, Byp[b]);
					//printf("NumTransfer[%i] = %i\n", b,NumTransfer[b]);
					index = b + NumTransfer[b] * nB;
					DestinationsBuffer[index] = Byp[b];
					TransferIndex[NumTransfer[b]] = na;
					TransferFlag[na] = 1;
					pxBuffer[index] = px[na];
					pyBuffer[index] = py[na];
					pzBuffer[index] = pz[na];
					vxBuffer[index] = vx[na];
					vyBuffer[index] = vy[na];
					vzBuffer[index] = vz[na];
					pqBuffer[index] = pq[na];
					pq[na] = 0;	// clear out particle charge so we don't think it's there later (there should be a better way of marking non-existent particles...)
					NumTransfer[b]++;
					//printf("py[%i] = %f, Bymax[%i] = %f\n", na, py[na], b, Bymax[b]);
					//printf("particle %i in branch %i Destination %i\n", na, b, DestinationsBuffer[index]);

				}
				else if (pz[na] < Bzmin[b] && Bzm[b] != -1 && KillFlag[na] == 0){
					index = b + NumTransfer[b] * nB;
					DestinationsBuffer[index] = Bzm[b];
					TransferIndex[NumTransfer[b]] = na;
					TransferFlag[na] = 1;
					pxBuffer[index] = px[na];
					pyBuffer[index] = py[na];
					pzBuffer[index] = pz[na];
					vxBuffer[index] = vx[na];
					vyBuffer[index] = vy[na];
					vzBuffer[index] = vz[na];
					pqBuffer[index] = pq[na];
					pq[na] = 0;	// clear out particle charge so we don't think it's there later (there should be a better way of marking non-existent particles...)
					NumTransfer[b]++;
					//printf("pz[%i] = %f, Bzmin[%i] = %f\n", na, pz[na], b, Bzmin[b]);
					//printf("particle %i in branch %i Destination %i\n", na, b, DestinationsBuffer[index]);
				}
				else if (pz[na] > Bzmax[b] && Bzp[b] != -1 && KillFlag[na] == 0){
					index = b + NumTransfer[b] * nB;
					DestinationsBuffer[index] = Bzp[b];
					TransferIndex[NumTransfer[b]] = na;
					TransferFlag[na] = 1;
					pxBuffer[index] = px[na];
					pyBuffer[index] = py[na];
					pzBuffer[index] = pz[na];
					vxBuffer[index] = vx[na];
					vyBuffer[index] = vy[na];
					vzBuffer[index] = vz[na];
					pqBuffer[index] = pq[na];
					pq[na] = 0;	// clear out particle charge so we don't think it's there later (there should be a better way of marking non-existent particles...)
					NumTransfer[b]++;
					//printf("pz[%i] = %f, Bzmax[%i] = %f\n", na, pz[na], b, Bzmax[b]);
					//printf("particle %i in branch %i Destination %i\n", na, b, DestinationsBuffer[index]);

				}
				else{
					TransferFlag[na] = 0;
					//printf("particle %i in branch %i no transfer\n", na, b);
					index = -1;
				}
			} // end if/else kill statement

		}// end loop through particles
	}// end loop through branches
}// end function