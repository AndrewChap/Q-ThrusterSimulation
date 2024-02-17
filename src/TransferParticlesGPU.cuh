#include <stdio.h>

__global__ void TransferParticlesGPU(
	float *px, float *py, float *pz,
	float *vx, float *vy, float *vz,
	float *ax, float *ay, float *az,
	int *pq,
	float *pxBuffer, float *pyBuffer, float *pzBuffer,
	float *vxBuffer, float *vyBuffer, float *vzBuffer,
	float *axBuffer, float *ayBuffer, float *azBuffer,
	int *pqBuffer, int *IDBuffer,
	int nB,
	int *pnum,
	int *DestinationsBuffer, int *TransferIndex,
	int *TransferFlag,
	int *NumTransfer, int ts, int *ID
	){
	int b = threadIdx.x + blockDim.x * blockIdx.x;		// Branch number
	int writeFlag = 1;
	int BC = 828;
	int BCt = 1215;
	//printf("transfer launched\n");
	while (b < nB){
	//for (int b = 0; b < nB; b++){
		for (int i = 0; i < NumTransfer[b]; i++){
			int readAddress = b + i*nB;									// where we are reading from in the buffer.  Read addresses are sequential (in the sense of the index "i") since all indices in the "Buffer" arrays below correspond to particles getting transferred
			int pnum_target = atomicAdd(&pnum[DestinationsBuffer[readAddress]], 1);	// we increment pnum first, with atomic add, and store in the output the number of particles in the branch we will be writing to (pnum_target is the number before we add, see the documentation on atomicAdd) so we can append these particles on to the end


			/*if (DestinationsBuffer[readAddress] == BC){
				printf("P-ID[%i] GOING TO %i\n", IDBuffer[readAddress], DestinationsBuffer[readAddress]);
			}
			if (b == BC && pqBuffer[readAddress] == 0){
				printf("ERR3-A: ts=%i, with pqBuffer[%i] = %i\n",ts, readAddress, pq[readAddress]);
			}*/

			//printf("pnum_target = %i\n", pnum_target);
			//if (b == 0) printf("b = %i, DestinationsBuffer[%i] = %i, pnum[%i] = %i\n", b, readAddress, DestinationsBuffer[readAddress], DestinationsBuffer[readAddress], pnum[DestinationsBuffer[readAddress]]);
			//if (writeFlag == 1 || writeFlag == 0){ printf("readAddress = %i, pz[%i] = %f\n",readAddress,readAddress,pz[readAddress]);  }
			//if (writeFlag == 1 || writeFlag == 0){ printf("buffAddress = %i, pz[%i] = %f\n", readAddress, readAddress, pzBuffer[readAddress]); }
			//int writeAddress = DestinationsBuffer[readAddress] + pnum[DestinationsBuffer[readAddress]] * nB;	// we are writing to an index that is after the last particle i.e. the pnum for that branch
			int writeAddress = DestinationsBuffer[readAddress] + (pnum_target) * nB;	// we are writing to an index that is after the last particle i.e. the pnum for that branch
			//printf("DestinationsBuffer[%i] = %i, pnum[%i] = %i, writeAddress = %i\n",readAddress, DestinationsBuffer[readAddress], DestinationsBuffer[readAddress], pnum[DestinationsBuffer[readAddress]], writeAddress);
			//if (writeAddress > np_branches){ printf("writeAddress = %i, b = %i, pnum[%i] = %i, readAddress = %i, DesBuf =[%i] = %i\n", writeAddress, b, DestinationsBuffer[readAddress], pnum[DestinationsBuffer[readAddress]], readAddress, DestinationsBuffer[readAddress]); }
			
			/*if (b == BC && DestinationsBuffer[readAddress] == BCt){
				printf("T%i from B%i: overwrite px[%i] = %1.4f with px[%i] = %1.4f\n", i, b, writeAddress,px[writeAddress],readAddress,px[readAddress]);
			}

			if (b == BC && (pq[writeAddress] != 0 || pqBuffer[readAddress] == 0)){
				printf("ERR3-B: ts=%i, overwrite pq[%i] = %i with pqBuffer[%i] = %i\n", ts, writeAddress, pq[writeAddress], readAddress, pqBuffer[readAddress]);
			}*/
			
			px[writeAddress] = pxBuffer[readAddress];
			py[writeAddress] = pyBuffer[readAddress];
			pz[writeAddress] = pzBuffer[readAddress];
			//if (writeFlag == 1 || writeFlag == 0){ printf("C: b = %i, i = %i, pz[%i] = %f\n", b, i, writeAddress, pz[writeAddress]); writeFlag = 0; }
			vx[writeAddress] = vxBuffer[readAddress];
			vy[writeAddress] = vyBuffer[readAddress];
			vz[writeAddress] = vzBuffer[readAddress];
			ax[writeAddress] = axBuffer[readAddress];
			ay[writeAddress] = ayBuffer[readAddress];
			az[writeAddress] = azBuffer[readAddress];
			pq[writeAddress] = pqBuffer[readAddress];
			ID[writeAddress] = IDBuffer[readAddress];
			/*if (DestinationsBuffer[readAddress] == BC){
				printf("ARR: pID[%i] pq=%i from %i GOING TO %i\n", ID[writeAddress], pqBuffer[readAddress], b,DestinationsBuffer[readAddress]);
			}
			if (b == BC){
				printf("LEV: pID[%i] pq=%i from %i GOING TO %i\n", ID[writeAddress], pqBuffer[readAddress], b, DestinationsBuffer[readAddress]);
			}*/
			//pnum[DestinationsBuffer[readAddress]]++;								// Increment the pnum of the destination branch, since we just added a particle to it
		}
		b += blockDim.x*gridDim.x;
	}
}