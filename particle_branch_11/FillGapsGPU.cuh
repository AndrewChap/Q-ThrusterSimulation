__global__ void FillGapsGPU(
	float *px, float *py, float *pz,
	float *vx, float *vy, float *vz,
	float *ax, float *ay, float *az,
	int *Bxm, int *Bxp, int *Bym, int *Byp, int *Bzm, int *Bzp,
	int *pq,
	int nB,
	int *pnum,
	int *NumTransfer, int *NumKill,
	int *TransferFlag, int *KillFlag,
	int *TransferIndex, int *KillIndex, int ts, int *ID
){
	int b = threadIdx.x + blockDim.x * blockIdx.x;		// Branch number
	int writeFlag = 1;
	int BC = 828;
	int BCt = 1215;

	//printf("fill gaps launched\n");
	while (b < nB){
	//for (int b = 0; b < nB; b++){
		/*if (b == BC && NumTransfer[b]>0){
			printf("fg: pnum[%i] = %i, NumTransfer = %i\n", b, pnum[b], NumTransfer[b]);
		}*/
		int replacer = pnum[b] - 1;		// when we loop though all the particles we find ones that have been transferred and replace them with particles from the end of the list, with index "replacer" intizialized here to the last particle in the list
		//if (NumTransfer[b] != 0 && b < 10){ printf("NumTransfer[%i] = %i , pnum[%i] from %i to %i\n",b,NumTransfer[b], b, pnum[b], pnum[b] - NumTransfer[b]); }
		pnum[b] -= NumTransfer[b];		// decrease the particle count by the number of particles that were transferred

		/*if (b == BC){
			for (int n = 0; n < pnum[b]; n++){
				int na = b + n*nB;
				if (pq[na] == 0){
					printf("ERROR2-A ts=%i, pq[%i] = 0, b = %i, n = %i\n",ts, na, b, n);
				}
			}
		}*/
		for (int n = 0; n < NumTransfer[b]; n++){		// loop through the number of particles that needed to be transferred
			int na = TransferIndex[b+n*nB];					// index of particle to be transferred
			int transfer = 0;							// 
			while (transfer == 0 && replacer >= pnum[b]){	// while we haven't found a particle to fill the gap AND while we haven't finished by getting to the last particle in pnum
				int replacer_mapped = b + replacer*nB;		// map the particle number "index" to the actual index in the particle array (the particle that we're moving to fill the empty gap)
				if (TransferFlag[replacer_mapped] == 0 && na != replacer_mapped){
					/*if (b == BC && NumTransfer[b] > 0){
						printf("fgtr: ts=%i, px[%i]-ID[%i] = %1.4f replaced by px[%i]-ID[%i] = %1.4f \n",ts, na,ID[na], px[na], replacer_mapped, ID[replacer_mapped], px[replacer_mapped]);
					}*/
					px[na] = px[replacer_mapped];
					py[na] = py[replacer_mapped];
					pz[na] = pz[replacer_mapped];
					//if (writeFlag == 1){ printf("B1: b = %i, n = %i, pz[%i] = %f\n", b, n, na, pz[na]); writeFlag = 0; }
					vx[na] = vx[replacer_mapped];
					vy[na] = vy[replacer_mapped];
					vz[na] = vz[replacer_mapped];
					ax[na] = ax[replacer_mapped];
					ay[na] = ay[replacer_mapped];
					az[na] = az[replacer_mapped];
					pq[na] = pq[replacer_mapped];
					ID[na] = ID[replacer_mapped];
					/*if (b == BC && pq[replacer_mapped] == 0){
						printf("ERROR2-B ts=%i, about to replace pq[%i]-ID[%i]=%i with pq[%i]-ID[%i]=%i\n",ts, na, ID[na], pq[na], replacer_mapped, ID[replacer_mapped], pq[replacer_mapped]);

					}*/
					pq[replacer_mapped] = 0;	// set the charge of the old location to zero so that it doesn't accidentally get plotted
					transfer = 1;
				}
				replacer--;
			}
		}
		/*if (b == BC || Bxm[b] == BC || Bxp[b] == BC || Bym[b] == BC || Byp[b] == BC || Bzm[b] == BC || Bzp[b] == BC){
			for (int n = 0; n < pnum[b]; n++){
				int na = b + n*nB;
				if (pq[na] == 0){
					if (b != BC)
						printf("ERROR2-C ts=%i, pq[%i]-ID[%i] = 0, b = %i, n = %i of %i\n",ts, na,ID[na], b, n,pnum[b]);
					else{
						printf("SUPERERROR2-C ts=%i, pq[%i]-ID[%i] = 0, b = %i, n = %i\n", ts, na, ID[na], b, n);
						//exit(0);
					}

				}

			}
		}*/
		replacer = pnum[b] - 1;		// when we loop though all the particles we find ones that have been killed and replace them with particles from the end of the list, with index "replacer" intizialized here to the last particle in the list
		pnum[b] -= NumKill[b];		// decrease the particle count by the number of particles that were killed
		/*if (NumKill[b] > 0){
			printf("NumKill[%i] =  %i, pnum[%i] = %i\n", b, NumKill[b], b, pnum[b]);
			getchar();
		}*/
		//printf("NumKill[%i] = %i\n", b, NumKill[b]);
		for (int n = 0; n < NumKill[b]; n++){	// loop through the number of particles that needed to be killed
			int na = KillIndex[b + n*nB];					// index of particle to be killed
			int kill = 0;						// 
			while (kill == 0 && replacer >= pnum[b]){	// while we haven't found a particle to fill the gap AND while we haven't finished by getting to the last particle in pnum
				int replacer_mapped = b + replacer*nB;		// map the particle number "index" to the actual index in the particle array
				if (KillFlag[replacer_mapped] == 0){
					px[na] = px[replacer_mapped];
					py[na] = py[replacer_mapped];
					pz[na] = pz[replacer_mapped];
					//if (writeFlag == 1){ printf("B2: b = %i, n = %i, pz[%i] = %f\n", b, n, na, pz[na]); writeFlag = 0; }
					vx[na] = vx[replacer_mapped];
					vy[na] = vy[replacer_mapped];
					vz[na] = vz[replacer_mapped];
					ax[na] = ax[replacer_mapped];
					ay[na] = ay[replacer_mapped];
					az[na] = az[replacer_mapped];
					pq[na] = pq[replacer_mapped];
					pq[replacer_mapped] = 0;	// set the charge of the old location to zero so that it doesn't accidentally get plotted
					kill = 1;
				}
				replacer--;
			}
		}
		b += blockDim.x*gridDim.x;
	}
}