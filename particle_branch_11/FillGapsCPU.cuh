void FillGapsCPU(
	float *px, float *py, float *pz,
	float *vx, float *vy, float *vz,
	int *pq,
	int nB,
	int *pnum,
	int *NumTransfer, int *NumKill,
	int *TransferFlag, int *KillFlag,
	int *TransferIndex, int *KillIndex
){
	for (int b = 0; b < nB; b++){
		int replacer = pnum[b] - 1;		// when we loop though all the particles we find ones that have been transferred and replace them with particles from the end of the list, with index "replacer" intizialized here to the last particle in the list
		pnum[b] -= NumTransfer[b];		// decrease the particle count by the number of particles that were transferred
		for (int n = 0; n < NumTransfer[b]; n++){	// loop through the number of particles that needed to be transferred
			int na = TransferIndex[n];					// index of particle to be transferred
			int transfer = 0;						// 
			while (transfer == 0 && replacer >= pnum[b]){	// while we haven't found a particle to fill the gap AND while we haven't finished by getting to the last particle in pnum
				int replacer_mapped = b + replacer*nB;		// map the particle number "index" to the actual index in the particle array
				if (TransferFlag[replacer_mapped] == 0 && na != replacer_mapped){
					px[na] = px[replacer_mapped];
					py[na] = py[replacer_mapped];
					pz[na] = pz[replacer_mapped];
					vx[na] = vx[replacer_mapped];
					vy[na] = vy[replacer_mapped];
					vz[na] = vz[replacer_mapped];
					pq[na] = pq[replacer_mapped];
					pq[replacer_mapped] = 0;	// set the charge of the old location to zero so that it doesn't accidentally get plotted
					transfer = 1;
				}
				replacer--;
			}
		}
		replacer = pnum[b] - 1;		// when we loop though all the particles we find ones that have been killed and replace them with particles from the end of the list, with index "replacer" intizialized here to the last particle in the list
		pnum[b] -= NumKill[b];		// decrease the particle count by the number of particles that were killed
		/*if (NumKill[b] > 0){
			printf("NumKill[%i] =  %i, pnum[%i] = %i\n", b, NumKill[b], b, pnum[b]);
			getchar();
		}*/
		//printf("NumKill[%i] = %i\n", b, NumKill[b]);
		for (int n = 0; n < NumKill[b]; n++){	// loop through the number of particles that needed to be killed
			int na = KillIndex[n];					// index of particle to be killed
			int kill = 0;						// 
			while (kill == 0 && replacer >= pnum[b]){	// while we haven't found a particle to fill the gap AND while we haven't finished by getting to the last particle in pnum
				int replacer_mapped = b + replacer*nB;		// map the particle number "index" to the actual index in the particle array
				if (KillFlag[replacer_mapped] == 0){
					px[na] = px[replacer_mapped];
					py[na] = py[replacer_mapped];
					pz[na] = pz[replacer_mapped];
					vx[na] = vx[replacer_mapped];
					vy[na] = vy[replacer_mapped];
					vz[na] = vz[replacer_mapped];
					pq[na] = pq[replacer_mapped];
					pq[replacer_mapped] = 0;	// set the charge of the old location to zero so that it doesn't accidentally get plotted
					kill = 1;
				}
				replacer--;
			}
		}
	}
}