void TransferParticlesCPU(
	float *px, float *py, float *pz,
	float *vx, float *vy, float *vz,
	int *pq,
	float *pxBuffer, float *pyBuffer, float *pzBuffer,
	float *vxBuffer, float *vyBuffer, float *vzBuffer,
	int *pqBuffer,
	int nB,
	int *pnum,
	int *DestinationsBuffer, int *TransferIndex,
	int *TransferFlag,
	int *NumTransfer,
	int np_branches
	){
	//printf("-\n");
	for (int b = 0; b < nB; b++){
		for (int i = 0; i < NumTransfer[b]; i++){
			int readAddress = b + i*nB;									// read addresses are sequential since all indices in the "Buffer" arrays below correspond to particles getting transferred
			//printf("b = %i, i = %i, nB = %i, readAddress = %i\n", b, i, nB, readAddress);
			//printf("px[%i] = %f\n", readAddress, px[readAddress]);
			//if (b == 0) printf("bNumTransfer[%i] = %i\n", b, NumTransfer[b]);
			//printf("b = %i, DestinationsBuffer[%i] = %i\n", b, readAddress, DestinationsBuffer[readAddress], DestinationsBuffer[readAddress]);
			//if (b == 0) printf("b = %i, DestinationsBuffer[%i] = %i\n", b, readAddress, DestinationsBuffer[readAddress]);
			//if (b == 0) printf("b = %i, DestinationsBuffer[%i] = %i, pnum[%i] = %i\n", b, readAddress, DestinationsBuffer[readAddress], DestinationsBuffer[readAddress], pnum[DestinationsBuffer[readAddress]]);
			int writeAddress = DestinationsBuffer[readAddress] + pnum[DestinationsBuffer[readAddress]] * nB;	// we are writing to an index that is after the last particle i.e. the pnum for that branch
			//printf("DestinationsBuffer[%i] = %i, pnum[%i] = %i, writeAddress = %i\n",readAddress, DestinationsBuffer[readAddress], DestinationsBuffer[readAddress], pnum[DestinationsBuffer[readAddress]], writeAddress);
			//if (writeAddress > np_branches){ printf("writeAddress = %i, b = %i, pnum[%i] = %i, readAddress = %i, DesBuf =[%i] = %i\n", writeAddress, b, DestinationsBuffer[readAddress], pnum[DestinationsBuffer[readAddress]], readAddress, DestinationsBuffer[readAddress]); }
			px[writeAddress] = pxBuffer[readAddress];
			py[writeAddress] = pyBuffer[readAddress];
			pz[writeAddress] = pzBuffer[readAddress];
			vx[writeAddress] = vxBuffer[readAddress];
			vy[writeAddress] = vyBuffer[readAddress];
			vz[writeAddress] = vzBuffer[readAddress];
			pq[writeAddress] = pqBuffer[readAddress];
			pnum[DestinationsBuffer[readAddress]]++;								// Increment the pnum of the destination branch, since we just added a particle to it
		}
	}
}