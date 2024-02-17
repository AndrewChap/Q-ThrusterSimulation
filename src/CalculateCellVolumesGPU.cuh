__global__ void CalculateCellVolumesGPU(
	float *branchVolumes,					// array where we write all the volumes of the branches/cells
	float *Bxmin, float *Bxmax,				// arrays for the coordinates for of the extent of the branch
	float *Bymin, float *Bymax,
	float *Bzmin, float *Bzmax,
	float dxB, float dyB, float dzB,		// spacing between branches
	int nB,									// number of branches
	int nx, int ny, int nz,					// how many subcells we are dividing to in each dimension
	float L, float R1, float R2
	){
	int b = threadIdx.x + blockDim.x * blockIdx.x;		// Branch number
	//printf("transfer launched\n");
	while (b < nB){
		float dx = (Bxmax[b] - Bxmin[b]) / nx;
		float dy = (Bymax[b] - Bymin[b]) / ny;
		float dz = (Bzmax[b] - Bzmin[b]) / nz;
		float vol = 0;						// total volume of the branch-cell inside the box, initialize to zero
		float subvol = dx*dy*dz;			// volume of a subcell in the branch (basically the dV volume element in a numerical volume integral)
		for (int k = 0; k < nz; k++){
			float z = Bzmin[b] + dz*((float)k + 0.5f);
			float rbound = z * (R2 - R1) / L + R1;					// radius of the cone at our z coordinate
			float rbound2 = rbound*rbound;							// radius squared
			for (int j = 0; j < ny; j++){
				float y = Bymin[b] + dy*((float)j + 0.5f);
				for (int i = 0; i < nx; i++){
					float x = Bxmin[b] + dx*((float)i + 0.5f);
					int incone = z < L && z > 0 && x*x + y*y < rbound2;		// boolean to tell us if the center of the subcell is in the cone or not
					vol += incone*subvol;
				}
			}
		}
		branchVolumes[b] = vol;
		b += blockDim.x*gridDim.x;
	}
}