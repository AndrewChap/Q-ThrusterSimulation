__global__ void AddParticlesGPU_Initialize(
	curandState *px_rand, curandState *py_rand, curandState *pz_rand,
	curandState *vx_rand, curandState *vy_rand, curandState *vz_rand,
	int nB
	) {
	int b = threadIdx.x + blockDim.x * blockIdx.x;
	while (b < nB){
		curand_init(1337, b, 0, &px_rand[b]);
		curand_init(1338, b, 0, &py_rand[b]);
		curand_init(1330, b, 0, &pz_rand[b]);
		curand_init(1331, b, 0, &vx_rand[b]);
		curand_init(1332, b, 0, &vy_rand[b]);
		curand_init(1333, b, 0, &vz_rand[b]);
		b += blockDim.x*gridDim.x;
	}
}

__global__ void AddParticlesGPU(
	float *px, float *py, float *pz,
	float *vx, float *vy, float *vz,
	int *pq,
	curandState *px_rand, curandState *py_rand, curandState *pz_rand,
	curandState *vx_rand, curandState *vy_rand, curandState *vz_rand,
	float *np_remainder,
	int nB,
	float CPPSperBranch,
	int *pnum,
	int ts,
	float dt,
	float xl, float yl, float zl,
	float xmin, float ymin, float zmin,
	float L, float R1, float R2
	){
	int b = threadIdx.x + blockDim.x * blockIdx.x;		// Branch number
	//printf("transfer launched\n");
	while (b < nB){
		float np_create_float = np_remainder[b] + CPPSperBranch*dt;		// non-integer number of macro-particles created in this cell
		float np_create_round = 2 * floor(np_create_float / 2);			// round down to even number
		np_remainder[b] = np_create_float - np_create_round;		// calculate the remainder to be added to the next time-step's calculation
		int np_create = (int)np_create_round;							// cast as an int for use in calculation
		for (int i = 0; i < np_create/2; i++){
			int na = b + pnum[b] * nB;
			float xrand = curand_uniform(&px_rand[b])*xl + xmin;
			float yrand = curand_uniform(&py_rand[b])*yl + ymin;
			float zrand = curand_uniform(&pz_rand[b])*zl + zmin;
			float rbound = zrand * (R2 - R1) / L + R1;
			float rbound2 = rbound*rbound;
			if (zrand > L || zrand < 0 || (xrand * xrand + yrand * yrand) > rbound2){
				continue;
			}
			px[na] = xrand;
			py[na] = yrand;
			pz[na] = zrand;
			vx[na] = 0;// curand_uniform(&vx_rand[idx]);
			vy[na] = 0;// curand_uniform(&vy_rand[idx]);
			vz[na] = 0;// curand_uniform(&vz_rand[idx]);
			pq[na] = 1;
			px[na + nB] = xrand;
			py[na + nB] = yrand;
			pz[na + nB] = zrand;
			vx[na + nB] = 0;
			vy[na + nB] = 0;
			vz[na + nB] = 0;
			pq[na + nB] = -1;
			pnum[b] += 2;

		}
		b += blockDim.x*gridDim.x;
	}
}
