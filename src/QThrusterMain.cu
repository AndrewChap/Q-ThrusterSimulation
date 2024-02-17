#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <sys/stat.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "gpuErrchk.cuh"

#include "ParticleMoverCPU.cuh"
#include "ParticleMoverGPU.cuh"
#include "FillGapsCPU.cuh"
#include "FillGapsGPU.cuh"
#include "TransferParticlesCPU.cuh"
#include "TransferParticlesGPU.cuh"
#include "Load_EB.cuh"
#include "BoxConeCollider.cuh"
#include "WriteVTK.cuh"
#include "WriteForce.cuh"
#include "TimerWrappers.cuh"
#include "WriteToSnowballGPU.cuh"
#include "DataIO.cuh"
#include "AddParticlesGPU.cuh"
#include "CalculateCellVolumesGPU.cuh"
#include "ThrustCalculationGPU1.cuh"
#include "ThrustCalculationGPU2.cuh"

#define CPUrun		0							// Run on CPU?	(this way is coded as a GPU code prototype but runs on the CPU)
#define GPUrun		1							// Run on GPU?
#define CPUorig		0							// Original way of running on the CPU
#define AddPartGPU  0							// there are two different ways to add particles to the simulation. If set to 0 particles are created on the CPU and then transferred to memory on the GPU.  If set to 1 the particles are created on the GPU.  I've found option 0 to be faster since very few particles are created per time-step, but in case that changes, I'm keeping both in

#define CPUtimers	3							// number of timers needed to time CPU functions
#define GPUtimers	5							// number of timers needed to time GPU functions

#define pi			3.141592653589793			//
#define eps0		8.85418782E-12				// Permittivity of free space
#define mu0			1.2566370614E-6				// Vacuum permeability; N-s2/C2
#define QE			1.602176565E-19				// elementary charge (C)
#define ME			9.10938215E-31				// electron rest mass (kg)
#define mu0o4pi		1E-7						// mu0/(4*pi)
#define fourpieps0	1.112650056352649e-10		// 4*pi*eps0
#define KILL_MAX    64							// max number of particles that can be killed in a time-step
#define char_amount 128							// number of characters to be allowed for string buffers

#define FACTOR      15  // I played around with these numbers until I got the best speedup
#define THREADS     256 // I played around with these numbers until I got the best speedup

typedef struct{
	float x;
	float y;
	float z;
} VecCPU;           //Struct for vectors for CPU calculation

typedef struct {
	float x;
	float y;
	float z;
} VecGPU;


typedef struct{
	float xmin;
	float xmax;
	float ymin;
	float ymax;
	float zmin;
	float zmax;
	float xl;
	float yl;
	float zl;
	float vol;
}BoxDim;

int main(void){


	// Read in input data to setup simulation variables
	srand(time(NULL)); // seed random number generator
	char *buf;
	char line[200];
	float A[100];
	FILE *file;
	file = fopen("input_deck_2_7.txt", "r");
	if (file == NULL)
	{
		printf("File not found!!");
		exit(0);
	}

	int jf = -1;
	while (fgets(line, sizeof(line), file))
	{
		jf++;
		buf = strtok(line, " \n\t");
		printf(" %s \n", buf);
		if (buf == NULL) { continue; }
		A[jf] = atof(buf);
	}
	fclose(file);

	// Test Article geometry
	float R1 = (float)A[0];				// meters = 5.5 inches
	printf("R1 = %f \n", R1);
	float R2 = (float)A[1];				// meters = 2.5 inches
	printf("R2 = %f \n", R2);
	float L = (float)A[2];				// meters = 9 inches
	printf("L = %f \n", L);
	float LX = (float)A[3];			// meters = 11 inches
	printf("LX = %f \n", LX);
	float LY = (float)A[4];			// meters = 11 inches
	printf("LY = %f \n", LY);
	float LZ = (float)A[5];			// meters = 9 inches
	printf("LZ = %f \n", LZ);
	float COMSOL_CONV = (float)A[6];		// converts inches to meters from COMSOL E & B files
	printf("COMSOL_CONV = %f \n", COMSOL_CONV);
	float GRIDL = (float)A[7];			// meters or 0.5 inches, WHICH MUST MATCH COMSOL FILE!!!!
	printf("GRIDL = %f \n", GRIDL);
	int CELL_LAYERS = (int)A[8];			// number of cell shells to be used to calculate detailed particle-particle interaction
	printf("CELL_LAYERS = %d \n", CELL_LAYERS);
	int INTERACTION = (int)A[9];
	printf("INTERACTION = %d \n", INTERACTION);
	int REPLACE = (int)A[10];
	printf("REPLACE = %d \n", REPLACE);
	int COMSOL_DIVISIONS = (int)A[11];		// number of divisions of 360 degrees -> Comsol can provide E & B field files at certain phase increments
	printf("COMSOL_DIVISIONS = %d \n", COMSOL_DIVISIONS);
	int COMSOL_ROWS = (int)A[12];			// number of rows of E & B fields for COMSOL file
	printf("COMSOL_ROWS = %d \n", COMSOL_ROWS);
	float WRITE_RATE = (float)A[13];				// number of time steps between recording data
	printf("WRITE_RATE = %d \n", WRITE_RATE);
	int DT_PER_PERIOD = (float)A[14];			// number of ts steps per COMSOL phase division
	printf("DT_PER_PERIOD = %f \n", DT_PER_PERIOD);
	int CYLINDER = (int)A[15];
	printf("CYLINDER = %d \n", CYLINDER);
	float THRESH = (float)A[16];			// minimum spacing for electrostatic calculation for particle-particle interaction (m)
	printf("THRESH = %f \n", THRESH);
	int PARTICLE_PACKING = (int)A[17];		// set to 10 if you want 10^3 particles per differential element box
	printf("PARTICLE_PACKING = %d \n", PARTICLE_PACKING);
	float END_TIME = (float)A[18];				// number of integral time steps
	printf("END_TIME = %e \n", END_TIME);
	float REL_MASS = (float)A[19];			// relativistic mass factor of fermions
	printf("REL_MASS = %f \n", REL_MASS);
	float SCALING = (float)A[20];				// Power scaling, take square root to factor E & B fields
	printf("SCALING = %f \n", SCALING);
	SCALING = pow(SCALING, 0.5);
	printf("E & B field magnitudes will be scaled by %f X COMSOL values \n", SCALING);
	float FREQUENCY = (float)A[21];				// frequency in Hz
	printf("FREQUENCY = %f Hz \n", FREQUENCY);
	int C_NUMBER = (int)A[22];				// number of cones
	printf("C_NUMBER = %d \n", C_NUMBER);
	int Loading_Flag = (int)A[23];
	printf("Loading flag = %i \n", Loading_Flag);
	int Random_Flag = (int)A[24];
	printf("Randomization of particle inputs = %i \n", Random_Flag);
	int LONGRANGE = (int)A[25];
	printf("Long Range interactions = %i \n", LONGRANGE);
	int Macro_flag = (int)A[26];
	printf("Macroparticle weighting on/off = %i \n", Macro_flag);
	float density_init = (float)A[27];
	printf("initial density (if macro weighting on) = %i \n", density_init);
	int np_init = (int)A[28];
	printf("np_init = %i \n", np_init);
	int CONSTIN = (int)A[29];
	printf("Constant particle input = %i \n", CONSTIN);
	float CPPS = (float)A[30];
	printf("Computational Particle input Per Second = %e \n", CPPS);
	float SOURCE = (float)A[31];
	printf("Source = %f \n", SOURCE);
	float NP_FACTOR = (float)A[32];
	printf("NP_FACTOR = %f \n", NP_FACTOR);
	int WRITE_FORCECURVES = (int)A[33];
	printf("WRITE_FORCECURVES = %i \n", WRITE_FORCECURVES);
	int WRITE_XYZ = (int)A[34];
	printf("WRITE_XYZ = %i \n", WRITE_XYZ);
	int WRITE_MATLAB = (int)A[35];
	printf("WRITE_MATLAB = %i \n", WRITE_MATLAB);
	int WRITE_VTK = (int)A[36];
	printf("WRITE_VTK = %i \n", WRITE_VTK);
	int nxB = (int)A[37];
	printf("nxB = %i \n", nxB);
	int nyB = (int)A[38];
	printf("nyB = %i \n", nyB);
	int nzB = (int)A[39];
	printf("nzB = %i \n", nzB);
	int METHOD = (int)A[40];
	printf("METHOD = %i \n", METHOD);
	int DT_SCAN = (int)A[41];
	printf("DT_SCAN = %i \n", DT_SCAN);
	float DT_FACTOR_MIN = (float)A[42];
	printf("DT_FACTOR_MIN = %f \n", DT_FACTOR_MIN);
	float DT_FACTOR_MAX = (float)A[43];
	printf("DT_FACTOR_MAX = %f \n", DT_FACTOR_MAX);
	int WRITE_EB = (int)A[44];
	printf("WRITE_EB = %i \n", WRITE_EB);
	int BORNWITHVELOCITY = (int)A[45];
	printf("BORNWITHVELOCITY = %i \n", BORNWITHVELOCITY);
	int RANDOMVELOCITY = (int)A[46];
	printf("RANDOMVELOCITY = %i \n", RANDOMVELOCITY);
	int NUMSCANS = (int)A[47];
	printf("NUMSCANS = %i \n", NUMSCANS);
	int CULL_OUTSIDE = (int)A[48];
	printf("CULL_OUTSIDE = %i \n", CULL_OUTSIDE);
	int pLimit = (int)A[49];
	printf("pLimit = %i \n", pLimit);
	int BRANCH_BUFFER = (int)A[50];
	printf("BRANCH_BUFFER = %i \n", BRANCH_BUFFER);

	static const int print_gots = 1;						// turn this on print a bunch of location identifiers throughout the code, useful for debugging

	int NUMINPUTS = 51;										// number of inputs above (including input zero)

	const float dt = (1 / FREQUENCY) / DT_PER_PERIOD;		// dt = period/dt_per_period
	int ts = (int)floor(END_TIME / dt);

	//ts = 1;
	//np = 1;
	float macrospacingfactor = .35;

	float qom = QE / (ME*REL_MASS);
	float phase;
	float dphase = 2 * pi*dt*FREQUENCY;
	float inv_thresh = 1 / THRESH;

	int n, b, c;	//n is particle index, b is branch index, c is COMSOL cell index
	int bx, by, bz, cx, cy, cz;
	float bxf, byf, bzf;

	float t, next_plot_time;
	int frame, write_frame, timestep, substep;
	int num_frames = (int)floor(END_TIME / WRITE_RATE) + 1;

	int nB_full = nxB*nyB*nzB;

	int nxC = floor(LX / GRIDL + 0.5);
	int nyC = floor(LY / GRIDL + 0.5);
	int nzC = floor(LZ / GRIDL + 0.5);

	int nC_full = nxC*nyC*nzC;
	int nC = nC_full;
	printf("nxC = %i, nyC = %i, nzC = %i, nC = %i, COMSOLROWS = %i\n", nxC, nyC, nzC, nC, COMSOL_ROWS);

	BoxDim boxdim;
	boxdim.xmin = -LX / 2;
	boxdim.xmax = LX / 2;
	boxdim.ymin = -LY / 2;
	boxdim.ymax = LY / 2;
	boxdim.zmin = 0;
	boxdim.zmax = LZ;
	boxdim.xl = boxdim.xmax - boxdim.xmin;
	boxdim.yl = boxdim.ymax - boxdim.ymin;
	boxdim.zl = boxdim.zmax - boxdim.zmin;
	boxdim.vol = boxdim.xl * boxdim.yl * boxdim.zl;

	printf("box is %f -by- %f -by- %f\n", boxdim.xl, boxdim.yl, boxdim.zl);
	printf("box vol = %f\n", boxdim.vol);
	float cavity_volume = pi * (L)* (R1*R1 + R2*R2 + R1*R2) / 3;
	float dxB = boxdim.xl / (nxB - 1);
	float dyB = boxdim.yl / (nyB - 1);
	float dzB = boxdim.zl / (nzB - 1);
	float cell_volume = dxB*dyB*dzB;
	float CPPSperBranch = CPPS*cell_volume/cavity_volume;
	printf("cavity vol = %f\n", boxdim.vol);

	int num_replaced;  //count how many particles we're replacing
	int num_replaced_last;  //used for calculating replacement rate
	float np_create_float;
	float np_create_round;
	float np_remainder;
	int np_create;
	int np_create_since, np_destroy_since;

	float RPM;
	float source_per_sec = SOURCE*cavity_volume;  // Source in #/s
	if (Macro_flag == 1){
		if (CONSTIN == 0){
			RPM = (density_init*cavity_volume) / (np_init); 	// real particles per macroparticle
		}
		else if (CONSTIN == 1){
			RPM = (SOURCE*cavity_volume) / (CPPS);  			// real particles per macroparticle
			//printf("source = %e, cavity vol = %e, CPPS = %e, RPM = %e",SOURCE, cavity_volume, CPPS, RPM);
			//exit(0);
		}
	}
	else if (Macro_flag == 0){
		RPM = 1;
	}
	float qRPM = QE*RPM;

	// -------------------------------- Select only the branches that are within the cone ------------------------------------- //

	int *boxIndexMapper;		// b = boxIndexMapper[bx + nxB*by + nxB*nyB*bz] stores the number index "b" in nB for each index "bx + nxB*by + nxB*nyB*bz" in nB_full  for indices where no "b" is defined boxIndexMapper will store a value of -1
	boxIndexMapper = (int*)malloc(sizeof(int)*nB_full);

	int nB;
	float rbound, rbound2;

	int *include_branch;
	include_branch = (int*)malloc(sizeof(int)*nB_full);

	if (CULL_OUTSIDE == 1){
		printf("Culling boxes inside the cone...\n");
		nB = 0;


		for (b = 0; b < nB_full; b++){
			bx = b % nxB;
			by = (b / nxB) % nyB;
			bz = b / (nxB*nyB);

			// x-y-z position of the center of the branch
			float x = boxdim.xmin + ((float)bx)*dxB;
			float y = boxdim.ymin + ((float)by)*dyB;
			float z = boxdim.zmin + ((float)bz)*dzB;

			// check if any part of the box is in the cone (returns 1 if yes, 0 if no)
			include_branch[b] = BoxConeCollider(x, y, z, dxB, dyB, dzB, R1, R2, L);
			boxIndexMapper[b] = -1; // Initialize to -1, boxes inside the cone will be changed to another value
			if (include_branch[b] == 1){
				boxIndexMapper[b] = nB;
				nB++;
			}
		}
	}
	else{
		nB = nB_full;
		for (b = 0; b < nB; b++){
			boxIndexMapper[b] = b;
		}
	}

	printf("nB = %i, nB_full = %i\n", nB, nB_full);
	int *boxIndexMapperInverse;		// boxIndexMapperInverse stores the value of the index "bx + nxB*by + nxB*nyB*bz" for each box index "b"  in nB
	boxIndexMapperInverse = (int*)malloc(sizeof(int)*nB);

	for (int b = 0; b < nB; b++){
		for (int bb = 0; bb < nB_full; bb++){
			if (boxIndexMapper[bb] == b){
				boxIndexMapperInverse[b] = bb;
			}
		}
	}

	// --------------------------------------------- Find Branch Coordinates -------------------------------------- //
	float *Bxmin, *Bxmax, *Bymin, *Bymax, *Bzmin, *Bzmax;
	Bxmin = (float*)malloc(sizeof(float)*nB);
	Bxmax = (float*)malloc(sizeof(float)*nB);
	Bymin = (float*)malloc(sizeof(float)*nB);
	Bymax = (float*)malloc(sizeof(float)*nB);
	Bzmin = (float*)malloc(sizeof(float)*nB);
	Bzmax = (float*)malloc(sizeof(float)*nB);

	for (int b = 0; b < nB; b++){
		int bb = boxIndexMapperInverse[b];		// get the "bx + nxB*by + nxB*nyB*bz" address for box index b
		bx = bb % nxB;
		by = (bb / nxB) % nyB;
		bz = bb / (nxB*nyB);
		Bxmin[b] = ((float)bx)*dxB + boxdim.xmin;
		Bxmax[b] = ((float)(bx + 1))*dxB + boxdim.xmin;
		Bymin[b] = ((float)by)*dyB + boxdim.ymin;
		Bymax[b] = ((float)(by + 1))*dyB + boxdim.ymin;
		Bzmin[b] = ((float)bz)*dzB + boxdim.zmin;
		Bzmax[b] = ((float)(bz + 1))*dzB + boxdim.zmin;
	}

	// ----------------------------------------------------- Find Branch Neighbors -------------------------------------------------------- //

	int *Bxm, *Bxp, *Bym, *Byp, *Bzm, *Bzp;		// vector arrays, for each box these hold the index of the 6 neighboring boxes ("m" and "p" denote the addresses of the neighbors in the "plus" and "minus" directions
	Bxm = (int*)malloc(sizeof(int)*nB);
	Bxp = (int*)malloc(sizeof(int)*nB);
	Bym = (int*)malloc(sizeof(int)*nB);
	Byp = (int*)malloc(sizeof(int)*nB);
	Bzm = (int*)malloc(sizeof(int)*nB);
	Bzp = (int*)malloc(sizeof(int)*nB);

	for (int b = 0; b < nB; b++){
		int bb = boxIndexMapperInverse[b];		// get the "bx + nxB*by + nxB*nyB*bz" address for box index b
		bx = bb % nxB;
		by = (bb / nxB) % nyB;
		bz = bb / (nxB*nyB);
		int bxm = bx - 1 + by*nxB + bz*nxB*nyB;		// index of the neighbor in the "minus x" direction (for full, non-culled box list)
		int bxp = bx + 1 + by*nxB + bz*nxB*nyB;		// index of the neighbor in the "plus x" direction (for full, non-culled box list)
		int bym = bx + (by - 1)*nxB + bz*nxB*nyB;	// index of the neighbor in the "minus y" direction (for full, non-culled box list)
		int byp = bx + (by + 1)*nxB + bz*nxB*nyB;	// index of the neighbor in the "plus y" direction (for full, non-culled box list)
		int bzm = bx + by*nxB + (bz - 1)*nxB*nyB;	// index of the neighbor in the "minus z" direction (for full, non-culled box list)
		int bzp = bx + by*nxB + (bz + 1)*nxB*nyB;	// index of the neighbor in the "plys z" direction (for full, non-culled box list)

		if (include_branch[bxm] && bx > 0){ Bxm[b] = boxIndexMapper[bxm]; }
		else { Bxm[b] = -1; }
		if (include_branch[bxp] && bx < nxB - 1){ Bxp[b] = boxIndexMapper[bxp]; }
		else { Bxp[b] = -1; }
		if (include_branch[bym] && by > 0){ Bym[b] = boxIndexMapper[bym]; }
		else { Bym[b] = -1; }
		if (include_branch[byp] && by < nyB - 1){ Byp[b] = boxIndexMapper[byp]; }
		else { Byp[b] = -1; }
		if (include_branch[bzm] && bz > 0){ Bzm[b] = boxIndexMapper[bzm]; }
		else { Bzm[b] = -1; }
		if (include_branch[bzp] && bz < nzB - 1){ Bzp[b] = boxIndexMapper[bzp]; }
		else { Bzp[b] = -1; }
	}

	int np_branches = nB*pLimit;
	int np_buffer = nB*BRANCH_BUFFER;
	int num_bytes = np_branches * 6 * sizeof(float);
	int *pnumcheck;
	printf("particle array will be %i entries (up to %i particles per branch) and %i bytes (%i mB, %i gB)\n", np_branches, pLimit, num_bytes, num_bytes / 1024 / 1024, num_bytes / 1024 / 1024 / 1024);

	float macrospacingsq = pow(cavity_volume / np_init, 0.666667);

	// -------- Populate box with random particles -------- //
	float *pox, *poy, *poz, *vox, *voy, *voz;			// position and velocity pointers for "original" particle data
	int *poq;											// charge (1 or -1) for original particle data
	int *oID;											// unique identifier for debugging
	int *pBA;											// box addresses for original particle data

	pox = (float*)malloc(sizeof(float)*np_init);             // allocate memory for position
	poy = (float*)malloc(sizeof(float)*np_init);             // allocate memory for position
	poz = (float*)malloc(sizeof(float)*np_init);             // allocate memory for position
	vox = (float*)malloc(sizeof(float)*np_init);             // allocate memory for velocity
	voy = (float*)malloc(sizeof(float)*np_init);             // allocate memory for velocity
	voz = (float*)malloc(sizeof(float)*np_init);             // allocate memory for velocity
	oID = (int*)malloc(sizeof(int)*np_init);					// allocate memory for IDs

	poq = (int*)malloc(sizeof(int)*np_init);					// allocate memory for charge
	pBA = (int*)malloc(sizeof(int)*np_init);					// allocate memory for branch addresses
	pnumcheck = (int*)calloc(nB, sizeof(int));

	int continue_flag;
	n = 0;
	int num_rejected = 0;
	int maxpnum = 0;

	float distsq;
	float px, py, pz, tx, ty, tz;
	while (n < np_init / 2){
        //printf("n: %i, np_init/2: %d\n", n, np_init / 2);

		// Create random position and velocity for particle n
		pox[n] = (float)rand() / (float)RAND_MAX * (boxdim.xl) + boxdim.xmin;
		poy[n] = (float)rand() / (float)RAND_MAX * (boxdim.yl) + boxdim.ymin;
		poz[n] = (float)rand() / (float)RAND_MAX * (boxdim.zl) + boxdim.zmin;
		//printf("n = %i")
		continue_flag = 0;

		float rbound = poz[n] * (R2 - R1) / L + R1;
		float rbound2 = rbound*rbound;
		if ((pox[n] * pox[n] + poy[n] * poy[n]) > rbound2){
			continue;
		}

		for (int m = 0; m < n; m++){
			distsq = (pox[n] - pox[m])*(pox[n] - pox[m]) + (poy[n] - poy[m])*(poy[n] - poy[m]) + (poz[n] - poz[m])*(poz[n] - poz[m]);
			if (distsq < macrospacingfactor*macrospacingsq){
				//printf(" n = %i, distsq = %f macrospacingsq = %f\n", n, distsq, macrospacingsq);
				continue_flag = 1;
				num_rejected++;
			}
		}
		if (continue_flag == 1){
			continue;
		}

		printf("n = %i success\n",n);

		vox[n] = 0;
		voy[n] = 0;
		voz[n] = 0;

		// Find box address of particle n
		bxf = floor((pox[n] - boxdim.xmin) / dxB + 0.5);
		byf = floor((poy[n] - boxdim.ymin) / dyB + 0.5);
		bzf = floor((poz[n] - boxdim.zmin) / dzB + 0.5);
		bx = (int)bxf;
		by = (int)byf;
		bz = (int)bzf;
		if (bx < 0){ bx = 0; }
		if (by < 0){ by = 0; }
		if (bz < 0){ bz = 0; }
		if (bx > nxB - 1){ bx = nxB - 1; }
		if (by > nyB - 1){ by = nyB - 1; }
		if (bz > nzB - 1){ bz = nzB - 1; }
		b = bx + by*nxB + bz*nxB*nyB; // assign linear address
		int ba = boxIndexMapper[b];		// map address to the address of the culled boxes

        printf("ba is: %i\n", ba);

		float q = 1;

		pBA[n] = ba;
		pnumcheck[ba] += 2;
		if (pnumcheck[ba] > maxpnum){
			maxpnum = pnumcheck[ba];
			if (maxpnum > pLimit){
				printf("\n*ERROR*\npnum[%i] = %i, exceeds pLimit of %i\n*ERROR*\n", ba, maxpnum, pLimit);
			}
		}
		if (n % 500 == 0){
			printf("%i of %i particles created, rejected %i, maxpnum = %i of limit %i\n", n * 2, np_init, num_rejected, maxpnum, pLimit);
		}


		poq[n] = 1;						// positron
		poq[n + np_init / 2] = -1;		// electron

        printf("got to positron assignment...");
		// now assign positron values to the electron
		pox[n + np_init / 2] = pox[n];
		poy[n + np_init / 2] = poy[n];
		poz[n + np_init / 2] = poz[n];
		vox[n + np_init / 2] = 0;
		voy[n + np_init / 2] = 0;
		voz[n + np_init / 2] = 0;
		pBA[n + np_init / 2] = pBA[n];
		oID[n] = n;
		oID[n + np_init / 2] = n + np_init / 2;
        printf("finished positron assignment.\n");
		n++;
	}
    printf("finished positron assignment LOOP.\n\n");

	// --------------------------------------------- Put particles into branches -------------------------------------- //
	float *pbx, *pby, *pbz, *vbx, *vby, *vbz;			// position and velocity pointers for "box-organized" particle data
	int *pbq, *pnum, *bID;											// Number of particles in each branch

	pbx = (float*)malloc(sizeof(float)*np_branches);             // allocate memory for position
	pby = (float*)malloc(sizeof(float)*np_branches);             // allocate memory for position
	pbz = (float*)malloc(sizeof(float)*np_branches);             // allocate memory for position
	vbx = (float*)malloc(sizeof(float)*np_branches);             // allocate memory for velocity
	vby = (float*)malloc(sizeof(float)*np_branches);             // allocate memory for velocity
	vbz = (float*)malloc(sizeof(float)*np_branches);             // allocate memory for velocity
	pbq = (int*)malloc(sizeof(int)*np_branches);				// allocate memory for charge
	bID = (int*)malloc(sizeof(int)*np_branches);				// allocate memory for particle IDs
	pnum = (int*)calloc(sizeof(int), nB);						// initialize pnums to zeros


	maxpnum = 0;
	int pnumindex = -1;
    printf("starting particle branch placement LOOP.\n");
	for (int n = 0; n < np_init; n++){

		int address = pBA[n] + nB*pnum[pBA[n]];		//memory address of particle.  Shifting by branch shifts by one memory address.  Shifting by particle (within a branch) shifts by nB memory addresses

		pbx[address] = pox[n];
		pby[address] = poy[n];
		pbz[address] = poz[n];
		vbx[address] = vox[n];
		vby[address] = voy[n];
		vbz[address] = voz[n];
		pbq[address] = poq[n];
		bID[address] = oID[n];

		pnum[pBA[n]]++;
		if (pnum[pBA[n]] > maxpnum){
			maxpnum = pnum[pBA[n]];
			pnumindex = pBA[n];
		}

	}



	// ------------------------------------------ Create E&B fields -------------------------------------------- //

    printf("Loading E&B Field.\n");
	float **E_realC, **E_imagC, **B_realC, **B_imagC;

	E_realC = (float**)malloc(COMSOL_ROWS * sizeof(float*));
	for (int r = 0; r < COMSOL_ROWS; r++){
		E_realC[r] = (float*)malloc(6 * sizeof(float));
	}
	E_imagC = (float**)malloc(COMSOL_ROWS * sizeof(float*));
	for (int r = 0; r < COMSOL_ROWS; r++){
		E_imagC[r] = (float*)malloc(6 * sizeof(float));
	}
	B_realC = (float**)malloc(COMSOL_ROWS * sizeof(float*));
	for (int r = 0; r < COMSOL_ROWS; r++){
		B_realC[r] = (float*)malloc(6 * sizeof(float));
	}
	B_imagC = (float**)malloc(COMSOL_ROWS * sizeof(float*));
	for (int r = 0; r < COMSOL_ROWS; r++){
		B_imagC[r] = (float*)malloc(6 * sizeof(float));
	}

	Load_EB(COMSOL_ROWS, SCALING, COMSOL_CONV, LX, LY, E_realC, E_imagC, B_realC, B_imagC);
    printf("E&B Field LOADED.\n");

	/*for (int c = 0; c < 200; c++){
		printf("x[%04i] = %1.4f\n",c, E_realC[c][0]);
	}
	exit(0);*/
//
//#define NXC 22
//
//
//	for (int c = 0; c < COMSOL_ROWS; c++){
//		printf("%i: %1.6f\n", c, E_realC[c][2]);
//	}
//
//	float xlist[NXC], xlistsort[NXC];
//	for (int l = 0; l < NXC; l++){
//		xlist[l] = -100;
//	}
//	int ind = 0;
//	for (int c = 0; c < COMSOL_ROWS; c++){
//		int gotit = 0;
//		for (int l = 0; l < NXC; l++){
//			if (E_realC[c][2] == xlist[l]){
//				gotit = 1;
//			}
//		}
//		if (gotit == 0){
//			xlist[ind] = E_realC[c][2];
//			ind++;
//		}
//	}
//	for (int l = 0; l < NXC; l++){
//		printf("%i: %1.6f\n", l, xlist[l]);
//	}
//	//exit(0);
//	printf("\n\n");
//	float currentmin = -100;
//	for (int l = 0; l < NXC; l++){
//		float minl = 100;
//		for (int m = 0; m < NXC; m++){
//			if (xlist[m] < minl && xlist[m] > currentmin){
//				xlistsort[l] = xlist[m];
//				minl = xlist[m];
//			}
//			if (l == 0){ printf("minl = %1.3f, xlist[%i] = %1.3f\n", minl,m,xlist[m]); }
//		}
//		currentmin = xlistsort[l];
//	}
//	for (int l = 0; l < NXC; l++){
//		printf("%i: %1.6f\n", l, xlistsort[l]);
//	}
//	for (int l = 1; l < NXC; l++){
//		printf("dxC = %e, dxB = %e\n", xlistsort[l] - xlistsort[l-1], dxB);
//	}
//	exit(0);


	float *E_realx, *E_imagx, *B_realx, *B_imagx;
	float *E_realy, *E_imagy, *B_realy, *B_imagy;
	float *E_realz, *E_imagz, *B_realz, *B_imagz;

	E_realx = (float*)malloc(sizeof(float)*nB);
	E_imagx = (float*)malloc(sizeof(float)*nB);
	B_realx = (float*)malloc(sizeof(float)*nB);
	B_imagx = (float*)malloc(sizeof(float)*nB);
	E_realy = (float*)malloc(sizeof(float)*nB);
	E_imagy = (float*)malloc(sizeof(float)*nB);
	B_realy = (float*)malloc(sizeof(float)*nB);
	B_imagy = (float*)malloc(sizeof(float)*nB);
	E_realz = (float*)malloc(sizeof(float)*nB);
	E_imagz = (float*)malloc(sizeof(float)*nB);
	B_realz = (float*)malloc(sizeof(float)*nB);
	B_imagz = (float*)malloc(sizeof(float)*nB);

	printf("mapping COMSOL fields to branches...\n");
	for (int b = 0; b < nB; b++){
		bx = boxIndexMapperInverse[b] % nxB;
		by = (boxIndexMapperInverse[b] / nxB) % nyB;
		bz = boxIndexMapperInverse[b] / (nxB*nyB);
		float closest_dist_sq = 1e9;					// initialize to a higher number so that we can find a min
		int closest_c;
		float x, y, z;
		for (int c = 0; c < COMSOL_ROWS; c++){
			x = ((float)bx)*dxB + boxdim.xmin - E_realC[c][0];	// difference in coordinates between the center of the branch and the center of the comsol row
			y = ((float)by)*dyB + boxdim.ymin - E_realC[c][1];
			z = ((float)bz)*dzB + boxdim.zmin - E_realC[c][2];
			float dist_sq = x*x + y*y + z*z;
			//if (b == 100 && (closest_dist_sq == 1e9 || dist_sq < 1)){
			//	printf("\nbx = %1.3f, by = %1.3f, bz = %1.3f  Cx = %1.3f, Cy = %1.3f, Cz = %1.3f\n", x, y, z, E_realC[c][0], E_realC[c][1], E_realC[c][2]);
			//	printf("distsq = %1.5f, closest_dist_sq = %1.5f\n", dist_sq, closest_dist_sq);
			//}
			if (dist_sq < closest_dist_sq){
				closest_dist_sq = dist_sq;
				closest_c = c;
			}
		}
		if (b > -1 && b < 200){
			printf("bx=%02i by=%02i bz=%02i, d=%1.2e\n", bx, by, bz, powf(closest_dist_sq,0.5f));
			//printf("c[%3i] = %i\n", b, closest_c);
		}
		E_realx[b] = E_realC[closest_c][3];
		E_realy[b] = E_realC[closest_c][4];
		E_realz[b] = E_realC[closest_c][5];
		E_imagx[b] = E_imagC[closest_c][3];
		E_imagy[b] = E_imagC[closest_c][4];
		E_imagz[b] = E_imagC[closest_c][5];
		B_realx[b] = B_realC[closest_c][3];
		B_realy[b] = B_realC[closest_c][4];
		B_realz[b] = B_realC[closest_c][5];
		B_imagx[b] = B_imagC[closest_c][3];
		B_imagy[b] = B_imagC[closest_c][4];
		B_imagz[b] = B_imagC[closest_c][5];
	}
	printf("mapping complete...\n");
	printf("dxB = %1.2e, dyB = %1.2e, dzB = %1.2e\n\n",dxB,dyB,dzB);
	//for (int b = 0; b < 100; b++){
	//	printf("E_realC[%4i][5] = %8.1f   E_realz[%4i] = %8.1f\n", b, E_realC[b][5], b, E_realz[b]);
	//}
	//exit(0);

	float xinit, yinit, zinit;


	// ---------------------------------------------------------------------------------------------------------------------------- //
	//														GPU SECTION																//
	// ---------------------------------------------------------------------------------------------------------------------------- //

#if GPUrun

	printf("\n");
	cudaError_t cudaStatus;
	//cudaError_t cudaProfilerInitialize();
	//cudaError_t cudaProfilerStart();
	float *pdx, *pdy, *pdz, *vdx, *vdy, *vdz, *adx, *ady, *adz;		// device arrays
	float *pgx, *pgy, *pgz, *vgx, *vgy, *vgz, *agx, *agy, *agz;		// arrays for storing the resultant GPU vectors (pd, vd) to display data
	//float *pgx_pinned, *pgy_pinned, *pgz_pinned;		// arrays for storing the resultant GPU vectors (pd, vd) to display data
	//int *pgq_pinned;
	int *pdq, *pgq, *gID, *dID, *d_pnum;

	int np_check = 0;
	int num_ele_check = 0;
	int num_pos_check = 0;

	phase = 0;
	next_plot_time = 0;
	t = 0;
	frame = 0, write_frame = 0, timestep = 0, substep = 0;

	num_replaced = 0;  //count how many particles we're replacing
	num_replaced_last = 0;  //used for calculating replacement rate
	np_create_float;
	np_create_round;
	np_remainder = 2;
	np_create;
	np_create_since = 0, np_destroy_since = 0;
	/*for (int n = 0; n<np_branches; n++){
	if (pbq[n] == 1 && num_pos_check < 50){
	printf( "%i, %10.5f%10.5f%10.5f\n",pbq[n], pbx[n], pby[n], pbz[n]);
	num_pos_check++;
	np_check++;
	}
	else if (pbq[n] == -1 && num_ele_check < 50){
	printf("%i, %10.5f%10.5f%10.5f\n", pbq[n], pbx[n], pby[n], pbz[n]);
	num_ele_check++;
	np_check++;
	}
	}*/

	//cudaFuncSetCacheConfig(ParticleMoverGPU, cudaFuncCachePreferL1);    //Set memory preference to L1 (doesn't have much effect)
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int cudaBlocks = deviceProp.multiProcessorCount;

	pgx = (float*)malloc(sizeof(float)*np_branches);
	pgy = (float*)malloc(sizeof(float)*np_branches);
	pgz = (float*)malloc(sizeof(float)*np_branches);
	vgx = (float*)malloc(sizeof(float)*np_branches);
	vgy = (float*)malloc(sizeof(float)*np_branches);
	vgz = (float*)malloc(sizeof(float)*np_branches);
	//agx = (float*)malloc(sizeof(float)*np_branches);
	//agy = (float*)malloc(sizeof(float)*np_branches);
	//agz = (float*)malloc(sizeof(float)*np_branches);
	pgq = (int*)malloc(sizeof(int)*np_branches);
	gID = (int*)malloc(sizeof(int)*np_branches);

	/*cudaMallocHost((void**)&pgx_pinned, sizeof(float)*np_branches);
	cudaMallocHost((void**)&pgy_pinned, sizeof(float)*np_branches);
	cudaMallocHost((void**)&pgz_pinned, sizeof(float)*np_branches);
	cudaMallocHost((void**)&pgq_pinned, sizeof(int)*np_branches);*/

	memcpy(pgx, pbx, sizeof(float)*np_branches);				// copy branch particle info to new vector on CPU to store our GPU values
	memcpy(pgy, pby, sizeof(float)*np_branches);
	memcpy(pgz, pbz, sizeof(float)*np_branches);
	memcpy(vgx, vbx, sizeof(float)*np_branches);
	memcpy(vgy, vby, sizeof(float)*np_branches);
	memcpy(vgz, vbz, sizeof(float)*np_branches);
	memcpy(pgq, pbq, sizeof(int)*np_branches);
	memcpy(gID, bID, sizeof(int)*np_branches);

	int *g_pnum;
	g_pnum = (int*)malloc(sizeof(int)*nB);
	int *gNumTransfer;
	gNumTransfer = (int*)malloc(sizeof(int)*nB);
	int *gNumKill;
	gNumKill = (int*)malloc(sizeof(int)*nB);

	gpuErrchk(cudaMalloc((void**)&pdx, sizeof(float)*np_branches));		// allocate memory for x-position on the GPU
	gpuErrchk(cudaMalloc((void**)&pdy, sizeof(float)*np_branches));		// allocate memory for y-position on the GPU
	gpuErrchk(cudaMalloc((void**)&pdz, sizeof(float)*np_branches));		// allocate memory for z-position on the GPU
	gpuErrchk(cudaMalloc((void**)&vdx, sizeof(float)*np_branches));		// allocate memory for x-velocity on the GPU
	gpuErrchk(cudaMalloc((void**)&vdy, sizeof(float)*np_branches));		// allocate memory for y-velocity on the GPU
	gpuErrchk(cudaMalloc((void**)&vdz, sizeof(float)*np_branches));		// allocate memory for z-velocity on the GPU
	gpuErrchk(cudaMalloc((void**)&adx, sizeof(float)*np_branches));		// allocate memory for x-acceleration on the GPU
	gpuErrchk(cudaMalloc((void**)&ady, sizeof(float)*np_branches));		// allocate memory for y-acceleration on the GPU
	gpuErrchk(cudaMalloc((void**)&adz, sizeof(float)*np_branches));		// allocate memory for z-acceleration on the GPU
	gpuErrchk(cudaMalloc((void**)&dID, sizeof(int)*np_branches));				// allocate memory for number-of-particles on branches on the GPU
	gpuErrchk(cudaMalloc((void**)&pdq, sizeof(int)*np_branches));				// allocate memory for number-of-particles on branches on the GPU
	gpuErrchk(cudaMalloc((void**)&d_pnum, sizeof(int)*nB));				// allocate memory for number-of-particles on branches on the GPU

	gpuErrchk(cudaMemcpy(pdx, pgx, sizeof(float)*np_branches, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(pdy, pgy, sizeof(float)*np_branches, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(pdz, pgz, sizeof(float)*np_branches, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(vdx, vgx, sizeof(float)*np_branches, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(vdy, vgy, sizeof(float)*np_branches, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(vdz, vgz, sizeof(float)*np_branches, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(pdq, pgq, sizeof(int)*np_branches, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dID, gID, sizeof(int)*np_branches, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_pnum, pnum, sizeof(int)*nB, cudaMemcpyHostToDevice));


	float *dE_realx, *dE_imagx, *dB_realx, *dB_imagx;
	float *dE_realy, *dE_imagy, *dB_realy, *dB_imagy;
	float *dE_realz, *dE_imagz, *dB_realz, *dB_imagz;

	gpuErrchk(cudaMalloc((void**)&dE_realx, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dE_imagx, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dB_realx, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dB_imagx, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dE_realy, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dE_imagy, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dB_realy, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dB_imagy, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dE_realz, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dE_imagz, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dB_realz, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dB_imagz, sizeof(float)*nB));

	gpuErrchk(cudaMemcpy(dE_imagx, E_imagx, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dE_realx, E_realx, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dB_realx, B_realx, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dB_imagx, B_imagx, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dE_realy, E_realy, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dE_imagy, E_imagy, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dB_realy, B_realy, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dB_imagy, B_imagy, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dE_realz, E_realz, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dE_imagz, E_imagz, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dB_realz, B_realz, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dB_imagz, B_imagz, sizeof(float)*nB, cudaMemcpyHostToDevice));

	float *pdxBuffer, *pdyBuffer, *pdzBuffer, *vdxBuffer, *vdyBuffer, *vdzBuffer, *adxBuffer, *adyBuffer, *adzBuffer;
	int *pdqBuffer, *dIDBuffer;
	gpuErrchk(cudaMalloc((void**)&pdxBuffer, sizeof(float)*np_buffer));
	gpuErrchk(cudaMalloc((void**)&pdyBuffer, sizeof(float)*np_buffer));
	gpuErrchk(cudaMalloc((void**)&pdzBuffer, sizeof(float)*np_buffer));
	gpuErrchk(cudaMalloc((void**)&vdxBuffer, sizeof(float)*np_buffer));
	gpuErrchk(cudaMalloc((void**)&vdyBuffer, sizeof(float)*np_buffer));
	gpuErrchk(cudaMalloc((void**)&vdzBuffer, sizeof(float)*np_buffer));
	gpuErrchk(cudaMalloc((void**)&adxBuffer, sizeof(float)*np_buffer));
	gpuErrchk(cudaMalloc((void**)&adyBuffer, sizeof(float)*np_buffer));
	gpuErrchk(cudaMalloc((void**)&adzBuffer, sizeof(float)*np_buffer));
	gpuErrchk(cudaMalloc((void**)&pdqBuffer, sizeof(int)*np_buffer));
	gpuErrchk(cudaMalloc((void**)&dIDBuffer, sizeof(int)*np_buffer));

	int *dDestinationsBuffer, *dTransferIndex, *dKillIndex;
	int *dTransferFlag, *dKillFlag;
	int *dNumTransfer, *dNumKill;
	gpuErrchk(cudaMalloc((void**)&dDestinationsBuffer, sizeof(int)*np_buffer));
	gpuErrchk(cudaMalloc((void**)&dTransferIndex, sizeof(int)*np_buffer));
	gpuErrchk(cudaMalloc((void**)&dKillIndex, sizeof(int)*np_buffer));
	gpuErrchk(cudaMalloc((void**)&dTransferFlag, sizeof(int)*np_branches));
	gpuErrchk(cudaMalloc((void**)&dKillFlag, sizeof(int)*np_branches));
	gpuErrchk(cudaMalloc((void**)&dNumTransfer, sizeof(int)*nB));
	gpuErrchk(cudaMalloc((void**)&dNumKill, sizeof(int)*nB));

	float *dBxmin, *dBxmax, *dBymin, *dBymax, *dBzmin, *dBzmax;
	gpuErrchk(cudaMalloc((void**)&dBxmin, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dBxmax, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dBymin, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dBymax, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dBzmin, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dBzmax, sizeof(float)*nB));

	gpuErrchk(cudaMemcpy(dBxmin, Bxmin, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dBxmax, Bxmax, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dBymin, Bymin, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dBymax, Bymax, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dBzmin, Bzmin, sizeof(float)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dBzmax, Bzmax, sizeof(float)*nB, cudaMemcpyHostToDevice));

	int *dBxm, *dBxp, *dBym, *dByp, *dBzm, *dBzp;		// vector arrays, for each box these hold the index of the 6 neighboring boxes ("m" and "p" denote the addresses of the neighbors in the "plus" and "minus" directions
	gpuErrchk(cudaMalloc((void**)&dBxm, sizeof(int)*nB));
	gpuErrchk(cudaMalloc((void**)&dBxp, sizeof(int)*nB));
	gpuErrchk(cudaMalloc((void**)&dBym, sizeof(int)*nB));
	gpuErrchk(cudaMalloc((void**)&dByp, sizeof(int)*nB));
	gpuErrchk(cudaMalloc((void**)&dBzm, sizeof(int)*nB));
	gpuErrchk(cudaMalloc((void**)&dBzp, sizeof(int)*nB));

	gpuErrchk(cudaMemcpy(dBxm, Bxm, sizeof(int)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dBxp, Bxp, sizeof(int)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dBym, Bym, sizeof(int)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dByp, Byp, sizeof(int)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dBzm, Bzm, sizeof(int)*nB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dBzp, Bzp, sizeof(int)*nB, cudaMemcpyHostToDevice));

	float *dpressureX, *dpressureY, *dpressureZ;
	gpuErrchk(cudaMalloc((void**)&dpressureX, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dpressureY, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dpressureZ, sizeof(float)*nB));

	float *dforceX1, *dforceY1, *dforceZ1;
	gpuErrchk(cudaMalloc((void**)&dforceX1, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dforceY1, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dforceZ1, sizeof(float)*nB));

	float *dforceX2, *dforceY2, *dforceZ2;
	gpuErrchk(cudaMalloc((void**)&dforceX2, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dforceY2, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dforceZ2, sizeof(float)*nB));

	float *dforceX3, *dforceY3, *dforceZ3;
	gpuErrchk(cudaMalloc((void**)&dforceX3, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dforceY3, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dforceZ3, sizeof(float)*nB));

	float *dvelX_p, *dvelY_p, *dvelZ_p;
	gpuErrchk(cudaMalloc((void**)&dvelX_p, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dvelY_p, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dvelZ_p, sizeof(float)*nB));

	float *dvelX_e, *dvelY_e, *dvelZ_e;
	gpuErrchk(cudaMalloc((void**)&dvelX_e, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dvelY_e, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&dvelZ_e, sizeof(float)*nB));

	float *daccelX1, *daccelY1, *daccelZ1;
	gpuErrchk(cudaMalloc((void**)&daccelX1, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&daccelY1, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&daccelZ1, sizeof(float)*nB));

	float *daccelX2, *daccelY2, *daccelZ2;
	gpuErrchk(cudaMalloc((void**)&daccelX2, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&daccelY2, sizeof(float)*nB));
	gpuErrchk(cudaMalloc((void**)&daccelZ2, sizeof(float)*nB));

	float *gforceX1, *gforceY1, *gforceZ1;
	gpuErrchk(cudaMallocHost((void**)&gforceX1, sizeof(float)*nB));
	gpuErrchk(cudaMallocHost((void**)&gforceY1, sizeof(float)*nB));
	gpuErrchk(cudaMallocHost((void**)&gforceZ1, sizeof(float)*nB));

	float *gforceX2, *gforceY2, *gforceZ2;
	gpuErrchk(cudaMallocHost((void**)&gforceX2, sizeof(float)*nB));
	gpuErrchk(cudaMallocHost((void**)&gforceY2, sizeof(float)*nB));
	gpuErrchk(cudaMallocHost((void**)&gforceZ2, sizeof(float)*nB));

	float *gforceX3, *gforceY3, *gforceZ3;
	gpuErrchk(cudaMallocHost((void**)&gforceX3, sizeof(float)*nB));
	gpuErrchk(cudaMallocHost((void**)&gforceY3, sizeof(float)*nB));
	gpuErrchk(cudaMallocHost((void**)&gforceZ3, sizeof(float)*nB));

	float *gvelX_p, *gvelY_p, *gvelZ_p;
	gpuErrchk(cudaMallocHost((void**)&gvelX_p, sizeof(float)*nB));
	gpuErrchk(cudaMallocHost((void**)&gvelY_p, sizeof(float)*nB));
	gpuErrchk(cudaMallocHost((void**)&gvelZ_p, sizeof(float)*nB));

	float *gvelX_e, *gvelY_e, *gvelZ_e;
	gpuErrchk(cudaMallocHost((void**)&gvelX_e, sizeof(float)*nB));
	gpuErrchk(cudaMallocHost((void**)&gvelY_e, sizeof(float)*nB));
	gpuErrchk(cudaMallocHost((void**)&gvelZ_e, sizeof(float)*nB));

	n = pBA[0];
	xinit = pgx[n];
	yinit = pgy[n];
	zinit = pgz[n];

	/*gpuErrchk(cudaMemcpy(pgz, pdz, sizeof(float)*np_branches, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(vgz, vdz, sizeof(float)*np_branches, cudaMemcpyDeviceToHost));

	gpuErrchk(cudaMemcpy(pgq, pdq, sizeof(int)*np_branches, cudaMemcpyDeviceToHost));
	for (int n = 0; n < np_branches; n++){
		if (pgq[n] == 1){
			printf("P3: pz[%i] = %f, vz[%i] = %f\n", n, pgz[n], n, vgz[n]);
			break;
		}
	}*/

	printf("\n\nStarting GPU kernel\n");
	n = pBA[0];
	printf("\nBEFORE:printing particle %i\n", n);
	printf("px[%i] = %e, py[%i] = %e, pz[%i] = %e\nvx[%i] = %e\n\n", n, pgx[n] - xinit, n, pgy[n] - yinit, n, pgz[n] - zinit, n, vgx[n]);


	phase = 0;
	next_plot_time = 0;
	t = 0;
	frame = 0; write_frame = 0; timestep = 0; substep = 0;
	int blocks = cudaBlocks*FACTOR;
	int threads = THREADS;

	float GPUtime;
	clock_t GPUstartTotal = clock();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);  cudaEventCreate(&stop);
	float GPUtimer[GPUtimers];
	for (int i = 0; i < GPUtimers; i++) GPUtimer[i] = 0;
	int numKillSum = 0;
	int numTransferSum = 0;

	//int BC = nB / 6;
	//int BCt = Bzp[BC];
	int BC = 828;
	int BCt = 1215;
	printf("nB = %i, BC =%i, BCt = %i\n", nB, BC, BCt);

	/*int *gDestinationsBuffer, *gTransferIndex, *gKillIndex;
	int *gTransferFlag, *gKillFlag;
	int *gNumTransfer, *gNumKill;
	int *pgqBuffer;
	gDestinationsBuffer = (int*)malloc(sizeof(int)*np_buffer);
	gTransferIndex = (int*)malloc(sizeof(int)*np_buffer);
	gKillIndex = (int*)malloc(sizeof(int)*np_buffer);
	gTransferFlag = (int*)malloc(sizeof(int)*np_branches);
	gKillFlag = (int*)malloc(sizeof(int)*np_branches);
	gNumTransfer = (int*)malloc(sizeof(int)*nB);
	gNumKill = (int*)malloc(sizeof(int)*nB);
	pgqBuffer = (int*)malloc(sizeof(int)*np_buffer);
	*/
	//exit(0);
	int pnumNC;

	clock_t clockFrame = clock();
	float timeFrame = 0, timeAvg = 0, timeTotal = 0;

	int snowball = 0, snowballsize = 10;// size of snowball (in # of frames)
	float *pgx_snow, *pgy_snow, *pgz_snow, *pdx_snow, *pdy_snow, *pdz_snow;
	int *pgq_snow, *pdq_snow;
	printf("np_branches = %i\n", np_branches);
	gpuErrchk(cudaMallocHost((void**)&pgx_snow, sizeof(float)*np_branches*snowballsize));
	//gpuErrchk(cudaHostAlloc((void**)&pgx_snow, sizeof(float)*np_branches*snowballsize, 0));
	gpuErrchk(cudaMallocHost((void**)&pgy_snow, sizeof(float)*np_branches*snowballsize));
	gpuErrchk(cudaMallocHost((void**)&pgz_snow, sizeof(float)*np_branches*snowballsize));
	gpuErrchk(cudaMallocHost((void**)&pgq_snow, sizeof(int)*np_branches*snowballsize));

	gpuErrchk(cudaMalloc((void**)&pdx_snow, sizeof(float)*np_branches*snowballsize));
	gpuErrchk(cudaMalloc((void**)&pdy_snow, sizeof(float)*np_branches*snowballsize));
	gpuErrchk(cudaMalloc((void**)&pdz_snow, sizeof(float)*np_branches*snowballsize));
	gpuErrchk(cudaMalloc((void**)&pdq_snow, sizeof(int)*np_branches*snowballsize));

	DataIOinitialize(WRITE_VTK, np_branches);
	curandState *pdx_rand, *pdy_rand, *pdz_rand, *vdx_rand, *vdy_rand, *vdz_rand;
	gpuErrchk(cudaMalloc((void**)&pdx_rand, sizeof(curandState)*nB));
	gpuErrchk(cudaMalloc((void**)&pdy_rand, sizeof(curandState)*nB));
	gpuErrchk(cudaMalloc((void**)&pdz_rand, sizeof(curandState)*nB));
	gpuErrchk(cudaMalloc((void**)&vdx_rand, sizeof(curandState)*nB));
	gpuErrchk(cudaMalloc((void**)&vdy_rand, sizeof(curandState)*nB));
	gpuErrchk(cudaMalloc((void**)&vdz_rand, sizeof(curandState)*nB));

#if AddPartGPU
	float *g_np_remainder, *d_np_remainder;
	g_np_remainder = (float*)malloc(nB*sizeof(float));
	gpuErrchk(cudaMalloc((void**)&d_np_remainder, nB*sizeof(float)));
	for (int b = 0; b < nB; b++) g_np_remainder[b] = 2.0f*(float)rand() / (float)RAND_MAX;
	cudaMemcpy(d_np_remainder, g_np_remainder, nB*sizeof(float), cudaMemcpyHostToDevice);
	AddParticlesGPU_Initialize <<< blocks, threads >>>(pdx_rand, pdy_rand, pdz_rand, vdx_rand, vdy_rand, vdz_rand, nB);
#endif

	float *d_branchVolumes, *branchVolumes;				// volumes of each branch to be used for thrust calculations
	gpuErrchk(cudaMalloc((void**)&d_branchVolumes, sizeof(float)*nB));
	gpuErrchk(cudaMallocHost((void**)&branchVolumes, sizeof(float)*nB));

	float *d_vz_print, *h_vz_print;				// volumes of each branch to be used for thrust calculations
	gpuErrchk(cudaMalloc((void**)&d_vz_print, sizeof(float)*nB));
	gpuErrchk(cudaMallocHost((void**)&h_vz_print, sizeof(float)*nB));

	float *d_rhovac, *h_rhovac;				// volumes of each branch to be used for thrust calculations
	gpuErrchk(cudaMalloc((void**)&d_rhovac, sizeof(float)*nB));
	gpuErrchk(cudaMallocHost((void**)&h_rhovac, sizeof(float)*nB));

	CalculateCellVolumesGPU <<< blocks, threads >>>(d_branchVolumes, dBxmin, dBxmax, dBymin, dBymax, dBzmin, dBzmax, dxB, dyB, dzB, nB, 100, 100, 100, L, R1, R2);
	gpuErrchk(cudaGetLastError()); gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(branchVolumes, d_branchVolumes, sizeof(float)*nB, cudaMemcpyDeviceToHost));
	float subcellvol = dxB*dyB*dzB / 100 / 100 / 100;

	float maxvol = 0;
	for (int b = 0; b < nB; b++){
		printf("vol[%i] = %1.2e  ", b, branchVolumes[b]);
		if (b % 3 == 2) printf("\n");
		if (branchVolumes[b] > maxvol) maxvol = branchVolumes[b];
	}
	int nummax = 0;
	int nummin = 0;
	int numzero = 0;
	float volsum = 0;
	for (int b = 0; b < nB; b++){
		if (fabsf(branchVolumes[b] - maxvol) < subcellvol/2) nummax++;
		if (branchVolumes[b] < subcellvol / 2) numzero++;
		else if (branchVolumes[b] < 3*subcellvol/2) nummin++;

		volsum += branchVolumes[b];
	}
	printf("maxvol = %1.2e, nummax = %i, nummin = %i, numzero = %i\n", maxvol,nummax, nummin, numzero);

	printf("full cell vol = %1.2e, subcell vol = %1.2e\n", dxB*dyB*dzB, dxB*dyB*dzB/100/100/100);
	printf("volsum = %e, article vol = %e\n", volsum, cavity_volume);
	float vz_print = 0, rhovac_print = 0;
	int subflag;
	//exit(0);
	while (t <= END_TIME){

		if (t >= next_plot_time && WRITE_RATE < END_TIME){
			substep = 1;

			if (WRITE_VTK || WRITE_XYZ){
				DataIO(WRITE_VTK, 1, frame, np_buffer, pdx, pdy, pdz, pdq);
			}

			timeFrame = ((float)(clock() - clockFrame)) / CLOCKS_PER_SEC;
			timeTotal += timeFrame;
			if (frame > 0)
				timeAvg = (timeAvg*(frame - 1) + timeFrame) / frame;
			clockFrame = clock();

			printf("G: frame %i of %i, t = %e, CT = %s, ", frame, num_frames, t, time_string(timeFrame));
			printf("AVG = %s\n", time_string(timeAvg));
			//printf("TOT = %s\n", time_string(timeTotal));

			printf("Mover: %s, ", time_string(GPUtimer[0] / ((float)substep)));
			printf("Fill: %s, ", time_string(GPUtimer[1] / ((float)substep)));
			printf("Transfer: %s", time_string(GPUtimer[2] / ((float)substep)));
			printf("Add: %s\n", time_string(GPUtimer[3] / ((float)substep)));


			gpuErrchk(cudaMemcpy(gforceX1, dforceX1, sizeof(float)*nB, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(gforceY1, dforceY1, sizeof(float)*nB, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(gforceZ1, dforceZ1, sizeof(float)*nB, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(gforceX2, dforceX2, sizeof(float)*nB, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(gforceY2, dforceY2, sizeof(float)*nB, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(gforceZ2, dforceZ2, sizeof(float)*nB, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(gforceX3, dforceX3, sizeof(float)*nB, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(gforceY3, dforceY3, sizeof(float)*nB, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(gforceZ3, dforceZ3, sizeof(float)*nB, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(pnum, d_pnum, sizeof(float)*nB, cudaMemcpyDeviceToHost));

			float FX1 = 0;
			float FY1 = 0;
			float FZ1 = 0;
			float FX2 = 0;
			float FY2 = 0;
			float FZ2 = 0;
			float FX3 = 0;
			float FY3 = 0;
			float FZ3 = 0;
			int pnumtot = 0;
			for (int b = 0; b < nB; b++){
				FX1 += gforceX1[b] * branchVolumes[b];
				FY1 += gforceY1[b] * branchVolumes[b];
				FZ1 += gforceZ1[b] * branchVolumes[b];
				FX2 += gforceX2[b] * branchVolumes[b];
				FY2 += gforceY2[b] * branchVolumes[b];
				FZ2 += gforceZ2[b] * branchVolumes[b];
				FX3 += gforceX3[b] * branchVolumes[b];
				FY3 += gforceY3[b] * branchVolumes[b];
				FZ3 += gforceZ3[b] * branchVolumes[b];
				pnumtot += pnum[b];
			}

			WriteForce(1, frame, t, FX1, FY1, FZ1, FX2, FY2, FZ2, FX3, FY3, FZ3);

			//printf("FX = %e\n", FX);
			//printf("FY = %e\n", FY);
			printf("FZ1 = %e\n", FZ1);
			printf("FZ2 = %e\n", FZ2);
			printf("FZ3 = %e\n", FZ3);
			printf("np = %i\n", pnumtot);
			printf("vz_print = %e, rhovac = %e\n", vz_print, rhovac_print);

			frame++;
			next_plot_time += WRITE_RATE;
			//ZeroOutForceGPU <<< blocks, threads >>>(nB, dforceX, dforceY, dforceZ);
			gpuErrchk(cudaGetLastError()); gpuErrchk(cudaDeviceSynchronize());



			substep = 0;
			for (int i = 0; i < GPUtimers; i++) GPUtimer[i] = 0;
			subflag = 1;

		}
		float cosphase = cos(phase);
		float sinphase = sin(phase);

		cudaEventRecord(start, 0);

		ParticleMoverGPU <<< blocks, threads >>>(
			pdx, pdy, pdz,
			vdx, vdy, vdz,
			adx, ady, adz,
			pdq,
			dE_realx,
			dE_imagx,
			dB_realx,
			dB_imagx,
			dE_realy,
			dE_imagy,
			dB_realy,
			dB_imagy,
			dE_realz,
			dE_imagz,
			dB_realz,
			dB_imagz,
			pdxBuffer, pdyBuffer, pdzBuffer,
			vdxBuffer, vdyBuffer, vdzBuffer,
			adxBuffer, adyBuffer, adzBuffer,
			pdqBuffer, dIDBuffer,
			dDestinationsBuffer, dTransferIndex, dKillIndex,
			dTransferFlag, dKillFlag,
			dNumTransfer, dNumKill,
			dBxm, dBxp, dBym, dByp, dBzm, dBzp,
			dBxmin, dBxmax, dBymin, dBymax, dBzmin, dBzmax,
			d_pnum, nB, dt, qom,
			LX, LY,
			cosphase, sinphase,
			L, R1, R2, inv_thresh, qRPM, INTERACTION, timestep, dID);

		gpuErrchk(cudaGetLastError()); gpuErrchk(cudaDeviceSynchronize());
		cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&GPUtime, start, stop);
		GPUtimer[0] += GPUtime*1e-3;

		cudaEventRecord(start);
		FillGapsGPU <<< blocks, threads >>>(
			pdx, pdy, pdz,
			vdx, vdy, vdz,
			adx, ady, adz,
			dBxm, dBxp, dBym, dByp, dBzm, dBzp,
			pdq,
			nB,
			d_pnum,
			dNumTransfer, dNumKill,
			dTransferFlag, dKillFlag,
			dTransferIndex, dKillIndex, timestep, dID
			);
		gpuErrchk(cudaGetLastError()); gpuErrchk(cudaDeviceSynchronize());
		cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&GPUtime, start, stop);
		GPUtimer[1] += GPUtime*1e-3;

		cudaEventRecord(start);
		TransferParticlesGPU <<< blocks, threads >>>(
			pdx, pdy, pdz,
			vdx, vdy, vdz,
			adx, ady, adz,
			pdq,
			pdxBuffer, pdyBuffer, pdzBuffer,
			vdxBuffer, vdyBuffer, vdzBuffer,
			adxBuffer, adyBuffer, adzBuffer,
			pdqBuffer, dIDBuffer,
			nB,
			d_pnum,
			dDestinationsBuffer, dTransferIndex,
			dTransferFlag,
			dNumTransfer, timestep, dID);
		gpuErrchk(cudaGetLastError()); gpuErrchk(cudaDeviceSynchronize());
		cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&GPUtime, start, stop);
		GPUtimer[2] += GPUtime*1e-3;
		//printf("outsideTCkernel1\b");

		cudaEventRecord(start);
		ThrustCalculationGPU1 <<< blocks, threads >>>(
			pdx, pdy, pdz,
			vdx, vdy, vdz,
			adx, ady, adz,
			pdq,
			dE_realx,
			dE_imagx,
			dB_realx,
			dB_imagx,
			dE_realy,
			dE_imagy,
			dB_realy,
			dB_imagy,
			dE_realz,
			dE_imagz,
			dB_realz,
			dB_imagz,
			dpressureX, dpressureY, dpressureZ,
			dvelX_p,
			dvelY_p,
			dvelZ_p,
			dvelX_e,
			dvelY_e,
			dvelZ_e,
			daccelX1, daccelY1, daccelZ1,
			daccelX2, daccelY2, daccelZ2,
			nB, FREQUENCY, cosphase, sinphase,
			d_pnum, ts, dt, d_vz_print, d_rhovac);
		gpuErrchk(cudaGetLastError()); gpuErrchk(cudaDeviceSynchronize());
		cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&GPUtime, start, stop);
		GPUtimer[3] += GPUtime*1e-3;
		gpuErrchk(cudaMemcpy(h_vz_print, d_vz_print, sizeof(float)*nB, cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(h_rhovac, d_rhovac, sizeof(float)*nB, cudaMemcpyDeviceToHost));

		vz_print = 0; rhovac_print = 0;
		for (int b = 0; b < nB; b++){
			vz_print += h_vz_print[b];
			rhovac_print += h_rhovac[b]/((float)nB);
		}
		//printf("substep = %i\n", substep);
		cudaEventRecord(start);
		ThrustCalculationGPU2 <<< blocks, threads >>>(
			dpressureX, dpressureY, dpressureZ,
			daccelX1, daccelY1, daccelZ1,
			daccelX2, daccelY2, daccelZ2,
			dforceX1, dforceY1, dforceZ1,
			dforceX2, dforceY2, dforceZ2,
			dforceX3, dforceY3, dforceZ3,
			d_rhovac,
			dxB, dyB, dzB,
			dBxm, dBxp, dBym, dByp, dBzm, dBzp,
			nB, substep, subflag);
		gpuErrchk(cudaGetLastError()); gpuErrchk(cudaDeviceSynchronize());
		cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&GPUtime, start, stop);
		GPUtimer[4] += GPUtime*1e-3;
		subflag = 0;

		// -- add new particles -- //
#if AddPartGPU
		cudaEventRecord(start);
		AddParticlesGPU <<< blocks, threads >>>(
			pdx, pdy, pdz,
			vdx, vdy, vdz,
			pdq,
			pdx_rand, pdy_rand, pdz_rand,
			vdx_rand, vdy_rand, vdz_rand,
			d_np_remainder,
			nB,
			CPPSperBranch,
			d_pnum, ts, dt,
			boxdim.xl, boxdim.yl, boxdim.zl,
			boxdim.xmin, boxdim.ymin, boxdim.zmin,
			L, R1, R2);

		gpuErrchk(cudaGetLastError()); gpuErrchk(cudaDeviceSynchronize());
		cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&GPUtime, start, stop);
		GPUtimer[3] += GPUtime*1e-3;
#else
		if (CONSTIN){
			np_create_float = np_remainder + CPPS*dt;					// non-integer number of macro-particles created
			np_create_round = 2 * floor(np_create_float / 2);			// round down to even number
			np_remainder = np_create_float - np_create_round;			// calculate the remainder to be added to the next time-step's calculation
			np_create = (int)np_create_round;							// cast as an int for use in calculation
			np_create_since += np_create;
			//printf("timestep = %i, total created = %i\n", timestep, np_create_since);
			//if (np_create > 0)printf("creating %i particles\n", np_create);
			int n = 0;
			while (n < np_create){

				float px = (float)rand() / (float)RAND_MAX * (boxdim.xl) + boxdim.xmin;
				float py = (float)rand() / (float)RAND_MAX * (boxdim.yl) + boxdim.ymin;
				float pz = (float)rand() / (float)RAND_MAX * (boxdim.zl) + boxdim.zmin;
				//printf("n = %i")
				continue_flag = 0;

				float rbound = pz * (R2 - R1) / L + R1;
				float rbound2 = rbound*rbound;
				if (pz > L || pz < 0 || (px * px + py * py) > rbound2){
					continue;
				}

				// Find box address of particle n
				bxf = floor((px - boxdim.xmin) / dxB + 0.5);
				byf = floor((py - boxdim.ymin) / dyB + 0.5);
				bzf = floor((pz - boxdim.zmin) / dzB + 0.5);
				bx = (int)bxf;
				by = (int)byf;
				bz = (int)bzf;
				if (bx < 0){ bx = 0; printf("BX\n"); }
				if (by < 0){ by = 0; printf("BY\n"); }
				if (bz < 0){ bz = 0; printf("BZ\n"); }
				if (bx > nxB - 1){ bx = nxB - 1; printf("BX2\n"); }
				if (by > nyB - 1){ by = nyB - 1; printf("BY2\n"); }
				if (bz > nzB - 1){ bz = nzB - 1; printf("BZ2\n"); }
				int b = boxIndexMapper[bx + by*nxB + bz*nxB*nyB];		// map address to the address of the culled boxes

				int pnumlocal;
				float vx_p = 0;
				float vy_p = 0;
				float vz_p = 0;
				float vx_e = 0;
				float vy_e = 0;
				float vz_e = 0;
				//printf("b = %i, d_pnum = %i, pnumlocal = %i\n", b, d_pnum, pnumlocal);
				//printf("px = %f, py = %f, pz = %f, rbound = %f, rxy = %f\n", px, py, pz, rbound, pow((px * px + py * py), 0.5));

				gpuErrchk(cudaMemcpy(&vx_p, gvelX_p + b, sizeof(float), cudaMemcpyDeviceToHost));
				gpuErrchk(cudaMemcpy(&vy_p, gvelY_p + b, sizeof(float), cudaMemcpyDeviceToHost));
				gpuErrchk(cudaMemcpy(&vz_p, gvelZ_p + b, sizeof(float), cudaMemcpyDeviceToHost));
				gpuErrchk(cudaMemcpy(&vx_e, gvelX_e + b, sizeof(float), cudaMemcpyDeviceToHost));
				gpuErrchk(cudaMemcpy(&vy_e, gvelY_e + b, sizeof(float), cudaMemcpyDeviceToHost));
				gpuErrchk(cudaMemcpy(&vz_e, gvelZ_e + b, sizeof(float), cudaMemcpyDeviceToHost));

				gpuErrchk(cudaMemcpy(&pnumlocal, d_pnum + b, sizeof(int), cudaMemcpyDeviceToHost));
				//printf("pnum_local[%i] = %i\n", b, pnumlocal);
				int na = b + nB*pnumlocal; int pq1 = 1;
				gpuErrchk(cudaMemcpy(pdx + na, &px, sizeof(float), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(pdy + na, &py, sizeof(float), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(pdz + na, &pz, sizeof(float), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(vdx + na, &vx_p, sizeof(float), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(vdy + na, &vy_p, sizeof(float), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(vdz + na, &vz_p, sizeof(float), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(pdq + na, &pq1, sizeof(int), cudaMemcpyHostToDevice));

				int na2 = b + nB*(pnumlocal + 1); int pq2 = -1;
				gpuErrchk(cudaMemcpy(pdx + na2, &px, sizeof(float), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(pdy + na2, &py, sizeof(float), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(pdz + na2, &pz, sizeof(float), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(vdx + na2, &vx_e, sizeof(float), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(vdy + na2, &vy_e, sizeof(float), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(vdz + na2, &vz_e, sizeof(float), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(pdq + na2, &pq2, sizeof(int), cudaMemcpyHostToDevice));

				pnumlocal += 2;

				gpuErrchk(cudaMemcpy(d_pnum + b, &pnumlocal, sizeof(int), cudaMemcpyHostToDevice));

				n += 2;
			}
		}
#endif

		phase += dphase;
		t += dt;
		timestep++;
		substep++;

	}
	cudaDeviceSynchronize();
	float GPUtimeTotal = ((float)(clock() - GPUstartTotal)) / CLOCKS_PER_SEC;
	printf("GPU kernel finished\n");
	//printf("GPUtime = %6.10f ms\n", ((float)GPUtime)*1E3);

	gpuErrchk(cudaMemcpy(pgx, pdx, sizeof(float)*np_branches, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(pgy, pdy, sizeof(float)*np_branches, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(pgz, pdz, sizeof(float)*np_branches, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(vgx, vdx, sizeof(float)*np_branches, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(vgy, vdy, sizeof(float)*np_branches, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(vgz, vdz, sizeof(float)*np_branches, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(pgq, pdq, sizeof(int)*np_branches, cudaMemcpyDeviceToHost));

	n = pBA[0];
	printf("\nAFTER:printing particle %i\n", n);
	//printf("GPU - px[%i] = %f\n", n, pgx[n]);
	int np_final = 0;
	for (int i = 0; i < np_branches; i++){
		if (pgq[i] != 0){
			np_final++;
		}
	}
	printf("np_final = %i!!!!!!!!!!!!!!!!!!!\n", np_final);
	printf("px[%i] = %e, py[%i] = %e, pz[%i] = %e\nvx[%i] = %e\n\n", n, pgx[n] - xinit, n, pgy[n] - yinit, n, pgz[n] - zinit, n, vgx[n]);

	//n = pBA[0];
	//for (int n = 0; n < nB; n++){
	//xinit = 0; yinit = 0; zinit = 0;
	//printf("GA - px[%i] = %f, py[%i] = %f, pz[%i] = %f, vx[%i] = %f\n", n, pgx[n] - xinit, n, pgy[n] - yinit, n, pgz[n] - zinit, n, vgx[n]);
	//}
	gpuErrchk(cudaFree(pdx));
	gpuErrchk(cudaFree(pdy));
	gpuErrchk(cudaFree(pdz));
	gpuErrchk(cudaFree(vdx));
	gpuErrchk(cudaFree(vdy));
	gpuErrchk(cudaFree(vdz));
	gpuErrchk(cudaFree(adx));
	gpuErrchk(cudaFree(ady));
	gpuErrchk(cudaFree(adz));
	gpuErrchk(cudaFree(pdq));
	gpuErrchk(cudaFree(pdxBuffer));
	gpuErrchk(cudaFree(pdyBuffer));
	gpuErrchk(cudaFree(pdzBuffer));
	gpuErrchk(cudaFree(vdxBuffer));
	gpuErrchk(cudaFree(vdyBuffer));
	gpuErrchk(cudaFree(vdzBuffer));
	gpuErrchk(cudaFree(adxBuffer));
	gpuErrchk(cudaFree(adyBuffer));
	gpuErrchk(cudaFree(adzBuffer));
	gpuErrchk(cudaFree(pdqBuffer));
	gpuErrchk(cudaFree(d_pnum));

	gpuErrchk(cudaFree(dDestinationsBuffer));
	gpuErrchk(cudaFree(dTransferIndex));
	gpuErrchk(cudaFree(dKillIndex));
	gpuErrchk(cudaFree(dTransferFlag));
	gpuErrchk(cudaFree(dKillFlag));
	gpuErrchk(cudaFree(dNumTransfer));
	gpuErrchk(cudaFree(dNumKill));

	gpuErrchk(cudaFree(dBxmin));
	gpuErrchk(cudaFree(dBxmax));
	gpuErrchk(cudaFree(dBymin));
	gpuErrchk(cudaFree(dBymax));
	gpuErrchk(cudaFree(dBzmin));
	gpuErrchk(cudaFree(dBzmax));

	gpuErrchk(cudaFree(dBxm));
	gpuErrchk(cudaFree(dBxp));
	gpuErrchk(cudaFree(dBym));
	gpuErrchk(cudaFree(dByp));
	gpuErrchk(cudaFree(dBzm));
	gpuErrchk(cudaFree(dBzp));

	free(pgx);
	free(pgy);
	free(pgz);
	free(vgx);
	free(vgy);
	free(vgz);
	free(pgq);
	free(g_pnum);
	free(gNumTransfer);
	free(gNumKill);


#endif

	// ---------------------------------------------------------------------------------------------------------------------------- //
	//														CPU SECTION																//
	// ---------------------------------------------------------------------------------------------------------------------------- //

#if CPUrun

	float *pcx, *pcy, *pcz, *vcx, *vcy, *vcz;
	int *pcq;
	pcx = (float*)malloc(sizeof(float)*np_branches);
	pcy = (float*)malloc(sizeof(float)*np_branches);
	pcz = (float*)malloc(sizeof(float)*np_branches);
	vcx = (float*)malloc(sizeof(float)*np_branches);
	vcy = (float*)malloc(sizeof(float)*np_branches);
	vcz = (float*)malloc(sizeof(float)*np_branches);
	pcq = (int*)malloc(sizeof(int)*np_branches);

	memcpy(pcx, pbx, sizeof(float)*np_branches);
	memcpy(pcy, pby, sizeof(float)*np_branches);
	memcpy(pcz, pbz, sizeof(float)*np_branches);
	memcpy(vcx, vbx, sizeof(float)*np_branches);
	memcpy(vcy, vby, sizeof(float)*np_branches);
	memcpy(vcz, vbz, sizeof(float)*np_branches);
	memcpy(pcq, pbq, sizeof(int)*np_branches);

	float *pcxBuffer, *pcyBuffer, *pczBuffer, *vcxBuffer, *vcyBuffer, *vczBuffer;
	int *pcqBuffer;
	pcxBuffer = (float*)malloc(sizeof(float)*np_buffer);
	pcyBuffer = (float*)malloc(sizeof(float)*np_buffer);
	pczBuffer = (float*)malloc(sizeof(float)*np_buffer);
	vcxBuffer = (float*)malloc(sizeof(float)*np_buffer);
	vcyBuffer = (float*)malloc(sizeof(float)*np_buffer);
	vczBuffer = (float*)malloc(sizeof(float)*np_buffer);
	pcqBuffer = (int*)malloc(sizeof(int)*np_buffer);

	int *cDestinationsBuffer, *cTransferIndex, *cKillIndex;
	int *cTransferFlag, *cKillFlag;
	int *cNumTransfer, *cNumKill;
	cDestinationsBuffer = (int*)malloc(sizeof(int)*np_buffer);
	cTransferIndex = (int*)malloc(sizeof(int)*np_buffer);
	cKillIndex = (int*)malloc(sizeof(int)*np_buffer);
	cTransferFlag = (int*)malloc(sizeof(int)*np_branches);
	cKillFlag = (int*)malloc(sizeof(int)*np_branches);
	cNumTransfer = (int*)malloc(sizeof(int)*nB);
	cNumKill = (int*)malloc(sizeof(int)*nB);

	n = pBA[0];
	xinit = pcx[n];
	yinit = pcy[n];
	zinit = pcz[n];

	printf("\n\nStarting CPU kernel\n");

	n = pBA[0];
	printf("\BEFORE: printing particle %i\n", n);
	printf("px[%i] = %e, py[%i] = %e, pz[%i] = %e\nvx[%i] = %e\n\n", n, pcx[n] - xinit, n, pcy[n] - yinit, n, pcz[n] - zinit, n, vcx[n]);

	phase = 0;
	next_plot_time = 0;
	t = 0;
	frame = 0, timestep = 0, substep = 0;

	num_replaced = 0;  //count how many particles we're replacing
	num_replaced_last = 0;  //used for calculating replacement rate
	np_create_float;
	np_create_round;
	np_remainder = 2;
	np_create;
	np_create_since = 0, np_destroy_since = 0;

	//for (int a = 0; a < 100; a++){
	//	printf("pcq[%i] = %i\n", a, pcq[a]);
	//}
	int num_killed = 0;
	//exit(0);
	clock_t CPU_start, CPUstartTotal = clock();
	float CPUtimer[CPUtimers];
	for (int i = 0; i < CPUtimers; i++) CPUtimer[i] = 0;

	while (t <= END_TIME){
		if (t >= next_plot_time && WRITE_RATE < END_TIME){
			printf("C: frame %i of %i, t = %e\n", frame, num_frames, t);
			substep = 1;
			printf("Mover: %s, ", time_string(CPUtimer[0] / ((float)substep)));
			printf("Fill: %s, ", time_string(CPUtimer[1] / ((float)substep)));
			printf("Transfer: %s\n", time_string(CPUtimer[2] / ((float)substep)));

			//if (WRITE_VTK)	WriteVTK(0, frame, np_branches, pcx, pcy, pcz, pcq);

			frame++;
			substep = 0;
			for (int i = 0; i < CPUtimers; i++) CPUtimer[i] = 0;
			next_plot_time += WRITE_RATE;
		}
		//printf("---------------------------t = %e\n", t);
		float cosphase = cos(phase);
		float sinphase = sin(phase);

		for (int n = 0; n < np_buffer; n++){
			cDestinationsBuffer[n] = -1;
		}

		CPU_start = clock();
		ParticleMoverCPU(
			pcx, pcy, pcz,
			vcx, vcy, vcz,
			pcq,
			E_realx,
			E_imagx,
			B_realx,
			B_imagx,
			E_realy,
			E_imagy,
			B_realy,
			B_imagy,
			E_realz,
			E_imagz,
			B_realz,
			B_imagz,
			pcxBuffer, pcyBuffer, pczBuffer,
			vcxBuffer, vcyBuffer, vczBuffer,
			pcqBuffer,
			cDestinationsBuffer, cTransferIndex, cKillIndex,
			cTransferFlag, cKillFlag,
			cNumTransfer, cNumKill,
			Bxm, Bxp, Bym, Byp, Bzm, Bzp,
			Bxmin, Bxmax, Bymin, Bymax, Bzmin, Bzmax,
			pnum, nB, dt, qom,
			LX, LY,
			cosphase, sinphase,
			L, R1, R2, inv_thresh, qRPM, INTERACTION);
		CPUtimer[0] += ((float)(clock() - CPU_start)) / CLOCKS_PER_SEC;
		//printf("0: elapsed time = %e, cumulative = %e\n", ((float)(clock() - CPU_start)) / CLOCKS_PER_SEC, CPUtimer[0]);

		CPU_start = clock();
		FillGapsCPU(
			pcx, pcy, pcz,
			vcx, vcy, vcz,
			pcq,
			nB,
			pnum,
			cNumTransfer, cNumKill,
			cTransferFlag, cKillFlag,
			cTransferIndex, cKillIndex
			);
		CPUtimer[1] += ((float)(clock() - CPU_start)) / CLOCKS_PER_SEC;
		//printf("1: elapsed time = %e, cumulative = %e\n", ((float)(clock() - CPU_start)) / CLOCKS_PER_SEC, CPUtimer[1]);

		CPU_start = clock();
		TransferParticlesCPU(
			pcx, pcy, pcz,
			vcx, vcy, vcz,
			pcq,
			pcxBuffer, pcyBuffer, pczBuffer,
			vcxBuffer, vcyBuffer, vczBuffer,
			pcqBuffer,
			nB,
			pnum,
			cDestinationsBuffer, cTransferIndex,
			cTransferFlag,
			cNumTransfer, np_branches);
		CPUtimer[2] += ((float)(clock() - CPU_start)) / CLOCKS_PER_SEC;
		//printf("2: elapsed time = %e, cumulative = %e\n", ((float)(clock() - CPU_start)) / CLOCKS_PER_SEC, CPUtimer[2]);

		// --- Add new particles --- //
		//for (int b = 0; b < nB; b++){

		//}
		np_create_float = np_remainder + CPPS*dt;					// non-integer number of macro-particles created
		np_create_round = 2 * floor(np_create_float / 2);			// round down to even number
		np_remainder = np_create_float - np_create_round;			// calculate the remainder to be added to the next time-step's calculation
		np_create = (int)np_create_round;							// cast as an int for use in calculation
		np_create_since += np_create;

		if (np_create > 0)printf("creating %i particles\n", np_create);
		int n = 0;
		while (n < np_create){
			float px = (float)rand() / (float)RAND_MAX * (boxdim.xl) + boxdim.xmin;
			float py = (float)rand() / (float)RAND_MAX * (boxdim.yl) + boxdim.ymin;
			float pz = (float)rand() / (float)RAND_MAX * (boxdim.zl) + boxdim.zmin;
			continue_flag = 0;

			float rbound = pz * (R2 - R1) / L + R1;
			float rbound2 = rbound*rbound;
			if (pz > L || pz < 0 || (px * px + py * py) > rbound2){
				continue;
			}

			// Find box address of particle n
			bxf = floor((px - boxdim.xmin) / dxB);
			byf = floor((py - boxdim.ymin) / dyB);
			bzf = floor((pz - boxdim.zmin) / dzB);
			bx = (int)bxf;
			by = (int)byf;
			bz = (int)bzf;
			if (bx < 0){ bx = 0; }
			if (by < 0){ by = 0; }
			if (bz < 0){ bz = 0; }
			if (bx > nxB - 1){ bx = nxB - 1; }
			if (by > nyB - 1){ by = nyB - 1; }
			if (bz > nzB - 1){ bz = nzB - 1; }
			int b = boxIndexMapper[bx + by*nxB + bz*nxB*nyB];		// map address to the address of the culled boxes
			int na = b + nB*pnum[b];
			int na2 = b + nB*(pnum[b] + 1);
			//printf("na = %i, na2 = %i\n", na, na2);
			pcx[na] = px;	// create positron
			pcy[na] = py;
			pcz[na] = pz;
			vcx[na] = 0;
			vcy[na] = 0;
			vcz[na] = 0;
			pcq[na] = 1;

			pcx[na2] = px;	// create electron
			pcy[na2] = py;
			pcz[na2] = pz;
			vcx[na2] = 0;
			vcy[na2] = 0;
			vcz[na2] = 0;
			pcq[na2] = -1;

			pnum[b] += 2;
			n += 2;
		}

		/*int np_check = 0;
		for (int b = 0; b < nB; b++){
		np_check += pnum[b];
		}
		maxpnum = 0;
		for (int b = 0; b < nB; b++){
		if (pnum[b] > maxpnum){
		maxpnum = pnum[b];
		}
		num_killed += cNumKill[b];
		}*/

		phase += dphase;
		t += dt;
		timestep++;
		substep++;
	}

	float CPUtimeTotal = ((float)(clock() - CPUstartTotal)) / CLOCKS_PER_SEC;
	printf("CPU kernel finished\n");
	printf("CPUtime = %6.10f ms\n", ((float)CPUtimeTotal)*1E3);

	n = pBA[0];
	printf("\AFTER: printing particle %i\n", n);
	//printf("GPU - px[%i] = %f\n", n, pgx[n]);
	printf("px[%i] = %e, py[%i] = %e, pz[%i] = %e\nvx[%i] = %e\n\n", n, pcx[n] - xinit, n, pcy[n] - yinit, n, pcz[n] - zinit, n, vcx[n]);

	free(pcx);
	free(pcy);
	free(pcz);
	free(vcx);
	free(vcy);
	free(vcz);
	free(pcq);

	free(pcxBuffer);
	free(pcyBuffer);
	free(pczBuffer);
	free(vcxBuffer);
	free(vcyBuffer);
	free(vczBuffer);
	free(pcqBuffer);

	free(cDestinationsBuffer);
	free(cTransferIndex);
	free(cKillIndex);
	free(cTransferFlag);
	free(cKillFlag);
	free(cNumTransfer);
	free(cNumKill);

#endif


	// ---------------------------------------------------------------------------------------------------------------------------- //
	//												CPU Original SECTION															//
	// ---------------------------------------------------------------------------------------------------------------------------- //

#if CPUorig

	phase = 0;
	next_plot_time = 0;
	t = 0;
	frame = 0, timestep = 0, substep = 0;

	num_replaced = 0;  //count how many particles we're replacing
	num_replaced_last = 0;  //used for calculating replacement rate
	np_create_float;
	np_create_round;
	np_remainder = 2;
	np_create;
	np_create_since = 0, np_destroy_since = 0;

	int np_buff = np_orig;

	pax = (float*)malloc(sizeof(float)*np_buff);
	pay = (float*)malloc(sizeof(float)*np_buff);
	paz = (float*)malloc(sizeof(float)*np_buff);
	vax = (float*)malloc(sizeof(float)*np_buff);
	vay = (float*)malloc(sizeof(float)*np_buff);
	vaz = (float*)malloc(sizeof(float)*np_buff);
	paq = (int*)malloc(sizeof(int)*np_buff);

	memcpy(pax, pox, sizeof(float)*np_init);				// copy branch particle info to new vector on CPU to store our GPU values
	memcpy(pay, poy, sizeof(float)*np_init);
	memcpy(paz, poz, sizeof(float)*np_init);
	memcpy(vax, vox, sizeof(float)*np_init);
	memcpy(vay, voy, sizeof(float)*np_init);
	memcpy(vaz, voz, sizeof(float)*np_init);
	memcpy(paq, poq, sizeof(int)*np_init);



	// Time loop
	printf("np before loop starts = %i\n", np);
	while (t < END_TIME) {
		if (t >= next_plot_time && WRITE_RATE < END_TIME){
			printf("C: frame %i of %i, t = %e\n", frame, num_frames, t);
			substep = 1;
			printf("Mover: %s, ", time_string(CPUtimer[0] / ((float)substep)));
			printf("Fill: %s, ", time_string(CPUtimer[1] / ((float)substep)));
			printf("Transfer: %s\n", time_string(CPUtimer[2] / ((float)substep)));

			if (WRITE_VTK)	WriteVTK(0, frame, np_branches, pcx, pcy, pcz, pcq);

			frame++;
			substep = 0;
			for (int i = 0; i < CPUtimers; i++) CPUtimer[i] = 0;
			next_plot_time += WRITE_RATE;
		}
		//printf("---------------------------t = %e\n", t);
		float cosphase = cos(phase);
		float sinphase = sin(phase);


		phase += dphase;


		nf_continuous = phase / (deg_per_comsol*pi / 180);
		nf_floor = floor(nf_continuous);
		nf_remainder = 0;//nf_continuous - nf_floor;
		nf_int = (int)nf_floor;

		//if (s>0){printf("phase = %f, nf_continuous = %f, nf_floor = %f, nf_remainder = %f, nf_int = %i\n",phase,nf_continuous,nf_floor,nf_remainder,nf_int);}
		if (print_gots == 1){ printf("got 2\n"); }

		// Calculate new positions, zero out new velocities and accelerations
		timers.C.MV = clock();

		for (n = 0; n<np; n++){
			prt[n].aEold = prt[n].aE;
			prt[n].aBold = prt[n].aB;
			prt[n].aBpold = prt[n].aBp;
			prt[n].rold = prt[n].r;
			prt[n].rold2 = prt[n].rold;
			prt[n].vold.x = prt[n].v.x;
			prt[n].vold.y = prt[n].v.y;
			prt[n].vold.z = prt[n].v.z;
			if (METHOD == 0 || METHOD == 2){
				prt[n].r.x += prt[n].v.x*dt + 0.5*(prt[n].aE.x + prt[n].aB.x)*dt*dt;
				prt[n].r.y += prt[n].v.y*dt + 0.5*(prt[n].aE.y + prt[n].aB.y)*dt*dt;
				prt[n].r.z += prt[n].v.z*dt + 0.5*(prt[n].aE.z + prt[n].aB.z)*dt*dt;
			}
			else if (METHOD == 1){
				prt[n].r.x += prt[n].v.x*dt;
				prt[n].r.y += prt[n].v.y*dt;
				prt[n].r.z += prt[n].v.z*dt;

			}
			else if (METHOD == 3){
				if (ts>3){
					prt[n].r.x = 2 * prt[n].rold.x - prt[n].rold2.x + (prt[n].aE.x + prt[n].aB.x)*dt*dt;
					prt[n].r.y = 2 * prt[n].rold.y - prt[n].rold2.y + (prt[n].aE.y + prt[n].aB.y)*dt*dt;
					prt[n].r.z = 2 * prt[n].rold.z - prt[n].rold2.z + (prt[n].aE.z + prt[n].aB.z)*dt*dt;
				}
				else{
					prt[n].r.x += prt[n].v.x*dt + 0.5*(prt[n].aE.x + prt[n].aB.x)*dt*dt;
					prt[n].r.y += prt[n].v.y*dt + 0.5*(prt[n].aE.y + prt[n].aB.y)*dt*dt;
					prt[n].r.z += prt[n].v.z*dt + 0.5*(prt[n].aE.z + prt[n].aB.z)*dt*dt;
				}
			}
			// predict velocities for when we want to calculate the electric field
			if (METHOD == 1){
				prt[n].v.x += 0.5*(prt[n].aE.x + prt[n].aB.x)*dt;	// Note: this is not an actual update of the velocity, since it gets overwritten later using integration from "vold".  this is a prediction so that we can better calculate the magnetic force between particles
				prt[n].v.y += 0.5*(prt[n].aE.y + prt[n].aB.y)*dt;
				prt[n].v.z += 0.5*(prt[n].aE.z + prt[n].aB.z)*dt;
			}
			else if (METHOD == 2 || METHOD == 3){
				prt[n].v.x += (prt[n].aE.x + prt[n].aB.x)*dt;
				prt[n].v.y += (prt[n].aE.y + prt[n].aB.y)*dt;
				prt[n].v.z += (prt[n].aE.z + prt[n].aB.z)*dt;
			}
			prt[n].aE.x = 0;
			prt[n].aE.y = 0;
			prt[n].aE.z = 0;
			prt[n].aB.x = 0;
			prt[n].aB.y = 0;
			prt[n].aB.z = 0;
			prt[n].B = (VecR){ 0, 0, 0 };
			prt[n].aB = (VecR){ 0, 0, 0 };
		}



		if (ts % 1 == 0){
			SetupCell(Cell, np, CX_LIMIT, CY_LIMIT, CZ_LIMIT, INITIALIZED, COMSOL_CONV, GRIDL);
		}

		if (print_gots == 1){ printf("got 2.2\n"); }
		// Loop COMSOL for external E+B fields ============================================
		for (cx = 0; cx < CX_LIMIT; cx++){
			for (cy = 0; cy < CY_LIMIT; cy++){
				for (cz = 0; cz < CZ_LIMIT; cz++){


					// Loop through the particles in the cell
					pnum = Cell[cx][cy][cz].pnum;

					/*double cvx_total = 0;
					double cvy_total = 0;
					double cvz_total = 0;*/

					for (ii = 0; ii < pnum; ii++){

						n = Cell[cx][cy][cz].plist[ii];
						// Search for acceleration value to apply
						// Go down list
						i = 1;

						// Look through E and B file data.. scan all rows of dataset to apply fields
						//if (ts % DT_PER_DPHASE == 0) {

						// Quickly assign Efield grid location to particle
						prt[n].ECellX = floor((prt[n].r.x + BOX_LX / 2) / GRIDL + 0.5);// + GRIDL/2) ;
						prt[n].ECellY = floor((prt[n].r.y + BOX_LY / 2) / GRIDL + 0.5);// + GRIDL/2);
						prt[n].ECellZ = floor(prt[n].r.z / GRIDL + 0.5);// + GRIDL/2);
						if (REPLACE == 1)
						{
							if (prt[n].ECellX >= CX_LIMIT){ prt[n].ECellX = CX_LIMIT - 1; }
							if (prt[n].ECellX < 0){ prt[n].ECellX = 0; }
							if (prt[n].ECellY >= CY_LIMIT){ prt[n].ECellY = CY_LIMIT - 1; }
							if (prt[n].ECellY < 0){ prt[n].ECellY = 0; }
							if (prt[n].ECellZ >= CZ_LIMIT){ prt[n].ECellZ = CZ_LIMIT - 1; }
							if (prt[n].ECellZ < 0){ prt[n].ECellZ = 0; }
						}
						else
						{
							/*if (prt[n].ECellX >= CX_LIMIT){
							prt[n].ECellX=CX_LIMIT-1;
							prt[n].EB_addr = COMSOL_ROWS+1;	// particle is no longer in cone
							prt[n].anew.x = 0;
							prt[n].anew.y = 0;
							prt[n].anew.z = 0;
							}
							if (prt[n].ECellX < 0){
							prt[n].ECellX=0;
							prt[n].EB_addr = COMSOL_ROWS+1;	// particle is no longer in cone
							prt[n].anew.x = 0;
							prt[n].anew.y = 0;
							prt[n].anew.z = 0;
							}
							if (prt[n].ECellY >= CY_LIMIT){
							prt[n].ECellY=CY_LIMIT-1;
							prt[n].EB_addr = COMSOL_ROWS+1;	// particle is no longer in cone
							prt[n].anew.x = 0;
							prt[n].anew.y = 0;
							prt[n].anew.z = 0;
							}
							if (prt[n].ECellY < 0){
							prt[n].ECellY=0;
							prt[n].EB_addr = COMSOL_ROWS+1;	// particle is no longer in cone
							prt[n].anew.x = 0;
							prt[n].anew.y = 0;
							prt[n].anew.z = 0;
							}
							if (prt[n].ECellZ >= CZ_LIMIT){
							prt[n].ECellZ=CZ_LIMIT-1;
							prt[n].EB_addr = COMSOL_ROWS+1;	// particle is no longer in cone
							prt[n].anew.x = 0;
							prt[n].anew.y = 0;
							prt[n].anew.z = 0;
							}
							if (prt[n].ECellZ < 0){
							prt[n].ECellZ=0;
							prt[n].EB_addr = COMSOL_ROWS+1;	// particle is no longer in cone
							prt[n].anew.x = 0;
							prt[n].anew.y = 0;
							prt[n].anew.z = 0;
							}*/
						}

						// Find row in Efield file in index matrix
						i = EField_indx[prt[n].ECellX][prt[n].ECellY][prt[n].ECellZ];

						if (i > 0) {
							p_qom = prt[n].q * qom;           // Particle's signed charge-to-mass ratio

							// Calculate Lorentz force on particle n (macroparticle considerations cancel out and so are not included here)
							/*if (nf_int != COMSOL_DIVISIONS-1){
							Ecomsol.x = (1-nf_remainder)*E[i][3][nf_int] + nf_remainder*E[i][3][nf_int+1];
							Ecomsol.y = (1-nf_remainder)*E[i][4][nf_int] + nf_remainder*E[i][4][nf_int+1];
							Ecomsol.z = (1-nf_remainder)*E[i][5][nf_int] + nf_remainder*E[i][5][nf_int+1];
							Bcomsol.x = (1-nf_remainder)*B[i][3][nf_int] + nf_remainder*B[i][3][nf_int+1];
							Bcomsol.y = (1-nf_remainder)*B[i][4][nf_int] + nf_remainder*B[i][4][nf_int+1];
							Bcomsol.z = (1-nf_remainder)*B[i][5][nf_int] + nf_remainder*B[i][5][nf_int+1];
							} else if (nf_int == COMSOL_DIVISIONS-1){
							Ecomsol.x = (1-nf_remainder)*E[i][3][nf_int] + nf_remainder*E[i][3][0];
							Ecomsol.y = (1-nf_remainder)*E[i][4][nf_int] + nf_remainder*E[i][4][0];
							Ecomsol.z = (1-nf_remainder)*E[i][5][nf_int] + nf_remainder*E[i][5][0];
							Bcomsol.x = (1-nf_remainder)*B[i][3][nf_int] + nf_remainder*B[i][3][0];
							Bcomsol.y = (1-nf_remainder)*B[i][4][nf_int] + nf_remainder*B[i][4][0];
							Bcomsol.z = (1-nf_remainder)*B[i][5][nf_int] + nf_remainder*B[i][5][0];
							}
							*/
							Ecomsol.x = E_real[i][3] * cosphase + E_imag[i][3] * sinphase;
							Ecomsol.y = E_real[i][4] * cosphase + E_imag[i][4] * sinphase;
							Ecomsol.z = E_real[i][5] * cosphase + E_imag[i][5] * sinphase;
							Bcomsol.x = B_real[i][3] * cosphase + B_imag[i][3] * sinphase;
							Bcomsol.y = B_real[i][4] * cosphase + B_imag[i][4] * sinphase;
							Bcomsol.z = B_real[i][5] * cosphase + B_imag[i][5] * sinphase;


							//printf("Eold[%i][5] = %e,  New[%i][5] = %e\n",i,Ecomsol.z,i, E_real[i][5]*cosphase + E_imag[i][5]*sinphase);

							prt[n].aE.x = p_qom*Ecomsol.x;
							prt[n].aE.y = p_qom*Ecomsol.y;
							prt[n].aE.z = p_qom*Ecomsol.z;

							prt[n].aB.x = p_qom*(prt[n].v.y*Bcomsol.z - prt[n].v.z*Bcomsol.y);
							prt[n].aB.y = p_qom*(prt[n].v.z*Bcomsol.x - prt[n].v.x*Bcomsol.z);
							prt[n].aB.z = p_qom*(prt[n].v.x*Bcomsol.y - prt[n].v.y*Bcomsol.x);

							prt[n].B.x = p_qom*Bcomsol.x;
							prt[n].B.y = p_qom*Bcomsol.y;
							prt[n].B.z = p_qom*Bcomsol.z;

							//printf("E%i = %10.5e  B%i = %10.5e\n",n,E[i][3][nf],n,B[i][4][nf]);


							//printf("A  prt[%i].anew.x = %10.5e\n",n,prt[n].anew.x);
						}
						//by commenting out the above routine, I need to add back in the else logic so particles outside of the  area of concern are binned in the junk box
						//}
						else if (prt[n].EB_addr != (COMSOL_ROWS + 1)){
							//printf("2\t");
							i = prt[n].EB_addr;
							if (i>COMSOL_ROWS){
								//printf("ZOINK Cell: %i-%i-%i, n: %i, i: %i, EB_addr: %i \n", cx, cy, cz, n, i, prt[n].EB_addr);
								continue;
							}

							p_qom = ((int)(prt[n].q)) * qom;           // Particle's signed charge-to-mass ratio

							/*if (nf_int != COMSOL_DIVISIONS){
							Ecomsol.x = (1-nf_remainder)*E[i][3][nf_int] + nf_remainder*E[i][3][nf_int+1];
							Ecomsol.y = (1-nf_remainder)*E[i][4][nf_int] + nf_remainder*E[i][4][nf_int+1];
							Ecomsol.z = (1-nf_remainder)*E[i][5][nf_int] + nf_remainder*E[i][5][nf_int+1];
							Bcomsol.x = (1-nf_remainder)*B[i][3][nf_int] + nf_remainder*B[i][3][nf_int+1];
							Bcomsol.y = (1-nf_remainder)*B[i][4][nf_int] + nf_remainder*B[i][4][nf_int+1];
							Bcomsol.z = (1-nf_remainder)*B[i][5][nf_int] + nf_remainder*B[i][5][nf_int+1];
							} else if (nf_int == COMSOL_DIVISIONS){
							Ecomsol.x = (1-nf_remainder)*E[i][3][nf_int] + nf_remainder*E[i][3][1];
							Ecomsol.y = (1-nf_remainder)*E[i][4][nf_int] + nf_remainder*E[i][4][1];
							Ecomsol.z = (1-nf_remainder)*E[i][5][nf_int] + nf_remainder*E[i][5][1];
							Bcomsol.x = (1-nf_remainder)*B[i][3][nf_int] + nf_remainder*B[i][3][1];
							Bcomsol.y = (1-nf_remainder)*B[i][4][nf_int] + nf_remainder*B[i][4][1];
							Bcomsol.z = (1-nf_remainder)*B[i][5][nf_int] + nf_remainder*B[i][5][1];
							}*/

							Ecomsol.x = E_real[i][3] * cosphase + E_imag[i][3] * sinphase;
							Ecomsol.y = E_real[i][4] * cosphase + E_imag[i][4] * sinphase;
							Ecomsol.z = E_real[i][5] * cosphase + E_imag[i][5] * sinphase;
							Bcomsol.x = B_real[i][3] * cosphase + B_imag[i][3] * sinphase;
							Bcomsol.y = B_real[i][4] * cosphase + B_imag[i][4] * sinphase;
							Bcomsol.z = B_real[i][5] * cosphase + B_imag[i][5] * sinphase;

							prt[n].aE.x = p_qom*Ecomsol.x;
							prt[n].aE.y = p_qom*Ecomsol.y;
							prt[n].aE.z = p_qom*Ecomsol.z;

							prt[n].aB.x = p_qom*(prt[n].v.y*Bcomsol.z - prt[n].v.z*Bcomsol.y);
							prt[n].aB.y = p_qom*(prt[n].v.z*Bcomsol.x - prt[n].v.x*Bcomsol.z);
							prt[n].aB.z = p_qom*(prt[n].v.x*Bcomsol.y - prt[n].v.y*Bcomsol.x);

							prt[n].B.x = p_qom*Bcomsol.x;
							prt[n].B.y = p_qom*Bcomsol.y;
							prt[n].B.z = p_qom*Bcomsol.z;

						}
						else{

							prt[n].EB_addr = COMSOL_ROWS;
							/*a_new.x = 0;
							a_new.y = 0;
							a_new.z = 0;
							prt[n].a.x = 0;
							prt[n].a.y = 0;
							prt[n].a.z = 0;*/

						}
					}
				}
			}
		}
		if (print_gots == 1){ printf("got 2.2\n"); }
		timers.T.CL += clock() - timers.C.CL;

		// Loop tree to find assign particles to branches
		for (bx = 0; bx < nxB; bx++){
			for (by = 0; by < nyB; by++){
				for (bz = 0; bz < nzB; bz++){
					Branch[bx][by][bz].pnum = 0; // zero out all pnums left over from previous time-step
				}
			}
		}

		if (print_gots == 1){ printf("got 2.3\n"); }
		for (n = 0; n<np; n++){
			rx = prt[n].r.x + LX / 2;
			ry = prt[n].r.y + LY / 2;
			rz = prt[n].r.z;
			bxf = floor((prt[n].r.x + LX / 2) / dxB);
			byf = floor((prt[n].r.y + LY / 2) / dyB);
			bzf = floor((prt[n].r.z) / dzB);
			bx = (int)bxf;
			by = (int)byf;
			bz = (int)bzf;
			if (bx<0){ bx = 0; }
			if (by<0){ by = 0; }
			if (bz<0){ bz = 0; }
			if (bx>nxB - 1){ bx = nxB - 1; }
			if (by>nyB - 1){ by = nyB - 1; }
			if (bz>nzB - 1){ bz = nzB - 1; }
			prt[n].BX = bx;
			prt[n].BY = by;
			prt[n].BZ = bz;
			pnum = Branch[bx][by][bz].pnum;
			Branch[bx][by][bz].plist[pnum] = n;
			Branch[bx][by][bz].pnum++;
		}


		if (print_gots == 1){ printf("got 2.4\n"); }

		if (LONGRANGE == 1 || BORNWITHVELOCITY == 1){
			// Loop tree for COM values
			for (bx = 0; bx < nxB; bx++){
				for (by = 0; by < nxB; by++){
					for (bz = 0; bz < nxB; bz++){

						// Loop through the particles in the cell
						pnum = Branch[bx][by][bz].pnum;
						if (pnum != 0){
							dpnum = (double)pnum;
							crt = (VecR){ 0.0, 0.0, 0.0 };
							cvt = (VecR){ 0.0, 0.0, 0.0 };
							cqt = 0;
							cvt_mag = 0;
							for (ii = 0; ii < pnum; ii++){
								n = Branch[bx][by][bz].plist[ii];
								crt.x += prt[n].r.x;
								crt.y += prt[n].r.y;
								crt.z += prt[n].r.z;
								cvt.x += prt[n].v.x;
								cvt.y += prt[n].v.y;
								cvt.z += prt[n].v.z;
								cvt_mag += pow(prt[n].v.x*prt[n].v.x + prt[n].v.y*prt[n].v.y + prt[n].v.z*prt[n].v.z, 0.5);
								cqt += prt[n].q;
							}
							//printf("n = %3i,  cx = %2i, cy = %2i, cz = %2i, cqt = %i, pnum = %i, crt.x = %f\n",n,cx,cy,cz,cqt,pnum,crt.x);
							//printf("x = %e    y = %e    z = %e\n",crt.x/dpnum, crt.y/dpnum, crt.z/dpnum);
							//printf("      x = %e    y = %e    z = %e pnum = %i, dpnum = %f bx = %i, by = %i, bz = %i,  mxB = %i\n",crt.x, crt.y, crt.z, pnum,dpnum,bx,by,bz,nxB);
							Branch[bx][by][bz].cr = (VecR){ crt.x / pnum, crt.y / pnum, crt.z / pnum };	// Center of mass position
							Branch[bx][by][bz].cv = (VecR){ cvt.x / pnum, cvt.y / pnum, cvt.z / pnum };	// Center of mass velocity
							Branch[bx][by][bz].v_mag = cvt_mag / pnum;
							Branch[bx][by][bz].cq = e*RPM*((double)cqt);							// Center of mass total charge
							//		printf("vmag = %e",Branch[bx][by][bz].v_mag);
						}
						else{
							//printf("ZEROS x = %e    y = %e    z = %e pnum = %i, dpnum = %f bx = %i, by = %i, bz = %i,  mxB = %i\n",crt.x, crt.y, crt.z, pnum,dpnum,bx,by,bz,nxB);
							// if pnum is zero we want to zero out the values in case they were non-zero from the previous time-step
							Branch[bx][by][bz].cr = (VecR){ 0, 0, 0 };	// Center of mass position
							Branch[bx][by][bz].cv = (VecR){ 0, 0, 0 };	// Center of mass velocity
							Branch[bx][by][bz].v_mag = 0;
							Branch[bx][by][bz].cq = 0;
						}
					}
				}
			}
		}
		timers.C.LR = clock();
		if (LONGRANGE == 1){
			if (print_gots == 1){ printf("got 2.5\n"); }

			//printf("n = %e,  v = %f, npinit = %i,  rpm = %e, np = %i,  e = %e,  total charge = %e, dt = %e\n",density_init,test_volume,np_init,RPM,np,e,Branch[0][0][0].cq, dt);

			// Loop Tree for p-p and p-t interactions (particle-particle and particle-tree)
			for (bx = 0; bx < nxB; bx++){
				for (by = 0; by < nxB; by++){
					for (bz = 0; bz < nxB; bz++){
						pnum = Branch[bx][by][bz].pnum;
						for (ii = 0; ii < pnum; ii++){ // Loop through particles in the branch
							n = Branch[bx][by][bz].plist[ii];
							//printf("n=%i\n",n);
							for (bx2 = 0; bx2 < nxB; bx2++){             // Loop through all cells again
								for (by2 = 0; by2 < nyB; by2++){
									for (bz2 = 0; bz2 < nzB; bz2++){
										if (Branch[bx2][by2][bz2].cq != 0){
											if (bx != bx2 || by != by2 || bz != bz2){ // Make sure we are not using the COM data from the cell we are currently in (because that would be double counting!)

												p_qom = (double)(prt[n].q)*qom; 		//particle charge-to-mass ratio "particle q over m" with sign
												branchq = Branch[bx2][by2][bz2].cq;		//Center-of-mass charge (already includes RPM and already in coulombs

												// relative positions
												drx = prt[n].r.x - Branch[bx2][by2][bz2].cr.x;
												dry = prt[n].r.y - Branch[bx2][by2][bz2].cr.y;
												drz = prt[n].r.z - Branch[bx2][by2][bz2].cr.z;
												rij2 = drx*drx + dry*dry + drz*drz;

												rij_s = pow(rij2, 0.5);
												rij_i = drx / rij_s;
												rij_j = dry / rij_s;
												rij_k = drz / rij_s;

												//second calculate relative velocity for Lorentz force
												dvx = prt[n].v.x - Branch[cx2][cy2][cz2].cv.x;
												dvy = prt[n].v.y - Branch[cx2][cy2][cz2].cv.y;
												dvz = prt[n].v.z - Branch[cx2][cy2][cz2].cv.z;

												// Calculate acceleration from electric field
												prt[n].aE.x += (p_qom*branchq*rij_i) / (fourpieps0*rij2);
												prt[n].aE.y += (p_qom*branchq*rij_j) / (fourpieps0*rij2);
												prt[n].aE.z += (p_qom*branchq*rij_k) / (fourpieps0*rij2);

												//Calculate acceleration from magnetic field
												Bjx = (mu0o4pi)*branchq*(dvy*drz - dvz*dry) / rij2;
												Bjy = (mu0o4pi)*branchq*(dvz*drx - dvx*drz) / rij2;
												Bjz = (mu0o4pi)*branchq*(dvx*dry - dvy*drx) / rij2;
												prt[n].aB.x += p_qom*(dvy*Bjz - dvz*Bjy);
												prt[n].aB.y += p_qom*(dvz*Bjx - dvx*Bjz);
												prt[n].aB.z += p_qom*(dvx*Bjy - dvy*Bjx);

												prt[n].aBp.x += p_qom*(dvy*Bjz - dvz*Bjy);
												prt[n].aBp.y += p_qom*(dvz*Bjx - dvx*Bjz);
												prt[n].aBp.z += p_qom*(dvx*Bjy - dvy*Bjx);
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		if (print_gots == 1){ printf("got 2.6\n"); }
		timers.T.LR += clock() - timers.C.LR;

		timers.C.SR = clock();
		if (INTERACTION == 1){
			for (bx = 0; bx < nxB; bx++){
				for (by = 0; by < nxB; by++){
					for (bz = 0; bz < nxB; bz++){
						for (ii = 0; ii < pnum; ii++){ // Loop through particles in the branch
							n = Branch[bx][by][bz].plist[ii];
							pnum = Branch[bx][by][bz].pnum;
							for (jj = 0; jj < pnum; jj++){
								nn = Branch[bx][by][bz].plist[jj];
								if (n != nn){

									N_qom = (float)(prt[n].q)*qom; 				// Particle charge-to-mass ratio "particle q over m" with sign
									NN_qomRPM = (float)(prt[nn].q)*qomRPM;			// Macroparticle weight from second particle does not cancel (since the mass of the second particle doesn't matter)

									//first calculate relative position vector for electrostatic force
									drx = prt[n].r.x - prt[nn].r.x;
									dry = prt[n].r.y - prt[nn].r.y;
									drz = prt[n].r.z - prt[nn].r.z;
									rij2 = drx*drx + dry*dry + drz*drz;

									rij_s = pow(rij2, 0.5);
									if (rij_s < thresh) { continue; }  // too close to infinite interaction, so skip this particle
									rij_i = drx / rij_s;
									rij_j = dry / rij_s;
									rij_k = drz / rij_s;

									//second calculate relative velocity for Lorentz force
									dvx = prt[n].v.x - prt[nn].v.x;
									dvy = prt[n].v.y - prt[nn].v.y;
									dvz = prt[n].v.z - prt[nn].v.z;

									// Calculate acceleration from electric field
									prt[n].aE.x += (N_qom*NN_qomRPM*rij_i) / (fourpieps0*rij2);
									prt[n].aE.y += (N_qom*NN_qomRPM*rij_j) / (fourpieps0*rij2);
									prt[n].aE.z += (N_qom*NN_qomRPM*rij_k) / (fourpieps0*rij2);

									//Calculate acceleration from magnetic field
									Bjx = (mu0o4pi)*NN_qomRPM*(dvy*drz - dvz*dry) / rij2;
									Bjy = (mu0o4pi)*NN_qomRPM*(dvz*drx - dvx*drz) / rij2;
									Bjz = (mu0o4pi)*NN_qomRPM*(dvx*dry - dvy*drx) / rij2;

									prt[n].aB.x += N_qom*(dvy*Bjz - dvz*Bjy);
									prt[n].aB.y += N_qom*(dvz*Bjx - dvx*Bjz);
									prt[n].aB.z += N_qom*(dvx*Bjy - dvy*Bjx);

									prt[n].aBp.x += p_qom*(dvy*Bjz - dvz*Bjy);
									prt[n].aBp.y += p_qom*(dvz*Bjx - dvx*Bjz);
									prt[n].aBp.z += p_qom*(dvx*Bjy - dvy*Bjx);
								}
							}
						}
					}
				}
			}
		}
		timers.T.SR += clock() - timers.C.SR;
		if (print_gots == 1){ printf("got 2.7\n"); }
		timers.C.MV = clock();
		for (n = 0; n<np; n++){
			if (METHOD == 0 || METHOD == 2 || METHOD == 3){
				aE_eff.x = 0.5*(prt[n].aE.x + prt[n].aEold.x);
				aE_eff.y = 0.5*(prt[n].aE.y + prt[n].aEold.y);
				aE_eff.z = 0.5*(prt[n].aE.z + prt[n].aEold.z);
				aB_eff.x = 0.5*(prt[n].aB.x + prt[n].aBold.x);
				aB_eff.y = 0.5*(prt[n].aB.y + prt[n].aBold.y);
				aB_eff.z = 0.5*(prt[n].aB.z + prt[n].aBold.z);
				aBp_eff.x = 0.5*(prt[n].aBp.x + prt[n].aBpold.x);
				aBp_eff.y = 0.5*(prt[n].aBp.y + prt[n].aBpold.y);
				aBp_eff.z = 0.5*(prt[n].aBp.z + prt[n].aBpold.z);
			}
			else if (METHOD == 1){
				aE_eff.x = prt[n].aE.x;
				aE_eff.y = prt[n].aE.y;
				aE_eff.z = prt[n].aE.z;
				aB_eff.x = prt[n].aB.x;
				aB_eff.y = prt[n].aB.y;
				aB_eff.z = prt[n].aB.z;
				aBp_eff.x = prt[n].aBp.x;
				aBp_eff.y = prt[n].aBp.y;
				aBp_eff.z = prt[n].aBp.z;
			}
			vminus.x = prt[n].vold.x + aE_eff.x*0.5*dt;
			vminus.y = prt[n].vold.y + aE_eff.y*0.5*dt;
			vminus.z = prt[n].vold.z + aE_eff.z*0.5*dt;
			tvec.x = prt[n].B.x*0.5*dt;
			tvec.y = prt[n].B.y*0.5*dt;
			tvec.z = prt[n].B.z*0.5*dt;
			tvec_fact = 2 / (1 + tvec.x*tvec.x + tvec.y*tvec.y + tvec.z*tvec.z);
			vprime.x = vminus.x + vminus.y*tvec.z - vminus.z*tvec.y;
			vprime.y = vminus.y + vminus.z*tvec.x - vminus.x*tvec.z;
			vprime.z = vminus.z + vminus.x*tvec.y - vminus.y*tvec.x;
			vplus.x = vminus.x + (vprime.y*tvec.z - vprime.z*tvec.y)*tvec_fact;
			vplus.y = vminus.y + (vprime.z*tvec.x - vprime.x*tvec.z)*tvec_fact;
			vplus.z = vminus.z + (vprime.x*tvec.y - vprime.y*tvec.x)*tvec_fact;
			prt[n].v.x = vplus.x + aE_eff.x*0.5*dt;
			prt[n].v.y = vplus.y + aE_eff.y*0.5*dt;
			prt[n].v.z = vplus.z + aE_eff.z*0.5*dt;
		}
		timers.T.MV += clock() - timers.C.MV;
		if (print_gots == 1){ printf("got 2.8\n"); }
		// Loop COMSOL to get cvx_total and other cell values (for output I think)
		timers.C.VA = clock();
		for (cx = 0; cx < CX_LIMIT; cx++){
			for (cy = 0; cy < CY_LIMIT; cy++){
				for (cz = 0; cz < CZ_LIMIT; cz++){
					// Loop through the particles in the cell
					pnum = Cell[cx][cy][cz].pnum;


					for (ii = 0; ii < pnum; ii++){

						n = Cell[cx][cy][cz].plist[ii];
						//-------------

						if (REPLACE == 1 && CONSTIN != 1)
						{
							// Apply particle reset if particle exits boundary
							x = prt[n].r.x;
							y = prt[n].r.y;
							z = prt[n].r.z;

							int locn;
							int replacef;
							replacef = 0;

							if (n >= num_pos){
								locn = n - num_pos;
							}
							else{
								locn = n;
							}

							if (CYLINDER == 1)
							{// use simple geometry filter
								// Apply particle reset if particle exits boundary
								if (z > 1 || z < -1){ printf("n= %d, z= %f \n", n, z); }
								if (z == 0){ rbound = R1; }
								if (z == L){ rbound = R2; }
								if (z > 0 && z < L){ rbound = z*(R2 - R1) / L + R1; }

								rxy = pow((x*x + y*y), 0.5);
								if (rxy > rbound || z < 0 || z > L)
								{
									replacef = 1;
									//printf("PARTICLE TO BE REPLACED_______________\n");
								}
							}
							else if (CYLINDER == 2)
							{// complex cylinders logic
								// multi-cylinder logic
								double Z_LOW, Z_HIGH, R_LOW, R_HIGH;
								Z_LOW = Z_HIGH = R_LOW = R_HIGH = 0;
								int mm;
								for (mm = 0; mm < C_NUMBER; mm++)
								{
									if (z >= Z[mm][0] && z < Z[mm][1])
									{
										Z_LOW = Z[mm][0];
										Z_HIGH = Z[mm][1];
										R_LOW = R[mm][0];
										R_HIGH = R[mm][1];
									}
									if (z == Z_LOW){ rbound = R_LOW; }
									if (z == Z_HIGH){ rbound = R_HIGH; }
									if (z > Z_LOW && z < Z_HIGH){ rbound = (z - Z_LOW)*(R_HIGH - R_LOW) / (Z_HIGH - Z_LOW) + R_LOW; }
									rxy = pow((x*x + y*y), 0.5);
									if (rxy > rbound || z < 0 || z > BOX_LZ){ replacef = 1; }
								}
							}
							if (replacef == 1) // if the particle needs to be replaced and we are not operating without a constant input source
							{
								prt[n].r.x = prt0[locn].r.x;
								prt[n].r.y = prt0[locn].r.y;
								prt[n].r.z = prt0[locn].r.z;

								prt[n].ECellX = floor((prt0[locn].r.x + BOX_LX / 2) / GRIDL + 0.5);
								prt[n].ECellY = floor((prt0[locn].r.y + BOX_LY / 2) / GRIDL + 0.5);
								prt[n].ECellZ = floor(prt0[locn].r.z / GRIDL + 0.5);
								if (prt[n].ECellX >= CX_LIMIT){ prt[n].ECellX = CX_LIMIT - 1; }
								if (prt[n].ECellX < 0){ prt[n].ECellX = 0; }
								if (prt[n].ECellY >= CY_LIMIT){ prt[n].ECellY = CY_LIMIT - 1; }
								if (prt[n].ECellY < 0){ prt[n].ECellY = 0; }
								if (prt[n].ECellZ >= CZ_LIMIT){ prt[n].ECellZ = CZ_LIMIT - 1; }
								if (prt[n].ECellZ < 0){ prt[n].ECellZ = 0; }

								int r;
								double t;
								r = rand() % 10;
								t = (double)r / 9;
								r = pow(-1.0, r);
								t = t*(double)r; // can use this to also make magnitude random

								prt[n].v.x = Cell[prt[n].ECellX][prt[n].ECellY][prt[n].ECellZ].vx_avg*t;
								prt[n].v.y = Cell[prt[n].ECellX][prt[n].ECellY][prt[n].ECellZ].vy_avg*t;
								prt[n].v.z = Cell[prt[n].ECellX][prt[n].ECellY][prt[n].ECellZ].vz_avg*t;

								/*prt[n].a.x = 0;
								prt[n].a.y = 0;
								prt[n].a.z = 0;*/
								if (n >= num_pos){ prt[n].q = -1; }
								else{ prt[n].q = 1; }
								num_replaced++;
							}

						}
					}


				}
			}
		}

		timers.T.VA += clock() - timers.C.VA;
		if (print_gots == 1){ printf("got 3\n"); }

		if (REPLACE == 1 && CONSTIN == 1){ // delete particles that have exited the domain
			int num_kill = 0;
			int live_particle = np - 1;
			for (n = 0; n<np; n++){
				x = prt[n].r.x;
				y = prt[n].r.y;
				z = prt[n].r.z;
				rbound = z*(R2 - R1) / L + R1;
				rxy = pow((x*x + y*y), 0.5);
				//printf("xyz = %f,%f,%f, rxy = %f, rbound = %f, L = %f\n",x,y,z,rxy,rbound,L);
				if (rxy > rbound || z < 0 || z > L){
					prt[n].flag = 1;	// flag particle as dead
					kill_index[num_kill] = n;
					num_kill++;
					if (prt[n].q == 1){
						num_pos--;
					}
					else if (prt[n].q == -1){
						num_ele--;
					}
					//printf("particle %i must be killed (np=%i)\n",n,np);
				}
			}
			if (num_kill>KILL_MAX){
				printf("ERROR: TRIED TO KILL TOO MANY PARTICLES (%i) IN A SINGLE TIME STEP, KILL_MAX = %i\n", num_kill, KILL_MAX);
				exit(0);
			}

			np -= num_kill; // decrease our total particle count by the number killed (couldn't do this in the loop above since we needed to loop to the full np known at the time)
			//np_destroy_since += num_kill;
			for (ki = 0; ki<num_kill; ki++){ // loop through particles that need to be killed
				n = kill_index[ki];		// index of particle that needs to be killed
				int alive = 0;
				while (alive == 0 && live_particle >= np){ 		// look for particles that are alive beyond np to replace the dead particles with
					if (prt[live_particle].flag == 0){	// if the particle is alive
						//printf("particle %i (D=%i) replaced with particle %i (D=%i)\n",n,prt[n].flag,live_particle,prt[live_particle].flag);
						prt[n] = prt[live_particle];// replace the dead particle with a live particle
						alive = 1;
					}
					live_particle--;
				}
			}
		}



		if (print_gots == 1){ printf("got 4\n"); }

		// ------------- CREATE NEW PARTICLES ------------- //

		if (CONSTIN == 1){
			np_create_float = np_remainder + CPPS*dt;					// non-integer number of macro-particles created
			np_create_round = 2 * floor(np_create_float / 2);				// round down to the nearest even number (since we create pairs)
			np_remainder = np_create_float - np_create_round;			// calculate the remainder to be added to the next time-step's calculation
			np_create = (int)np_create_round;							// cast as an int for use in calculation
			if (np + np_create > np_buffer){										// if we would create more particles than the buffer allows then we need to reduce np_create and print a warning
				np_warning_flag = 1;											// prints a warning later in output to console
				np_create -= (np + np_create) - np_buffer;						// reduce np_create to a number so that np+np_create does not exceed np_buffer
				np_create = (np_create % 2 == 0) ? np_create : np_create - 1;		// round np_create down to the nearest even number so that we only create electron-positron pairs
			}
			np_create_since += np_create;
			n = np;

			while (n<np + np_create){
				continue_flag = 0;
				x = rand() / (double)RAND_MAX*LX - LX / 2;
				y = rand() / (double)RAND_MAX*LY - LY / 2;
				z = rand() / (double)RAND_MAX*LZ;

				rbound = z*(R2 - R1) / L + R1;
				rxy = pow((x*x + y*y), 0.5);
				if (rxy > rbound || z < 0 || z > L){ // reject particle outside of cone
					continue;
				}
				/*for (nn = 0; nn<np + np_create; nn++){ // loop through all other particles to make sure we don't get too close to any other particles
				if (n != nn){
				xdif = x - prt[nn].r.x; ydif = y - prt[nn].r.y; zdif = z - prt[nn].r.z; //difference between particle n and particle nn
				dist = pow((xdif*xdif + ydif*ydif + zdif*zdif), 0.5);
				if (dist < 0.25*macro_spacing){	// if it's too close to another particle we reject it
				continue_flag = 1;  // "continue" for nested loop
				continue;
				}
				}
				}
				if (continue_flag == 1){
				continue;
				}*/

				for (nn = n; nn <= n + 1; nn++){    // create two particles (n and n+1)
					prt[nn].r.x = x;
					prt[nn].r.y = y;
					prt[nn].r.z = z;
					if (nn == n){ prt[n].q = 1; num_pos++; }				// positron
					else if (nn == n + 1){ prt[n + 1].q = -1; num_ele++; }		// electron

					if (BORNWITHVELOCITY == 0){
						prt[nn].v.x = 0;
						prt[nn].v.y = 0;
						prt[nn].v.z = 0;
						prt[nn].vold.x = 0;
						prt[nn].vold.y = 0;
						prt[nn].vold.z = 0;

					}
					else if (BORNWITHVELOCITY == 1){
						prt[nn].BX = floor((x + BOX_LX / 2) / dxB + 0.5);
						prt[nn].BY = floor((y + BOX_LY / 2) / dyB + 0.5);
						prt[nn].BZ = floor(z / dzB + 0.5);
						if (prt[nn].BX >= nxB){ prt[nn].BX = nxB - 1; }
						if (prt[nn].BX < 0){ prt[nn].BX = 0; }
						if (prt[nn].BY >= nyB){ prt[nn].BY = nyB - 1; }
						if (prt[nn].BY < 0){ prt[nn].BY = 0; }
						if (prt[nn].BZ >= nzB){ prt[nn].BZ = nzB - 1; }
						if (prt[nn].BZ < 0){ prt[nn].BZ = 0; }

						if (RANDOMVELOCITY == 0){  // just get velocity from the average velocity of particles in that cell
							vx = Branch[prt[nn].BX][prt[nn].BY][prt[nn].BZ].cv.x;
							vy = Branch[prt[nn].BX][prt[nn].BY][prt[nn].BZ].cv.y;
							vz = Branch[prt[nn].BX][prt[nn].BY][prt[nn].BZ].cv.z;

						}
						else if (RANDOMVELOCITY == 1){
							// The following is from http://mathworld.wolfram.com/SpherePointPicking.html
							double Urand = rand() / (double)RAND_MAX;		// Seed 1
							double Vrand = rand() / (double)RAND_MAX;		// Seed 2
							double Trand = 2 * pi*Urand;					// Theta
							double Prand = acos(2 * Vrand - 1);			// Phi
							double Rrand = 2 * rand() / (double)RAND_MAX*Branch[prt[nn].BX][prt[nn].BY][prt[nn].BZ].v_mag;		// Rho (Random Magnitude between 0 and 1 multiplied by avg velocity magnitude of that cell)

							// Convert from spherical to Cartesian
							vx = Rrand*sin(Prand)*cos(Trand);
							vy = Rrand*sin(Prand)*sin(Trand);
							vz = Rrand*cos(Prand);
							//v_magnitude = pow(vx*vx+vy*vy+vz*vz,0.5);
							//printf(KRED "x[%i] = %f, Rrand1[%i] = %e\n" RESET,nn,prt[nn].r.x,nn,v_magnitude);

						}


						prt[nn].v.x = vx;
						prt[nn].v.y = vy;
						prt[nn].v.z = vz;
						prt[nn].vold.x = vx;
						prt[nn].vold.y = vy;
						prt[nn].vold.z = vz;
						//v_magnitude = pow(vx*vx+vy*vy+vz*vz,0.5);
						//printf(KRED "x[%i] = %f, Rrand2[%i] = %e\n" RESET,nn,prt[nn].r.x,nn,v_magnitude);
					}
				}
				n += 2;
			}
			np += np_create;
		}


		phase += dphase;
		t += dt;
		timestep++;
		substep++;
	}
#endif


#if CPUrun
	printf("CPUtime = %6.10f ms\n", ((float)CPUtimeTotal)*1E3);
#endif
#if GPUrun
	printf("GPUtime = %6.10f s\n", GPUtimeTotal);
	cudaDeviceReset();
#endif
#if CPUrun & GPUrun
	printf("Speedup = %4.8f\n", CPUtimeTotal / GPUtimeTotal);
	cudaDeviceReset();
#endif

}
