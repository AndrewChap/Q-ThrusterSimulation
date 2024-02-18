#include <stdio.h>
#include <stdlib.h>

#define char_amount			100
char forcefilename[char_amount];	// positron data

void WriteForce(int CUDA, int timestep, float t,
	float FX1,
	float FY1,
	float FZ1,
	float FX2,
	float FY2,
	float FZ2,
	float FX3,
	float FY3,
	float FZ3
	){

	static char timename[char_amount];

	if (timestep == 0){
		sprintf(timename, datetime(time(0)));
		if (CUDA == 1){
			sprintf(forcefilename, "ForceCurveGPU");
		}
		else if (CUDA == 0){
			sprintf(forcefilename, "ForceCurveCPU");
		}

		sprintf(forcefilename + strlen(forcefilename), timename);
		sprintf(forcefilename + strlen(forcefilename), ".txt");

		printf("\nCreating force curve file:\n%s\n", forcefilename);
	}
	FILE *ForcesOut;
	/*if (timestep == 0){
		ForcesOut = fopen(forcefilename, "w");
	}
	else{*/
		ForcesOut = fopen(forcefilename, "a");
	//}

		fprintf(ForcesOut, "t: %.5e   Fx1: %.5e   Fy1: %.5e   Fz1: %.5e  Fx2: %.5e   Fy2: %.5e   Fz2: %.5e  Fx3: %.5e   Fy3: %.5e   Fz3: %.5e \n", t, FX1, FY1, FZ1, FX2, FY2, FZ2, FX3, FY3, FZ3);
	fclose(ForcesOut);
}
