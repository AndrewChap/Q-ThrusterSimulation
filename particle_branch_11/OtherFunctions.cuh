#include <stdio.h>
#include <math.h>
#include <time.h>

// function to take a measured time and display it in a string as HH:MM:SS or seconds, milliseconds, microseconds, nanosecond, or picoseconds
const char * time_string(float seconds)
{
	float absseconds = pow(seconds*seconds, 0.5); // absolute value
	static char output[50];
	if (absseconds >= 60){
		int hours = (int)absseconds / 3600;
		int minutes = (int)absseconds / 60 - hours * 60;
		int secondsint = (int)(absseconds)-hours * 3600 - minutes * 60;
		sprintf(output, "%02i:%02i:%02i (HH:MM:SS)", hours, minutes, secondsint);
	}
	else if (absseconds >= 1){
		sprintf(output, "%6.2f s ", seconds);
	}
	else if (absseconds > 1E-3){
		sprintf(output, "%6.2f ms", seconds*1E3);
	}
	else if (absseconds > 1E-6){
		sprintf(output, "%6.2f us", seconds*1E6);
	}
	else if (absseconds > 1E-9){
		sprintf(output, "%6.2f ns", seconds*1E9);
	}
	else{
		sprintf(output, "%6.2f ps", seconds*1E12);
	}

	return output;
}

// function to output the date and time in a format suitable for output files
const char * datetime(time_t currenttime){
	static char output[22];
	//time_t t = time(NULL);
	struct tm tm = *localtime(&currenttime);
	sprintf(output, "%d_%02d_%02d__%02d_%02d_%02d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
	return output;
}

__global__ void ZeroOutForceGPU(int nB, float *forceX, float *forceY, float *forceZ){
	int b = threadIdx.x + blockDim.x * blockIdx.x;		// Branch number
	//printf("transfer launched\n");
	while (b < nB){
		forceX[b] = 0;
		forceY[b] = 0;
		forceZ[b] = 0;
		b += blockDim.x*gridDim.x;
	}
}





#if 0

typedef struct {                /* Cell structure */
	int ID;                     // Cell ID number
	int cnlist[26];             // Cell neighbor list
	int plist[PLIMIT];          // List of particles in cell
	int IN_FLAG;				// is cell in model? Yes = 1, No = 0
	int EDGE_FLAG;				// is cell on edge of model? Yes = 1, No = 0
	int CORNERS[8];				// align the eight corners with rows in the COMSOL file for force calculation
	int FCORNERS[8];			// corner flags
	int pnum;                   // Number of particles in cell
	float xmax;				// particle depth in x
	float ymax;				// particle depth in y
	float zmax;				// particle depth in z
	float vx_avg;				// average x velocity
	float vy_avg;				// average y velocity
	float vz_avg;				// average z velocity
	float Ex;					// V/m
	float Ey;
	float Ez;
	float E2;
	float RHO_E;				// N/m^2
	float Bx;					// Tesla
	float By;
	float Bz;
	float B2;
	float RHO_B;				// N/m^2
	float RHO_VAC;				// kg/m^3
	float volume;				// m^3
	float a1, b1, c1, d1;		// clipping plane 1
	float a2, b2, c2, d2;		// clipping plane 2
	float vertices[8][3];
	VecR cr;					// center of mass position
	VecR cv;					// center of mass velocity
	float cq;					// center of mass charge
	float den;					// number density
	int dnum;
} CellStruct;


void SetupCell(CellStruct ***lCell, int lnp, int lcx, int lcy, int lcz, int flag, float lCONV, float lGRIDL)
{
	float rx, ry, rz, xv, yv, zv;
	float x_left, x_right, y_down, y_up, z_back, z_front, x, y, z;
	//!!float vertices[8][3];
	int cx, cy, cz;
	int pnum, n, i, j;
	int i10, i20, i30, i40, i50, i60, i70, i80, i90;
	i10 = i20 = i30 = i40 = i50 = i60 = i70 = i80 = i90 = 0;

	// First zero cell items
	for (cx = 0; cx < lcx; cx++) {
		for (cy = 0; cy < lcy; cy++) {
			for (cz = 0; cz < lcz; cz++) {
				lCell[cx][cy][cz].pnum = 0;
				lCell[cx][cy][cz].cr = (VecR){ 0, 0, 0 };
				lCell[cx][cy][cz].cv = (VecR){ 0, 0, 0 };
				lCell[cx][cy][cz].cq = 0;
				if (flag == 0) {
					lCell[cx][cy][cz].IN_FLAG = 0;
					lCell[cx][cy][cz].EDGE_FLAG = 0;
					lCell[cx][cy][cz].volume = 0;
					lCell[cx][cy][cz].xmax = 0;				// particle depth in x
					lCell[cx][cy][cz].ymax = 0;				// particle depth in y
					lCell[cx][cy][cz].zmax = 0;				// particle depth in z
					lCell[cx][cy][cz].vx_avg = 0;			// average x velocity
					lCell[cx][cy][cz].vy_avg = 0;			// average y velocity
					lCell[cx][cy][cz].vz_avg = 0;			// average z velocity
					lCell[cx][cy][cz].Ex = 0;				// V/m
					lCell[cx][cy][cz].Ey = 0;
					lCell[cx][cy][cz].Ez = 0;
					lCell[cx][cy][cz].E2 = 0;
					lCell[cx][cy][cz].RHO_E = 0;			// N/m^2
					lCell[cx][cy][cz].Bx = 0;				// Tesla
					lCell[cx][cy][cz].By = 0;
					lCell[cx][cy][cz].Bz = 0;
					lCell[cx][cy][cz].B2 = 0;
					lCell[cx][cy][cz].RHO_B = 0;			// N/m^2
					lCell[cx][cy][cz].RHO_VAC = 0;			// kg/m^3
					lCell[cx][cy][cz].volume = 0;			// m^3					
					for (j = 0; j<8; j++) {
						lCell[cx][cy][cz].CORNERS[j] = -1;
						lCell[cx][cy][cz].FCORNERS[j] = 0;
					}
				}
			}
		}
	}

	if (flag == 0){
		// determine intersection of vertices with COMSOL rows	
		for (cx = 0; cx < lcx; cx++) {
			for (cy = 0; cy < lcy; cy++) {
				for (cz = 0; cz < lcz; cz++) {

					//printf ("herehere!\n");
					x_left = -LX / 2 + lGRIDL*(cx);
					x_right = -LX / 2 + lGRIDL*(cx + 1);

					//printf("XL = %f, XR = %f\n", (x_left*COMSOL_CONV), (x_right*COMSOL_CONV));

					y_down = -LY / 2 + lGRIDL*(cy);
					y_up = -LY / 2 + lGRIDL*(cy + 1);

					z_back = lGRIDL*(cz);
					z_front = lGRIDL*(cz + 1);

					lCell[cx][cy][cz].vertices[0][0] = x_left;
					lCell[cx][cy][cz].vertices[0][1] = y_down;
					lCell[cx][cy][cz].vertices[0][2] = z_back;

					lCell[cx][cy][cz].vertices[1][0] = x_right;
					lCell[cx][cy][cz].vertices[1][1] = y_down;
					lCell[cx][cy][cz].vertices[1][2] = z_back;

					lCell[cx][cy][cz].vertices[2][0] = x_left;
					lCell[cx][cy][cz].vertices[2][1] = y_up;
					lCell[cx][cy][cz].vertices[2][2] = z_back;

					lCell[cx][cy][cz].vertices[3][0] = x_right;
					lCell[cx][cy][cz].vertices[3][1] = y_up;
					lCell[cx][cy][cz].vertices[3][2] = z_back;

					lCell[cx][cy][cz].vertices[4][0] = x_left;
					lCell[cx][cy][cz].vertices[4][1] = y_down;
					lCell[cx][cy][cz].vertices[4][2] = z_front;

					lCell[cx][cy][cz].vertices[5][0] = x_right;
					lCell[cx][cy][cz].vertices[5][1] = y_down;
					lCell[cx][cy][cz].vertices[5][2] = z_front;

					lCell[cx][cy][cz].vertices[6][0] = x_left;
					lCell[cx][cy][cz].vertices[6][1] = y_up;
					lCell[cx][cy][cz].vertices[6][2] = z_front;

					lCell[cx][cy][cz].vertices[7][0] = x_right;
					lCell[cx][cy][cz].vertices[7][1] = y_up;
					lCell[cx][cy][cz].vertices[7][2] = z_front;

					for (j = 0; j<8; j++) {
						x = lCell[cx][cy][cz].vertices[j][0];
						y = lCell[cx][cy][cz].vertices[j][1];
						z = lCell[cx][cy][cz].vertices[j][2];
						//printf("x = %f, y = %f, z = %f\n", (x*COMSOL_CONV), (y*COMSOL_CONV), (z*COMSOL_CONV));
						i = 0;
						while (i < nr) {
							xv = E_real[i][0] / lCONV;		// X coordinate
							yv = E_real[i][1] / lCONV;	    // Y coordinate
							zv = E_real[i][2] / lCONV;		// Z coordinate
							//printf("x = %f, y = %f, z = %f\n", (x*COMSOL_CONV), (y*COMSOL_CONV), (z*COMSOL_CONV));
							//printf("xv = %f, yv = %f, zv = %f\n", (xv*COMSOL_CONV), (yv*COMSOL_CONV), (zv*COMSOL_CONV));
							if (z >= (zv - 0.4*lGRIDL) && z <= (zv + 0.4*lGRIDL)){
								if (y >= (yv - 0.4*lGRIDL) && y <= (yv + 0.4*lGRIDL)){
									if (x >= (xv - 0.4*lGRIDL) && x <= (xv + 0.4*lGRIDL)){
										// Assign EB address to vertex
										lCell[cx][cy][cz].CORNERS[j] = i;
										lCell[cx][cy][cz].FCORNERS[j] = 1;
										lCell[cx][cy][cz].IN_FLAG = 1;
										i = nr;
										//printf ("here!\n");										
									}
								}
							}
							i++;
						}
					}
				}
			}
			if (cx * 100 / lcx > 10 && cx * 100 / lcx < 20 && i10 == 0){ printf("10%% cells initialized \n"); i10 = 1; }
			if (cx * 100 / lcx > 20 && cx * 100 / lcx < 30 && i20 == 0){ printf("20%% cells initialized \n"); i20 = 1; }
			if (cx * 100 / lcx > 30 && cx * 100 / lcx < 40 && i30 == 0){ printf("30%% cells initialized \n"); i30 = 1; }
			if (cx * 100 / lcx > 40 && cx * 100 / lcx < 50 && i40 == 0){ printf("40%% cells initialized \n"); i40 = 1; }
			if (cx * 100 / lcx > 50 && cx * 100 / lcx < 60 && i50 == 0){ printf("50%% cells initialized \n"); i50 = 1; }
			if (cx * 100 / lcx > 60 && cx * 100 / lcx < 70 && i60 == 0){ printf("60%% cells initialized \n"); i60 = 1; }
			if (cx * 100 / lcx > 70 && cx * 100 / lcx < 80 && i70 == 0){ printf("70%% cells initialized \n"); i70 = 1; }
			if (cx * 100 / lcx > 80 && cx * 100 / lcx < 90 && i80 == 0){ printf("80%% cells initialized \n"); i80 = 1; }
			if (cx * 100 / lcx > 90 && i90 == 0){ printf("90%% cells initialized \n"); i90 = 1; }
		}

		i10 = i20 = i30 = i40 = i50 = i60 = i70 = i80 = i90 = 0;

		// edge detection
		for (cx = 0; cx < lcx; cx++) {
			for (cy = 0; cy < lcy; cy++) {
				for (cz = 0; cz < lcz; cz++) {
					if (lCell[cx][cy][cz].IN_FLAG == 1){
						// Check cubes in body of simulation space
						if (cx != 0 && cy != 0 && cz != 0 && cx != (lcx - 1) && cy != (lcy - 1) && cz != (lcz - 1)){
							if (lCell[cx - 1][cy][cz].IN_FLAG == 0 || lCell[cx + 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy - 1][cz].IN_FLAG == 0 || lCell[cx][cy + 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz - 1].IN_FLAG == 0 || lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}

						//(technically all of the following checks are by definition edge pieces, but may need this code for the gradient algorithm, so can cut and paste 
						// Check cubes on edge faces of simulation space 
						//LEFT FACE
						if (cx == 0 && cy != 0 && cz != 0 && cy != (lcy - 1) && cz != (lcz - 1)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx + 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy - 1][cz].IN_FLAG == 0 || lCell[cx][cy + 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz - 1].IN_FLAG == 0 || lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						//RIGHT FACE
						if (cx == (lcx - 1) && cy != 0 && cz != 0 && cy != (lcy - 1) && cz != (lcz - 1)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx - 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy - 1][cz].IN_FLAG == 0 || lCell[cx][cy + 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz - 1].IN_FLAG == 0 || lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						//BOTTOM FACE
						if (cx != 0 && cx != (lcx - 1) && cy == 0 && cz != 0 && cz != (lcz - 1)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx - 1][cy][cz].IN_FLAG == 0 || lCell[cx + 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy + 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz - 1].IN_FLAG == 0 || lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						//TOP FACE
						if (cx != 0 && cx != (lcx - 1) && cy == (lcy - 1) && cz != 0 && cz != (lcz - 1)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx - 1][cy][cz].IN_FLAG == 0 || lCell[cx + 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy - 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz - 1].IN_FLAG == 0 || lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						//BACK FACE
						if (cx != 0 && cx != (lcx - 1) && cy != 0 && cy != (lcy - 1) && cz == 0){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx - 1][cy][cz].IN_FLAG == 0 || lCell[cx + 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy - 1][cz].IN_FLAG == 0 || lCell[cx][cy + 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						//FRONT FACE
						if (cx != 0 && cx != (lcx - 1) && cy != 0 && cy != (lcy - 1) && cz == (lcz - 1)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx - 1][cy][cz].IN_FLAG == 0 || lCell[cx + 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy - 1][cz].IN_FLAG == 0 || lCell[cx][cy + 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz - 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}


						// Check cubes on ridges of simulation space (12 ridges)
						//Z-AXIS
						if (cx == 0 && cy == 0 && cz != 0 && cz != (lcz - 1)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx + 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy + 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz - 1].IN_FLAG == 0 || lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						//Y-AXIS
						if (cx == 0 && cy != 0 && cy != (lcy - 1) && cz == 0){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx + 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy - 1][cz].IN_FLAG == 0 || lCell[cx][cy + 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						//X-AXIS
						if (cx != 0 && cx != (lcx - 1) && cy == 0 && cz == 0){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx - 1][cy][cz].IN_FLAG == 0 || lCell[cx + 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy + 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// ||Z-AXIS top right edge	
						if (cx == (lcx - 1) && cy == (lcy - 1) && cz != 0 && cz != (lcz - 1)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx - 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy - 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz - 1].IN_FLAG == 0 || lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// ||Y-AXIS front right edge
						if (cx == (lcx - 1) && cy != 0 && cy != (lcy - 1) && cz == (lcz - 1)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx - 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy - 1][cz].IN_FLAG == 0 || lCell[cx][cy + 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz - 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// ||X-AXIS top front edge
						if (cx != 0 && cx != (lcx - 1) && cy == (lcy - 1) && cz == (lcz - 1)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx - 1][cy][cz].IN_FLAG == 0 || lCell[cx + 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy - 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz - 1].IN_FLAG == 0) {
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// || Z-AXIS top left edge
						if (cx == (0) && cy == (lcy - 1) && cz != 0 && cz != (lcz - 1)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx + 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy - 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz - 1].IN_FLAG == 0 || lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// || Y-AXIS front left edge
						if (cx == (0) && cy != 0 && cy != (lcy - 1) && cz == (lcz - 1)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx + 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy - 1][cz].IN_FLAG == 0 || lCell[cx][cy + 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz - 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// || X-AXIS bottom front edge
						if (cx != 0 && cx != (lcx - 1) && cy == (0) && cz == (lcz - 1)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx - 1][cy][cz].IN_FLAG == 0 || lCell[cx + 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy + 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz - 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// || Z-AXIS bottom right edge
						if (cx == (lcx - 1) && cy == (0) && cz != 0 && cz != (lcz - 1)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx - 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy + 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz - 1].IN_FLAG == 0 || lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// || Y-AXIS back right edge
						if (cx == (lcx - 1) && cy != 0 && cy != (lcy - 1) && cz == (0)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx - 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy - 1][cz].IN_FLAG == 0 || lCell[cx][cy + 1][cz].IN_FLAG == 0
								|| lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// || X-AXIS back top edge
						if (cx != 0 && cx != (lcx - 1) && cy == (lcy - 1) && cz == (0)){
							if (lCell[cx][cy][cz].IN_FLAG == 1 ||
								lCell[cx - 1][cy][cz].IN_FLAG == 0 || lCell[cx + 1][cy][cz].IN_FLAG == 0 ||
								lCell[cx][cy - 1][cz].IN_FLAG == 0 ||
								lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}

						//Check the eight corners
						// Corner 0, origin or left bottom back vertex
						if (cx == 0 && cy == 0 && cz == 0){
							if (lCell[cx][cy][cz].IN_FLAG == 1 || lCell[cx + 1][cy][cz].IN_FLAG == 0 || lCell[cx][cy + 1][cz].IN_FLAG == 0 || lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// Corner 1, right bottom back vertex
						if (cx == (lcx - 1) && cy == (0) && cz == 0) {
							if (lCell[cx][cy][cz].IN_FLAG == 1 || lCell[cx - 1][cy][cz].IN_FLAG == 0 || lCell[cx][cy + 1][cz].IN_FLAG == 0 || lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// Corner 2, left top back vertex
						if (cx == 0 && cy == (lcy - 1) && cz == 0){
							if (lCell[cx][cy][cz].IN_FLAG == 1 || lCell[cx + 1][cy][cz].IN_FLAG == 0 || lCell[cx][cy - 1][cz].IN_FLAG == 0 || lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// Corner 3, right top back vertex
						if (cx == (lcx - 1) && cy == (lcy - 1) && cz == 0) {
							if (lCell[cx][cy][cz].IN_FLAG == 1 || lCell[cx - 1][cy][cz].IN_FLAG == 0 || lCell[cx][cy - 1][cz].IN_FLAG == 0 || lCell[cx][cy][cz + 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// Corner 4, left bottom front vertex
						if (cx == (0) && cy == (0) && cz == (lcz - 1)) {
							if (lCell[cx][cy][cz].IN_FLAG == 1 || lCell[cx + 1][cy][cz].IN_FLAG == 0 || lCell[cx][cy + 1][cz].IN_FLAG == 0 || lCell[cx][cy][cz - 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// Corner 5, right bottom front vertex
						if (cx == (lcx - 1) && cy == (0) && cz == (lcz - 1)) {
							if (lCell[cx][cy][cz].IN_FLAG == 1 || lCell[cx - 1][cy][cz].IN_FLAG == 0 || lCell[cx][cy + 1][cz].IN_FLAG == 0 || lCell[cx][cy][cz - 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// Corner 6, left top front vertex
						if (cx == (0) && cy == (lcy - 1) && cz == (lcz - 1)) {
							if (lCell[cx][cy][cz].IN_FLAG == 1 || lCell[cx + 1][cy][cz].IN_FLAG == 0 || lCell[cx][cy - 1][cz].IN_FLAG == 0 || lCell[cx][cy][cz - 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}
						// Corner 7, right top front vertex
						if (cx == (lcx - 1) && cy == (lcy - 1) && cz == (lcz - 1)) {
							if (lCell[cx][cy][cz].IN_FLAG == 1 || lCell[cx - 1][cy][cz].IN_FLAG == 0 || lCell[cx][cy - 1][cz].IN_FLAG == 0 || lCell[cx][cy][cz - 1].IN_FLAG == 0)
							{
								lCell[cx][cy][cz].EDGE_FLAG = 1;
							}
						}


						// Full volume 
						if (lCell[cx][cy][cz].EDGE_FLAG != 1 && lCell[cx][cy][cz].IN_FLAG == 1) {
							lCell[cx][cy][cz].volume = lGRIDL*lGRIDL*lGRIDL;
						}
						else
							// partial volume
						{
							int volume_count = 0;
							for (j = 0; j<8; j++) {
								if (lCell[cx][cy][cz].CORNERS[j] != -1) {
									volume_count = volume_count + 1;
								}
							}
							if (volume_count == 8){ lCell[cx][cy][cz].volume = lGRIDL*lGRIDL*lGRIDL; }
							if (volume_count >3)
							{
								// need something to deal with four points that are planar!
								if (volume_count == 4){
									if ((lCell[cx][cy][cz].FCORNERS[0] == 1 && lCell[cx][cy][cz].FCORNERS[2] == 1 && lCell[cx][cy][cz].FCORNERS[4] == 1 && lCell[cx][cy][cz].FCORNERS[6] == 1)
										|| (lCell[cx][cy][cz].FCORNERS[1] == 1 && lCell[cx][cy][cz].FCORNERS[3] == 1 && lCell[cx][cy][cz].FCORNERS[5] == 1 && lCell[cx][cy][cz].FCORNERS[7] == 1)
										|| (lCell[cx][cy][cz].FCORNERS[0] == 1 && lCell[cx][cy][cz].FCORNERS[1] == 1 && lCell[cx][cy][cz].FCORNERS[4] == 1 && lCell[cx][cy][cz].FCORNERS[5] == 1)
										|| (lCell[cx][cy][cz].FCORNERS[2] == 1 && lCell[cx][cy][cz].FCORNERS[3] == 1 && lCell[cx][cy][cz].FCORNERS[6] == 1 && lCell[cx][cy][cz].FCORNERS[7] == 1)
										|| (lCell[cx][cy][cz].FCORNERS[0] == 1 && lCell[cx][cy][cz].FCORNERS[1] == 1 && lCell[cx][cy][cz].FCORNERS[2] == 1 && lCell[cx][cy][cz].FCORNERS[3] == 1)
										|| (lCell[cx][cy][cz].FCORNERS[4] == 1 && lCell[cx][cy][cz].FCORNERS[5] == 1 && lCell[cx][cy][cz].FCORNERS[6] == 1 && lCell[cx][cy][cz].FCORNERS[7] == 1)
										)
									{
										lCell[cx][cy][cz].volume = 0;
									}
									else{ lCell[cx][cy][cz].volume = lGRIDL*lGRIDL*lGRIDL / 6; }
								}
								if (volume_count == 5){
									lCell[cx][cy][cz].volume = lGRIDL*lGRIDL*lGRIDL / 4;
									//printf("ping5");
								}
								if (volume_count == 6){
									lCell[cx][cy][cz].volume = lGRIDL*lGRIDL*lGRIDL / 2;
									//printf("ping6");
								}
								if (volume_count == 7){
									lCell[cx][cy][cz].volume = 5 * lGRIDL*lGRIDL*lGRIDL / 6;
									//printf("ping7");
								}
							}
							else
							{
								lCell[cx][cy][cz].volume = 0;
							}
						}
					}
				}
			}
			if (cx * 100 / lcx > 10 && cx * 100 / lcx < 20 && i10 == 0){ printf("10%% edge detection complete \n"); i10 = 1; }
			if (cx * 100 / lcx > 20 && cx * 100 / lcx < 30 && i20 == 0){ printf("20%% edge detection complete \n"); i20 = 1; }
			if (cx * 100 / lcx > 30 && cx * 100 / lcx < 40 && i30 == 0){ printf("30%% edge detection complete \n"); i30 = 1; }
			if (cx * 100 / lcx > 40 && cx * 100 / lcx < 50 && i40 == 0){ printf("40%% edge detection complete \n"); i40 = 1; }
			if (cx * 100 / lcx > 50 && cx * 100 / lcx < 60 && i50 == 0){ printf("50%% edge detection complete \n"); i50 = 1; }
			if (cx * 100 / lcx > 60 && cx * 100 / lcx < 70 && i60 == 0){ printf("60%% edge detection complete \n"); i60 = 1; }
			if (cx * 100 / lcx > 70 && cx * 100 / lcx < 80 && i70 == 0){ printf("70%% edge detection complete \n"); i70 = 1; }
			if (cx * 100 / lcx > 80 && cx * 100 / lcx < 90 && i80 == 0){ printf("80%% edge detection complete \n"); i80 = 1; }
			if (cx * 100 / lcx > 90 && i90 == 0){ printf("90%% edge detection complete \n"); i90 = 1; }
		}

		flag = 1;
	}

	for (n = 0; n < lnp; n++)
	{
		rx = prt[n].r.x + LX / 2;
		ry = prt[n].r.y + LY / 2;
		rz = prt[n].r.z;

		cx = floor(rx / (lGRIDL)+0.5);
		if (cx >(lcx - 1) || cx < 0){ cx = (lcx - 1); }
		cy = floor(ry / (lGRIDL)+0.5);
		if (cy >(lcy - 1) || cy < 0){ cy = (lcy - 1); }
		cz = floor(rz / (lGRIDL)+0.5);
		if (cz >(lcz - 1) || cz < 0){ cz = (lcz - 1); }

		pnum = lCell[cx][cy][cz].pnum;

		lCell[cx][cy][cz].plist[pnum] = n;
		lCell[cx][cy][cz].pnum++;
		if (pnum >= PLIMIT){
			//printf("pnum exceeded particle limit, unstable behaviour expected, core dump imminent!\n");
			//printf("c: %i %i %i pnum: %i\n", cx, cy, cz, pnum);
		}
	}
}

#endif