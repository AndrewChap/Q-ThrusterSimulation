#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "OtherFunctions.cuh"

#define char_amount			100


void WriteVTK(
				int CUDA, int frame, int np_buffer,
				float *px,
				float *py,
				float *pz,
				int *pq,
				int snowballsize
				){

	static char timename[char_amount];
	//static char codename[char_amount];
	char foldername[char_amount];

	char filename_p[char_amount];	// positron data
	char filename_e[char_amount];	// electron data
	char filename_d[char_amount];	// density data
	static char dir_positrons[char_amount];
	static char dir_electrons[char_amount];
	static char dir_density[char_amount];
	printf("frame = %i\n",frame);
	if (frame == 0){
		sprintf(timename, datetime(time(0)));
		//sprintf(codename, __FILE__);
		if (CUDA == 1){
			sprintf(foldername, "GPU");
		}
		else if (CUDA == 0){
			sprintf(foldername, "CPU");
		}

		sprintf(foldername + strlen(foldername), timename);
		//sprintf(foldername + strlen(foldername), "%s", codename);
		printf("\ncreating directory:\n%s\n", foldername);
	    mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH; // Equivalent to 0755
        if (mkdir(foldername, mode) == -1) {
            perror("mkdir foldername failed");
        }


		sprintf(dir_positrons, "%s/positrons", foldername);
		sprintf(dir_electrons, "%s/electrons", foldername);
		sprintf(dir_density, "%s/density", foldername);
		printf("\ncreating directory:\n%s\n", dir_positrons);
        if (mkdir(dir_positrons, mode) == -1) {
            perror("mkdir positrons failed");
        }
        if (mkdir(dir_electrons, mode) == -1) {
            perror("mkdir electrons failed");
        }
        if (mkdir(dir_density, mode) == -1) {
            perror("mkdir density failed");
        }
	}


	int print_gots = 0;

	for (int s = 0; s < snowballsize; s++){

		int thisframe = frame + s;
		int n_begin = s*np_buffer;
		int n_end = (s + 1)*np_buffer;
		//printf("s = %i, n_begin = %i, n_end = %i, np_buffer = %i\n", s, n_begin, n_end, np_buffer);
		sprintf(filename_p, "%s/positrons%05i.vtk", dir_positrons, thisframe);
		sprintf(filename_e, "%s/electrons%05i.vtk", dir_electrons, thisframe);
		sprintf(filename_d, "%s/density%05i.vtk", dir_density, thisframe);

		FILE  *vtk_out_p, *vtk_out_e, *vtk_out_d;

		if (print_gots == 1){ printf("got 5.2\n"); }
		vtk_out_p = fopen(filename_p, "w");
		vtk_out_e = fopen(filename_e, "w");
		vtk_out_d = fopen(filename_d, "w");
		if (print_gots == 1){ printf("got 5.3a\n"); }

		int num_pos_check = 0;
		int num_ele_check = 0;
		int np_check = 0;

		/* was having trouble with num_pos_check not equaling num_pos and same with num_ele after many time steps.
		For example, on Frame 817, after running fine for 163400 time-steps, got this (from commented out error catch written below):
		"ERROR: num_pos = 125, num_pos_check = 126, num_ele = 133, num_ele_check = 132, np = 258, np_check = 258"
		Can't figure out why this happens, which is why I now explicitly count the number of positrons and electrons below */

		if (print_gots == 1){ printf("got 5.3b\n"); }
		for (int n = n_begin; n < n_end; n++){
			if (pq[n] == 1){
				num_pos_check++;
				np_check++;
			}
			else if (pq[n] == -1){
				num_ele_check++;
				np_check++;
			}
			//else{
			//		printf("PARTICLE %i q = %i\n", n, prt[n].q);
			//}
		}
		if (print_gots == 1){ printf("got 5.3c\n"); }
		//printf("s = %i, num_pos_check = %i\n", s, num_pos_check);
		fprintf(vtk_out_p, "# vtk DataFile Version 2.0\nPOSITRONS\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS %i double\n", num_pos_check);
		fprintf(vtk_out_e, "# vtk DataFile Version 2.0\nELECTRONS\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS %i double\n", num_ele_check);
		if (print_gots == 1){ printf("got 5.4\n"); }

		num_pos_check = 0;
		num_ele_check = 0;
		np_check = 0;

		for (int n = n_begin; n < n_end; n++){
			if (pq[n] == 1){
				fprintf(vtk_out_p, "%10.5f%10.5f%10.5f\n", px[n], py[n], pz[n]);
				num_pos_check++;
				np_check++;
			}
			else if (pq[n] == -1){
				fprintf(vtk_out_e, "%10.5f%10.5f%10.5f\n", px[n], py[n], pz[n]);
				num_ele_check++;
				np_check++;
			}
		}
		//printf("%4i positrons, %4i electrons\n", num_pos_check, num_ele_check);

		if (print_gots == 1){ printf("got 5.5\n"); }
		/*if (num_pos != num_pos_check || num_ele != num_ele_check){
		printf("ERROR: num_pos = %i, num_pos_check = %i, num_ele = %i, num_ele_check = %i, np = %i, np_check = %i\n",num_pos,num_pos_check,num_ele,num_ele_check,np,np_check);
		exit(0);
		}*/  // See above explanation for why this is commented out

		/*for (bx = 0; bx < nxB; bx++){
		for (by = 0; by < nyB; by++){
		for (bz = 0; bz < nzB; bz++){
		Density[bx][by][bz] = 0.0;
		//Density[cx][cy][cz].v = (VecF){ 0.0, 0.0, 0.0 };
		//Density[cx][cy][cz].q = 0.0;
		}
		}
		}

		if (print_gots == 1){ printf("got 5.6a\n"); }
		float ai, aj, ak, ri, rj, rk, wif, wjf, wkf;
		int wi, wj, wk;

		float weight = RPM / dvd;
		//printf("weight = %e #/m^-3,  RPM = %e, dvd = %e m^-3\n",weight, RPM, dvd);
		// Linear interpolation
		if (print_gots == 1){ printf("got 5.6b\n"); }
		//printf("n = %i\n",n);
		for (n = 0; n<np; n++){
		//printf("s%i ",n);
		if (prt[n].ECellX != (CX_LIMIT - 1) && prt[n].ECellY != (CY_LIMIT - 1) && prt[n].ECellZ != (CZ_LIMIT - 1)){
		ai = (prt[n].r.x + LX / 2) / dxd - 0.5;			// how many dxd's the particle is from the left boundary.  Subtract 0.5 to correctly align (the middle of cell 1 is at -LX/2+dxd/2)
		aj = (prt[n].r.y + LY / 2) / dyd - 0.5;
		ak = (prt[n].r.z) / dzd - 0.5;
		wif = floor(ai);							// round down.  this will be the left cell where the weighting is deposited
		wjf = floor(aj);
		wkf = floor(ak);
		ri = ai - wif;								// weighting linearly based on how close the particle is to the left cell
		rj = aj - wjf;
		rk = ak - wkf;
		wi = (int)wif;
		wj = (int)wjf;
		wk = (int)wkf;

		// distribute the particle weight across the cells
		if (wi >= 0 && wj >= 0 && wk >= 0){ Density[wi][wj][wk].n += (1 - ri)*(1 - rj)*(1 - rk)*weight; }
		if (wi >= 0 && wj >= 0 && wk <nzd - 1){ Density[wi][wj][wk + 1].n += (1 - ri)*(1 - rj)*(rk)*weight; }
		if (wi >= 0 && wj<nyd - 1 && wk >= 0){ Density[wi][wj + 1][wk].n += (1 - ri)*(rj)*(1 - rk)*weight; }
		if (wi >= 0 && wj<nyd - 1 && wk <nzd - 1){ Density[wi][wj + 1][wk + 1].n += (1 - ri)*(rj)*(rk)*weight; }
		if (wi<nxd - 1 && wj >= 0 && wk >= 0){ Density[wi + 1][wj][wk].n += (ri)*(1 - rj)*(1 - rk)*weight; }
		if (wi<nxd - 1 && wj >= 0 && wk <nzd - 1){ Density[wi + 1][wj][wk + 1].n += (ri)*(1 - rj)*(rk)*weight; }
		if (wi<nxd - 1 && wj<nyd - 1 && wk >= 0){ Density[wi + 1][wj + 1][wk].n += (ri)*(rj)*(1 - rk)*weight; }
		if (wi<nxd - 1 && wj<nyd - 1 && wk <nzd - 1){ Density[wi + 1][wj + 1][wk + 1].n += (ri)*(rj)*(rk)*weight; }
		}
		}

		if (print_gots == 1){ printf("got 5.6c\n"); }
		fprintf(vtk_out_d, "# vtk DataFile Version 2.0\nNUMBER DENSITY\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS %i %i %i\nORIGIN%10.5f%10.5f%10.5f\nSPACING%10.5f%10.5f%10.5f", nxd, nyd, nzd, -LX / 2 + dxd / 2, -LY / 2 + dyd / 2, dzd / 2, dxd, dyd, dzd);

		fprintf(vtk_out_d, "\nPOINT_DATA %i\nSCALARS Number_Density double\nLOOKUP_TABLE default\n", nxd*nyd*nzd);
		float dens, densvx, densvy, densvz;
		for (cz = 0; cz < nzd; cz++){
		for (cy = 0; cy < nyd; cy++){
		for (cx = 0; cx < nxd; cx++){
		dens = Density[cx][cy][cz].n;
		fprintf(vtk_out_d, "%f\n", dens);
		}
		}
		}*/

		//printf("c1 = %i, c2 = %i, cxcycz = %i\n", c_total1, c_total2, cx*cy*cz);

		if (print_gots == 1){ printf("got 5.9\n"); }
		fclose(vtk_out_p);
		fclose(vtk_out_e);
		fclose(vtk_out_d);
		if (print_gots == 1){ printf("got 5.10\n"); }
	}

#if 0
	for (int a = 0; a < 10; a++){
		printf("pq[%i] = %i\n", a, pq[a]);
	}
	exit(0);
#endif
}
