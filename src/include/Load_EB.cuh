

void Load_EB(int nr, float SCALING, float COMSOL_CONV, float LX, float LY, float **E_realC, float **E_imagC, float **B_realC, float **B_imagC){

	char filename[25];

	int r, c;

	sprintf(filename, "complex/e_real.txt");
	FILE *fileID;
	fileID = fopen(filename, "r");
	for (r = 0; r < nr; r++) {
		for (c = 0; c < 6; c++) {
			fscanf(fileID, "%f", &E_realC[r][c]);
		}
	}
	sprintf(filename, "complex/e_imag.txt");
	fileID = fopen(filename, "r");
	for (r = 0; r < nr; r++) {
		for (c = 0; c < 6; c++) {
			fscanf(fileID, "%f", &E_imagC[r][c]);
		}
	}

	sprintf(filename, "complex/b_real.txt");
	fileID = fopen(filename, "r");
	for (r = 0; r < nr; r++) {
		for (c = 0; c < 6; c++) {
			fscanf(fileID, "%f", &B_realC[r][c]);
		}
	}

	sprintf(filename, "complex/b_imag.txt");
	fileID = fopen(filename, "r");
	for (r = 0; r < nr; r++) {
		for (c = 0; c < 6; c++) {
			fscanf(fileID, "%f", &B_imagC[r][c]);
		}
	}

	for (r = 0; r < nr; r++){
		for (c = 3; c < 6; c++){
			E_realC[r][c] *= SCALING;
			E_imagC[r][c] *= SCALING;
			B_realC[r][c] *= SCALING;
			B_imagC[r][c] *= SCALING;
		}
		E_realC[r][0] = E_realC[r][0] / COMSOL_CONV;// +LX / 2;
		E_realC[r][1] = E_realC[r][1] / COMSOL_CONV;// +LY / 2;
		E_realC[r][2] = E_realC[r][2] / COMSOL_CONV;
	}

}
