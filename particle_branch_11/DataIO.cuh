
typedef struct {
	int CUDA;			// 1 or 0 to let us know if the data is coming from the GPU or CPU
	int frame;
	int np_buffer;
	float *px;
	float *py;
	float *pz;
	int *pq;
	float *px_pinned;
	float *py_pinned;
	float *pz_pinned;
	int *pq_pinned;
}DataIOstruct;

DataIOstruct *arg;
float *pgx_pinned, *pgy_pinned, *pgz_pinned;		// arrays for storing the resultant GPU vectors (pd, vd) to display data
int *pgq_pinned;
float *pdx_transfer, *pdy_transfer, *pdz_transfer;		// arrays for storing the resultant GPU vectors (pd, vd) to display data
int *pdq_transfer;
void DataIOinitialize(int WRITE_VTK, int np_buffer){
	if (WRITE_VTK){
		arg = (DataIOstruct*)malloc(sizeof(DataIOstruct));
		// allocate pinned memory on the CPU (cudaMallocHost allocates on the CPU, NOT the GPU!)
		cudaMallocHost((void**)&pgx_pinned, sizeof(float)*np_buffer);
		cudaMallocHost((void**)&pgy_pinned, sizeof(float)*np_buffer);
		cudaMallocHost((void**)&pgz_pinned, sizeof(float)*np_buffer);
		cudaMallocHost((void**)&pgq_pinned, sizeof(int)*np_buffer);
		// allocate memory on the GPU where we transfer the current data so we are not in danger of overwriting data on a later timestep before it has been written to the disk
		cudaMalloc((void**)&pdx_transfer, sizeof(float)*np_buffer);
		cudaMalloc((void**)&pdy_transfer, sizeof(float)*np_buffer);
		cudaMalloc((void**)&pdz_transfer, sizeof(float)*np_buffer);
		cudaMalloc((void**)&pdq_transfer, sizeof(int)*np_buffer);
	}
}

void DataIOthread(void *arg_ptr);

void DataIO(int WRITE_VTK, int CUDA, int frame, int np_buffer, float *pdx, float *pdy, float *pdz, int *pdq){
	if (WRITE_VTK){
		gpuErrchk(cudaMemcpy(pdx_transfer, pdx, sizeof(float)*np_buffer, cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(pdy_transfer, pdy, sizeof(float)*np_buffer, cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(pdz_transfer, pdz, sizeof(float)*np_buffer, cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(pdq_transfer, pdq, sizeof(int)*np_buffer, cudaMemcpyDeviceToDevice));

		arg->CUDA = CUDA;
		arg->frame = frame;		// equivalent to (*arg).frame I think?
		arg->np_buffer = np_buffer;
		/*arg->px = pgx_pinned;
		arg->py = pgy_pinned;
		arg->pz = pgz_pinned;
		arg->pq = pgq_pinned;*/
		arg->px = pdx_transfer;
		arg->py = pdy_transfer;
		arg->pz = pdz_transfer;
		arg->pq = pdq_transfer;
		arg->px_pinned = pgx_pinned;
		arg->py_pinned = pgy_pinned;
		arg->pz_pinned = pgz_pinned;
		arg->pq_pinned = pgq_pinned;
		_beginthread(DataIOthread, 0, (void*)arg);	// do data transfer on separate thread
	}
}

void DataIOthread(void *arg_ptr){

	DataIOstruct *args = (DataIOstruct*)arg_ptr;
	int frame = args->frame;
	int CUDA = args->CUDA;
	int np_buffer = args->np_buffer;
	/*float *px = args->px;
	float *py = args->py;
	float *pz = args->pz;
	int *pq = args->pq;*/
	float *pdx = args->px;
	float *pdy = args->py;
	float *pdz = args->pz;
	int *pdq = args->pq;
	float *px = args->px_pinned;
	float *py = args->py_pinned;
	float *pz = args->pz_pinned;
	int *pq = args->pq_pinned;


	gpuErrchk(cudaMemcpy(px, pdx, sizeof(float)*np_buffer, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(py, pdy, sizeof(float)*np_buffer, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(pz, pdz, sizeof(float)*np_buffer, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(pq, pdq, sizeof(int)*np_buffer, cudaMemcpyDeviceToHost));

	static char timename[char_amount];
	static char codename[char_amount];
	char foldername[char_amount];

	char filename_p[char_amount];	// positron data
	char filename_e[char_amount];	// electron data
	char filename_d[char_amount];	// density data
	static char dir_positrons[char_amount];
	static char dir_electrons[char_amount];
	static char dir_density[char_amount];
	printf("frame = %i\n", frame);
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
		CreateDirectory(foldername, NULL);


		sprintf(dir_positrons, "%s/positrons", foldername);
		sprintf(dir_electrons, "%s/electrons", foldername);
		sprintf(dir_density, "%s/density", foldername);
		CreateDirectory(dir_positrons, NULL);
		CreateDirectory(dir_electrons, NULL);
		CreateDirectory(dir_density, NULL);
		printf("\ncreating directory:\n%s\n", dir_positrons);
	}


	int print_gots = 0;

	for (int s = 0; s < 1; s++){

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
		

		if (print_gots == 1){ printf("got 5.9\n"); }
		fclose(vtk_out_p);
		fclose(vtk_out_e);
		fclose(vtk_out_d);
		if (print_gots == 1){ printf("got 5.10\n"); }
	}
	//free(args);
	_endthread();
}



/*
typedef struct {
	int frame;
	int somenumber;
	int *somearray;
}DataIOstruct;

void DataIO(void *arg_ptr){
	DataIOstruct *args = (DataIOstruct*)arg_ptr;
	int frame = args->frame;
	int somenumber = args->somenumber;
	int *myarray = args->somearray;
	printf("\nHELLO FROM THREAD frame# = %i, somenumber = %i\n", frame,somenumber);
	for (int i = 0; i < 3; i++){
		printf("array[%i] = %i\n", i, myarray[i]);
	}
	free(args);
}*/


/*
void DataIO(void *DIO){
	int outnumber = 1;// ((DataIOstruct*)(DIO)->frame);// +((DataIOstruct*)(DIO)->somenumber);
	DIO.somenumber = 10;
	printf("\nHELLO FROM THREAD frame# = %i\n", (DataIOstruct*)(DIO)->frame);
	_endthread();

}*/