#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <sys/stat.h>

/*
#define gpuErrchk(ans) { gpuAssert((ans), __FILENAME__, __LINE__); }					// replace '__FILENAME__' with '__FILE__' if you want the full path
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#define TimerWrapperCPU(func) { timer(func); }					// replace '__FILENAME__' with '__FILE__' if you want the full path
inline float timer(int func)
{
	printf("\n\nStarting CPU kernel\n");
	clock_t startCPU;
	float CPUtime;
	startCPU = clock();

	int success = func;

	CPUtime = ((float)(clock() - startCPU)) / CLOCKS_PER_SEC;
	printf("CPU kernel finished\n");
	printf("CPUtime = %6.10f ms\n", ((float)CPUtime)*1E3);
	return CPUtime;

}
*/
#if 0
#define TimerInitializeCPU \
	clock_t CPU_start;

#define TimerWrapperCPU(func,time) \
	/*printf("\n\nStarting CPU kernel\n");*/ \
	CPU_start = clock(); \
	func; \
	time += ((float)(clock() - CPU_start)) / CLOCKS_PER_SEC; \
	/*printf("CPU kernel finished\n");*/
	/*printf("CPUtime = %6.10f ms\n", ((float)CPUtime)*1E3);*/

/*#define TimerInitializeGPU	\
	cudaEvent_t start, stop;	\
	cudaEventCreate(&start);  cudaEventCreate(&stop);
	*/



#define TimerInitializeCPU	\
	clock_t CPU_start;	    \
	float timerCPU[CPUtimers];

#define TimerStartCPU	\
	CPU_start = clock();

#define TimerStopCPU	\
	timerCPU[__COUNTER__] += ((float)(clock() - CPU_start)) / CLOCKS_PER_SEC;

#define TimerOutputCPU	\
	for (int i = 0; i < CPUtimers - 1; i++){
		printf("%s time = %f"),
	}


#define TimerWrapperGPU(func,time) \
	cudaEventRecord(start); \
	func; \
	cudaEventRecord(stop); \
	cudaEventSynchronize(stop); \
	cudaEventElapsedTime(&time, start, end);

#define TimerInitializeGPU	\
	cudaEvent_t start, stop;	\
	cudaEventCreate(&start);  cudaEventCreate(&stop);	\
	float timerGPU[10];

#define TimerStartGPU
	cudaEventRecord(start);

#define TimerStopGPU
	cudaEventRecord(stop); \
	cudaEventSynchronize(stop); \
	cudaEventElapsedTime(&timerGPU[__COUNTER__], start, end);

#define TimerOutputGPU

/*


printf("\n\nStarting CPU kernel\n");
clock_t startCPU;
float CPUtime;
startCPU = clock();

// code

CPUtime = ((float)(clock() - startCPU)) / CLOCKS_PER_SEC;
printf("CPU kernel finished\n");
printf("CPUtime = %6.10f ms\n", ((float)CPUtime)*1E3);


//----

cudaEvent_t start, end;

cudaEventCreate(&start);
cudaEventCreate(&end);

float milliseconds;

cudaEventRecord(start);

// Kernel launch or memory copy

cudaEventRecord(end);
cudaEventSynchronize(end);

cudaEventElapsedTime(&milliseconds, start, end);

fprintf(stderr, "Elapsed Time = %f milliseconds\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(end);

*/
#endif
