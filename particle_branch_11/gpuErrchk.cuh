#include <string.h>

#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)	// copied from http://stackoverflow.com/questions/8487986/file-macro-shows-full-path, I prefer not getting the full path for easier readability

#define gpuErrchk(ans) { gpuAssert((ans), __FILENAME__, __LINE__); }					// replace '__FILENAME__' with '__FILE__' if you want the full path
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}