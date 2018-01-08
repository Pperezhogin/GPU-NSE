#include	<stdio.h>
#include	"model.h"
#include	"kernels.h"

int main (int argc, char *  argv [])
{
	init();
	int i;
	
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	
	for (i = 0; i < 1; i++)
		jacobi_noshared(d, par, gpu_c);
	
	residual(d, par, gpu_c);
	printf(" residual = %E, maxpsi = %E \n", d.residual(par), d.maxpsi(par));
	
	clear();
	init();
	
	for (i = 0; i < 1; i++)
		jacobi(d, par, gpu_c);
	
	residual(d, par, gpu_c);
	printf(" residual = %E, maxpsi = %E \n", d.residual(par), d.maxpsi(par));
	
	
	clear();	
	return 0;
}
