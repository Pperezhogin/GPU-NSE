#include <stdlib.h> 
#include <string.h>

#include "data.h"

template < typename T >
inline T sqr ( T x ) { return x * x; }
	
	
// * gpu_constants init * //
template < typename T >
void gpu_constants_init ( gpu_constants < T >*& host, gpu_constants < T >*& gpu, params < T >& par ) {
	
	host = ( struct gpu_constants < T >* ) malloc( sizeof( struct gpu_constants < T > ) );
	cudaMalloc ( (void**)&gpu, sizeof( struct gpu_constants < T > ) );
	
	host->jacobi.a = + sqr(par.dy) / (T) 2.0 / ( sqr(par.dx) + sqr(par.dy));
	host->jacobi.b = + sqr(par.dx) / (T) 2.0 / ( sqr(par.dx) + sqr(par.dy));
	host->jacobi.c = - sqr(par.dx) * sqr(par.dy) / (T) 2.0 / ( sqr(par.dx) + sqr(par.dy));
	
	host->d2n.cx0 = (T) 2.0 / sqr(par.dx);
	host->d2n.cx1 = (T)-5.0 / sqr(par.dx);
	host->d2n.cx2 = (T) 4.0 / sqr(par.dx);
	host->d2n.cx3 = (T)-1.0 / sqr(par.dx);
	
	host->d2n.cy0 = (T) 2.0 / sqr(par.dy);
	host->d2n.cy1 = (T)-5.0 / sqr(par.dy);
	host->d2n.cy2 = (T) 4.0 / sqr(par.dy);
	host->d2n.cy3 = (T)-1.0 / sqr(par.dy);
	
	host->laplace.c0 = (T)-2.0 / sqr(par.dx) + (T)-2.0 / sqr(par.dy);
	host->laplace.cx = (T) 1.0 / sqr(par.dx);
	host->laplace.cy = (T) 1.0 / sqr(par.dy);	
	
	host->tile.y   = par.tile_y;
	host->tile.xp2 = par.tile_x + 2;
	host->tile.xp3 = par.tile_x + 3;
	host->tile.nx  = par.nx;
	host->tile.ny  = par.ny;
	host->tile.ratio = par.tile_y / par.Block.y;
	host->tile.distance = host->tile.ratio * par.tile_x;
	host->tile.distance_b = host->tile.ratio * ( par.tile_x + 2 );
	
	host->square.x1 = par.x1;
	host->square.y1 = par.y1;
	host->square.x2 = par.x2;
	host->square.y2 = par.y2;
	
	cudaMemcpy ( gpu, host, sizeof( struct gpu_constants < T > ), cudaMemcpyHostToDevice );
}

// * params * //
template < typename T >
void params < T >::init_const ( T _length, T _width, int _nx, 
			        int _ny, T _CFL, T _nu, T _V ) {
	//define host variables
	length = _length;
	width  = _width;
	nx  = _nx;
	ny  = _ny;
	size = nx * ny;
	dx = length / (T) (nx - 1);
	dy = width / (T) (ny - 1);
	dt = _CFL * dx / _V;
	nu = _nu;	
	v  = _V;
}

template < typename T >
void params < T >::init_block ( int _tile_x, int _tile_y,
			        int _gcx, int _gcy, int _block_x, int _block_y ) {
	Block.x = _block_x;
	Block.y = _block_y;;
	Block.z = 1;
	
	Grid.x = nx / _tile_x;
	Grid.y = ny / _tile_y;
	Grid.z = 1; 
	
	tile_x = _tile_x;
	tile_y = _tile_y;
	gcx = _gcx;
	gcy = _gcy;
	
	numBytes_tile = sizeof( T ) * tile_x * tile_y;
	numBytes_tile_b = sizeof( T ) * ( tile_x + 2 ) * ( tile_y + 2 );
}

template < typename T >
void params < T >::init_square ( int _cx, int _cy, int _l, int _w ) {
	x1 = _cx - _l / 2;
	x2 = _cx + _l / 2;
	y1 = _cy - _l / 2;
	y2 = _cy + _l / 2;
}


// * data * //
template < typename T >
void data < T >::init ( params < T >& par ) {
	numBytes = sizeof(T) * par.size;
	counter = 0;
	
	// allocate field variables
	hpsi = ( T* ) malloc ( numBytes );
	hw   = ( T* ) malloc ( numBytes );
	hr   = ( T* ) malloc ( numBytes );

	cudaMalloc ( (void**)&psi[0], numBytes );
	cudaMalloc ( (void**)&psi[1], numBytes );
	cudaMalloc ( (void**)&w, numBytes );
	cudaMalloc ( (void**)&r, numBytes );
	
	// null variables
	memset ( hpsi, 0, numBytes );
	memset ( hw, 0, numBytes );
	memset ( hr, 0, numBytes );
	
	// set boundary values
	for ( int j = 0; j < par.ny; j++ )
		hpsi[ j * par.nx ] = - (T) j / (T) par.ny * par.width * par.v;
	
	for ( int i = 0; i < par.nx; i++)
		hpsi[ (par.ny - 1) * par.nx + i ] = - par.width * par.v;
	
	for ( int i = par.x1; i <= par.x2; i++ )
		for ( int j = par.y1; j <= par.y2; j++ )
			hpsi[ i + j * par.nx ] = - (T) 0.5 * par.width * par.v;	
	
	cudaMemcpy ( psi[0], hpsi, numBytes, cudaMemcpyHostToDevice );
	cudaMemcpy ( psi[1], hpsi, numBytes, cudaMemcpyHostToDevice );
	cudaMemcpy ( w, hw, numBytes, cudaMemcpyHostToDevice );
	cudaMemcpy ( r, hr, numBytes, cudaMemcpyHostToDevice );
}

template < typename T >
void data < T >::clear() {
	free(hpsi);
	free(hw);
	
	cudaFree(psi);
	cudaFree(w);
}

template < typename T >
void data < T >::psi_HostToDevice () { cudaMemcpy ( psi[counter], hpsi, numBytes, cudaMemcpyHostToDevice ); }

template < typename T >
void data < T >::psi_DeviceToHost () { cudaMemcpy ( hpsi, psi[counter], numBytes, cudaMemcpyDeviceToHost ); }

template < typename T >
void data < T >::w_HostToDevice () { cudaMemcpy ( w, hw, numBytes, cudaMemcpyHostToDevice ); }

template < typename T >
void data < T >::w_DeviceToHost () { cudaMemcpy ( hw, w, numBytes, cudaMemcpyDeviceToHost ); }

template < typename T >
void data < T >::r_HostToDevice () { cudaMemcpy ( r, hr, numBytes, cudaMemcpyHostToDevice ); }

template < typename T >
void data < T >::r_DeviceToHost () { cudaMemcpy ( hr, r, numBytes, cudaMemcpyDeviceToHost ); }

template < typename T >
T data < T >::residual ( params < T >& par ) {
	T out = (T) 0.0;
	
	r_DeviceToHost();
	
	for ( int i = 0; i < par.size; i++ )
		if ( fabs(hr[i]) > out )
			out = fabs(hr[i]);
	return out;
}

template < typename T >
void data < T >::writeToFile ( params < T >& par ) {
	FILE* ptr = fopen("psi.txt", "w");
	
	psi_DeviceToHost();
	for (int i = 0; i < par.size; i++)
		fprintf ( ptr, "%E\n", hpsi[i]);
}	

template < typename T >
T data < T >::maxpsi ( params < T >& par ) {
	T out = (T) 0.0;
	
	psi_DeviceToHost();
	
	for ( int i = 0; i < par.size; i++ )
		if ( fabs(hpsi[i]) > out )
			out = fabs(hpsi[i]);
	return out;
}

// ----------------------------------------------------------------------------------------- //

template struct jacobi_s < float >;
template struct jacobi_s < double >;

template struct d2n_s < float >;
template struct d2n_s < double >;

template struct laplace_s < float >;
template struct laplace_s < double >;

template struct params < float >;
template struct params < double >;

template struct data < float >;
template struct data < double >;

// ----------------------------------------------------------------------------------------- //
template void gpu_constants_init ( gpu_constants < float >*& host, gpu_constants < float >*& gpu, params < float >& par ); 
template void gpu_constants_init ( gpu_constants < double >*& host, gpu_constants < double >*& gpu, params < double >& par ); 