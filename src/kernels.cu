#include "kernels.h"
#include "device.cuh"
#include "sharedmem.cuh"

// ----------------------------------------------------------------------------------------- //
// * kernels * //
// ----------------------------------------------------------------------------------------- //

template < typename T >
__global__ void
jacobi_k ( T* psi, const T* psi_old, const T* w, const gpu_constants < T >* gpu_c ) {
	int idx_g, idx_sb;
	SharedMemory<T> smem;
	T* psi_s = smem.getPointer();
	
	idx_g  = id_g  ( gpu_c->tile );
	idx_sb = id_sb ( gpu_c->tile );

	memcpy_cross ( psi_s, psi_old, idx_sb, idx_g, gpu_c->tile);
	__syncthreads();

	jacobi_d ( psi, psi_s, w, idx_g, idx_sb, gpu_c);
}

template < typename T >
__global__ void
jacobi_k_noshared ( T* psi, const T* psi_old, const T* w, const gpu_constants < T >* gpu_c ) {
	int idx_g;

	idx_g  = id_g  ( gpu_c->tile );
	
	jacobi_d_noshared ( psi, psi_old, w, idx_g, gpu_c);
}

template < typename T >
__global__ void
residual_k ( T* r, const T* psi, T* w, const gpu_constants < T >* gpu_c ) {
	int idx_g, idx_sb;
	SharedMemory<T> smem;
	T* psi_s = smem.getPointer();
	
	idx_g  = id_g  ( gpu_c->tile );
	idx_sb = id_sb ( gpu_c->tile );
	
	memcpy_cross ( psi_s, psi, idx_sb, idx_g, gpu_c->tile);
	__syncthreads();

	residual_d ( r, psi_s, w, idx_g, idx_sb, gpu_c);
}

// ----------------------------------------------------------------------------------------- //
// * launchers * //
// ----------------------------------------------------------------------------------------- //

template < typename T >
void jacobi ( data < T >& d, params < T >& par, gpu_constants < T >* gpu_c ) {
	jacobi_k<<<par.Grid, par.Block, par.numBytes_tile_b>>> ( d.psi[d.counter^1], d.psi[d.counter], d.w, gpu_c);
	d.counter ^= 1;
}

template < typename T >
void jacobi_noshared ( data < T >& d, params < T >& par, gpu_constants < T >* gpu_c ) {
	jacobi_k_noshared<<<par.Grid, par.Block>>> ( d.psi[d.counter^1], d.psi[d.counter], d.w, gpu_c);
	d.counter ^= 1;
}

template < typename T >
void residual ( data < T >& d, params < T >& par, gpu_constants < T >* gpu_c ) {
	residual_k<<<par.Grid, par.Block, par.numBytes_tile_b>>> ( d.r, d.psi[d.counter], d.w, gpu_c);
}
// ----------------------------------------------------------------------------------------- //

template void jacobi( data < float >& d, params < float >& par, gpu_constants < float >* gpu_c);
template void jacobi( data < double >& d, params < double >& par, gpu_constants < double >* gpu_c);

template void jacobi_noshared( data < float >& d, params < float >& par, gpu_constants < float >* gpu_c);
template void jacobi_noshared( data < double >& d, params < double >& par, gpu_constants < double >* gpu_c);

template void residual( data < float >& d, params < float >& par, gpu_constants < float >* gpu_c);
template void residual( data < double >& d, params < double >& par, gpu_constants < double >* gpu_c);