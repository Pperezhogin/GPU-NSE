#include "device.cuh"

// global index
__device__ int id_g ( const tile_s& tile ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * tile.y + threadIdx.y * tile.ratio;
	return i + j * tile.nx;
}

// shared memory index including boundary with gcx = 1, gcy =1
__device__ int id_sb ( const tile_s& tile ) {
	return threadIdx.x + tile.xp3 + tile.distance_b * threadIdx.y;
}

// shared memory index without boundary
__device__ int id_s ( const tile_s& tile ) {
	return threadIdx.x + tile.distance * threadIdx.y;
}

__device__ bool isSquare ( const int& idx, const square_s& square, const tile_s& tile ) {
	int i = idx % tile.nx;
	int j = idx / tile.nx;
	return ( ( i >= square.x1 ) && ( i <= square.x2 ) && ( j >= square.y1 ) && (j <= square.y2) ) ? true:false;
}

__device__ bool outofBorder ( const int& idx, const tile_s& tile ) {
	int i = idx % tile.nx;
	int j = idx / tile.nx;
	return ( ( i < 0 ) || ( j < 0 ) || ( i > tile.nx - 1 ) || ( j > tile.ny - 1 ) ) ? true:false;
}

__device__ int inside ( const int& idx, const square_s& square, const tile_s& tile ) {
	int i = idx % tile.nx;
	int j = idx / tile.nx;
	int mark = INSIDE;
	
	if ( ( i == 0 ) || ( j == 0 ) || ( j == tile.ny - 1 ) || isSquare ( idx, square, tile ))
		mark = DIRICHLET;

	if ( i == tile.nx - 1 )
		mark = OUTFLOW;
	
	return  mark;
}

template < typename T >
__device__ void memcpy_cross ( T* destination, const T* source,
			   int idx_sb, int idx_g, const tile_s& tile ) {
	int idx;
	
	// copy boundary
	if ( threadIdx.y == 0 ) {
		idx = idx_g - tile.nx;
		if ( outofBorder ( idx, tile ) == false )
			destination[idx_sb - tile.xp2] = source[idx];
	}

	if ( threadIdx.y == 1 ) {
		idx = blockDim.x * blockIdx.x - 1 + tile.nx * ( blockIdx.y * tile.y + threadIdx.x );
		if ( outofBorder ( idx, tile ) == false )
			if ( threadIdx.x < tile.y )
				destination[( threadIdx.x + 1 ) * tile.xp2] = source[idx];
	}
	
	if ( threadIdx.y == 2 ) {
		idx = blockDim.x * ( blockIdx.x + 1 ) + tile.nx * ( blockIdx.y * tile.y + threadIdx.x );
		if ( outofBorder ( idx, tile ) == false )
			if ( threadIdx.x < tile.y )
				destination[( threadIdx.x + 2 ) * tile.xp2 - 1] = source[idx];
	}
	
	if ( threadIdx.y == blockDim.y - 1 ) {
		idx = idx_g + tile.ratio * tile.nx;
		if ( outofBorder ( idx, tile ) == false )
			destination[idx_sb + tile.distance_b] = source[idx];
	}
	
	// copy interior of tile	
	for ( int offset = 0; offset < tile.ratio; offset++, idx_g += tile.nx, idx_sb += tile.xp2 )
		destination[idx_sb] = source[idx_g];
}

template < typename T >
__device__ void memcpy_halo ( T* destination, const T* source,
			   int idx_sb, int idx_g, const tile_s& tile ) {
	int idx;
	// copy boundary
	switch( threadIdx.y ) {
	case 0:
		idx = idx_g - tile.nx - 1;
		if ( outofBorder ( idx, tile ) == true )
			break;
		else
			destination[idx_sb - tile.xp2 - 1] = source[idx];
		
		idx = idx_g - tile.nx + 1;
		if ( outofBorder ( idx, tile ) == true )
			break;
		else if ( threadIdx.x > blockDim.x - 3 )
			destination[idx_sb - tile.xp2 + 1] = source[idx];
		
		break;
		
	case 1:
		idx = blockDim.x * blockIdx.x - 1 + tile.nx * ( blockIdx.y * tile.y + threadIdx.x );
		if ( outofBorder ( idx, tile ) == true )
			break;
		else
			destination[( threadIdx.x + 1 ) * tile.xp2] = source[idx];
		
		break;
	
	case 2:
		idx = blockDim.x * ( blockIdx.x + 1 ) + tile.nx * ( blockIdx.y * tile.y + threadIdx.x );
		if ( outofBorder ( idx, tile ) == true )
			break;
		else
			destination[( threadIdx.x + 2 ) * tile.xp2 - 1] = source[idx];
		
		break;
		
	case 3:
		idx = idx_g + tile.ratio * tile.nx - 1;
		if ( outofBorder ( idx, tile ) == true )
			break;
		else
			destination[idx_sb + tile.distance_b - 1] = source[idx];
		
		idx = idx_g + tile.ratio * tile.nx + 1;
		if ( outofBorder ( idx, tile ) == true )
			break;
		else if ( threadIdx.x > blockDim.x - 3 )
			destination[idx_sb + tile.distance_b + 1] = source[idx];
		
		break;
	}
	
	// copy interior of tile	
	for ( int offset = 0; offset < tile.ratio; offset++, idx_g += tile.nx, idx_sb += tile.xp2 )
		destination[idx_sb] = source[idx_g];
}

template < typename T >
__device__ void memcpy ( T* destination, const T* source,
			   int idx_sb, int idx_g, const tile_s& tile ) {
	// copy interior of tile	
	for ( int offset = 0; offset < tile.ratio; offset++, idx_g += tile.nx, idx_sb += tile.xp2 )
		destination[idx_sb] = source[idx_g];
}

template < typename T >
__device__ void jacobi_d ( T* psi, const T* psi_s, const T* w,
			int idx_g, int idx_sb, const gpu_constants < T >* gpu_c ) {
	int mark;
	const square_s &square = gpu_c->square;
	const tile_s &tile = gpu_c->tile;
	const jacobi_s < T > &jacobi = gpu_c->jacobi;
	
	for ( int offset = 0; offset < tile.ratio; offset++, idx_g += tile.nx, idx_sb += tile.xp2 ) {
		mark = inside ( idx_g, square, tile );
		if ( mark == INSIDE ) {
			psi[idx_g] = jacobi.a * ( psi_s[idx_sb + 1] + psi_s[idx_sb - 1] )
				    + jacobi.b * ( psi_s[idx_sb + tile.xp2] + psi_s[idx_sb - tile.xp2] )
				    + jacobi.c * w[idx_g];
		}
		else if ( mark == OUTFLOW )
			psi[idx_g] = psi_s[idx_sb - 1];
	}
}

template < typename T >
__device__ void jacobi_d_noshared ( T* psi, const T* psi_old, const T* w,
			int idx_g, const gpu_constants < T >* gpu_c ) {
	int mark;
	const square_s &square = gpu_c->square;
	const tile_s &tile = gpu_c->tile;
	const jacobi_s < T > &jacobi = gpu_c->jacobi;
	
	for ( int offset = 0; offset < tile.ratio; offset++, idx_g += tile.nx ) {
		mark = inside ( idx_g, square, tile );
		if ( mark == INSIDE ) {
			psi[idx_g] = jacobi.a * ( psi_old[idx_g + 1] + psi_old[idx_g - 1] )
				    + jacobi.b * ( psi_old[idx_g + tile.nx] + psi_old[idx_g - tile.nx] )
				    + jacobi.c * w[idx_g];
		}
		else if ( mark == OUTFLOW )
			psi[idx_g] = psi_old[idx_g - 1];
	}
}

template < typename T >
__device__ void residual_d ( T* r, const T* psi_s, const T* w,
			int idx_g, int idx_sb, const gpu_constants < T >* gpu_c ) {
	int mark;
	const square_s &square = gpu_c->square;
	const tile_s &tile = gpu_c->tile;
	const laplace_s < T > &laplace = gpu_c->laplace;
	
	for ( int offset = 0; offset < tile.ratio; offset++, idx_g += tile.nx, idx_sb += tile.xp2 ) {
		mark = inside ( idx_g, square, tile );
		
		if ( mark == INSIDE )
			r[idx_g] = laplace.c0 * psi_s[idx_sb] 
				  + laplace.cx * ( psi_s[idx_sb + 1] + psi_s[idx_sb - 1] )
				  + laplace.cy * ( psi_s[idx_sb + tile.xp2] + psi_s[idx_sb - tile.xp2] ) 
				  - w[idx_g];
		else if ( mark == OUTFLOW )
			r[idx_g] = laplace.cx * ( - psi_s[idx_sb] + psi_s[idx_sb - 1] );
		else
			r[idx_g] = (T) 0.0;
	}
	
}

// ------------------------------------------------------------------------------------------------------- //

template __device__ 
void memcpy_cross ( float* destination, const float* source, 
	      int idx_sb, int idx_g, const tile_s& tile );
template __device__ 
void memcpy_cross ( double* destination, const double* source, 
	      int idx_sb, int idx_g, const tile_s& tile );

template __device__ 
void memcpy_halo ( float* destination, const float* source, 
	      int idx_sb, int idx_g, const tile_s& tile );
template __device__ 
void memcpy_halo ( double* destination, const double* source, 
	      int idx_sb, int idx_g, const tile_s& tile );

template __device__ 
void memcpy ( float* destination, const float* source, 
	      int idx_sb, int idx_g, const tile_s& tile );
template __device__ 
void memcpy ( double* destination, const double* source, 
	      int idx_sb, int idx_g, const tile_s& tile );

template __device__
void jacobi_d ( float* psi, const float* psi_s, const float* w, 
		  int idx_sb, int idx_g, const gpu_constants < float >* gpu_c );
template __device__ 
void jacobi_d ( double* psi, const double* psi_s, const double* w, 
		  int idx_sb, int idx_g, const gpu_constants < double >* gpu_c );

template __device__
void jacobi_d_noshared ( float* psi, const float* psi_old, const float* w, 
		  int idx_g, const gpu_constants < float >* gpu_c );
template __device__ 
void jacobi_d_noshared ( double* psi, const double* psi_old, const double* w, 
		  int idx_g, const gpu_constants < double >* gpu_c );

template __device__
void residual_d ( float* r, const float* psi_s, const float* w, 
		  int idx_sb, int idx_g, const gpu_constants < float >* gpu_c );
template __device__ 
void residual_d ( double* r, const double* psi_s, const double* w, 
		  int idx_sb, int idx_g, const gpu_constants < double >* gpu_c );
