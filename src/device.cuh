#pragma once

#include "data.h"

enum { INSIDE, OUTFLOW, DIRICHLET };

__device__ int id_g  ( const tile_s& tile );
__device__ int id_sb ( const tile_s& tile );
__device__ int id_s  ( const tile_s& tile );

__device__ bool isSquare ( const int& idx, const square_s& square, const tile_s& tile );
__device__ bool outofBorder ( const int& idx, const tile_s& tile );
__device__ int  inside   ( const int& idx, const square_s& square, const tile_s& tile );


template < typename T >
__device__ void memcpy_cross ( T* destination, const T* source,
			   int idx_sb, int idx_g, const tile_s& tile );
			   
template < typename T >
__device__ void memcpy_halo ( T* destination, const T* source,
			   int idx_sb, int idx_g, const tile_s& tile );
			   
template < typename T >
__device__ void memcpy ( T* destination, const T* source,
			   int idx_sb, int idx_g, const tile_s& tile );

template < typename T >
__device__ void jacobi_d ( T* psi, const T* psi_s, const T* w, 
			int idx_sb, int idx_g, const gpu_constants < T >* gpu_c ); 
			
template < typename T >
__device__ void jacobi_d_noshared ( T* psi, const T* psi_old, const T* w, 
			int idx_g, const gpu_constants < T >* gpu_c ); 
			
template < typename T >
__device__ void residual_d ( T* r, const T* psi_s, const T* w, 
			int idx_sb, int idx_g, const gpu_constants < T >* gpu_c ); 