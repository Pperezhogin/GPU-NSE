#pragma once

#include "data.h"
#include "defines.h"

// params declaration
params < Real > par;

// constants on gpu
__constant__ __device__ gpu_constants < Real > *gpu_c;
gpu_constants < Real > *host_c;

// data on GPU and host
data < Real > d;

void init() {
	// init params on host
	par.init_const( (Real) LENGTH, (Real) WIDTH, 
		  NX, NY, (Real) CFL, (Real) NU, (Real) V );
	par.init_block( TILE_X, TILE_Y, GCX, GCY, BLOCK_X, BLOCK_Y );
	par.init_square( SQUARE_CX, SQUARE_CY, SQUARE_L, SQUARE_W );

	gpu_constants_init( host_c, gpu_c, par );
	
	d.init ( par );
}

void clear() { d.clear(); }