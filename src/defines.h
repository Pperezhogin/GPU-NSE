#pragma once

#define Real float

#define NX 512
#define NY 256
#define GCX 1  // only 1!
#define GCY 1

#define SQUARE_CX 128
#define SQUARE_CY 128
#define SQUARE_L  64 
#define SQUARE_W  64

// accessible area by 1 block
#define TILE_X 32 // tile_x == block_x
#define TILE_Y 32 // tile_y < tile_x

#define BLOCK_X 32 //
#define BLOCK_Y 4 // not less then 4

#define LENGTH 1.0
#define WIDTH 0.5
#define CFL 1.6
#define V 1.0 //  inflow velocity
#define NU 0.01 //viscosity