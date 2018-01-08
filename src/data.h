#pragma once
#include <stdio.h>

// ----------------------------------------------------------------------------------------- //
// * operators * //
// ----------------------------------------------------------------------------------------- //

// * jacobi iterations * //
template < typename T >
struct jacobi_s {
	T a, b, c;
};

// * normal second derivative * //
template < typename T >
struct d2n_s {
	T cx0, cx1, cx2, cx3;
	T cy0, cy1, cy2, cy3;
};

// * laplace pattern * //
template < typename T >
struct laplace_s {
	T c0;
	T cx;
	T cy;
};

// ----------------------------------------------------------------------------------------- //
// * parameters * //
// ----------------------------------------------------------------------------------------- //

// * computational tile * //
struct tile_s {
	// y = tile_y, xp2 = tile_x + 2,  ratio = tile_y / block_y, distance between (i,j) and (i,j+1) in shared memory, distance_b including boundary
	int y, xp2, xp3, nx, ny, ratio, distance, distance_b;
};

// * square * //
struct square_s { 
	int x1, y1, x2, y2;
};

// ----------------------------------------------------------------------------------------- //
// * collect structures * //
// ----------------------------------------------------------------------------------------- //
template < typename T >
struct gpu_constants {
	jacobi_s < T > jacobi;
	d2n_s < T > d2n;
	laplace_s < T > laplace;
	
	tile_s tile;
	square_s square;
};

template < typename T >
struct params {
	T dt;
	T dx;
	T dy;
	T nu;
	T v;

	int nx, ny, size;
	T length, width;
	
	int tile_x, tile_y, gcx, gcy;
	int x1, x2, y1, y2; // square params
	
	dim3 Block;
	dim3 Grid;   
	
	int numBytes_tile;
	int numBytes_tile_b;
  
	public:
		void init_const( T _length, T _width, int _nx, int _ny, T _CFL, T _nu, T _V );
		void init_block( int _tile_x, int _tile_y, int _gcx, int _gcy, int _block_x, int _block_y );
		void init_square( int _cx, int _cy, int _l, int _w );
};

// * variables on host and gpu, kernel configuration * //
// * and interface for data transfer 		      * //

template < typename T >
struct data {
	int numBytes;	
	int counter; // number of the current array
	
	// on host
	T *hpsi;
	T *hw;
	T *hr;
	// on gpu
	T *psi[2];
	T *w;
	T *r;
	
	public:
		void init ( params < T >& par );
		void clear ();
		void psi_HostToDevice ();
		void psi_DeviceToHost ();
		void w_HostToDevice ();
		void w_DeviceToHost ();
		void r_HostToDevice ();
		void r_DeviceToHost ();
		T residual ( params < T >& par );
		void writeToFile( params < T >& par );
		T maxpsi( params < T >& par );
};

// ----------------------------------------------------------------------------------------- //
// * functions * //
// ----------------------------------------------------------------------------------------- //

template < typename T >
void gpu_constants_init ( gpu_constants < T >*& host, gpu_constants < T >*& gpu, params < T >& par ); 