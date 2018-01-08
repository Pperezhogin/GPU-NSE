#pragma once
#include "data.h"

// * kernel launchers * //

template< typename T >
void jacobi( data < T >& d, params < T >& par, gpu_constants < T >* gpu_c );

template< typename T >
void jacobi_noshared( data < T >& d, params < T >& par, gpu_constants < T >* gpu_c );

template< typename T >
void residual( data < T >& d, params < T >& par, gpu_constants < T >* gpu_c );