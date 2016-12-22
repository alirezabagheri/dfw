/*	
	Common functionality for Distributed Frank Wolfe
	Copyright (C) 2014-2015  Alireza Bagheri Garakani (me@alirezabagheri.com)

	This program is free software; you can redistribute it and/or
	modify it under the terms of the GNU General Public License
	as published by the Free Software Foundation; either version 2
	of the License, or (at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program; if not, write to the Free Software
	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

*/

#include <set>
#include <fstream> 
#include <iostream>
#include <vector>
#include <string>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/bimap/bimap.hpp>
#include <boost/bimap/unordered_set_of.hpp>
#include <mpi.h>
#include <ctime>
#include <chrono>


/* Definitions
*/
typedef boost::numeric::ublas::matrix<float> mat;
typedef boost::numeric::ublas::vector<float> vec;
typedef boost::numeric::ublas::matrix_range<mat> mat_range;
typedef boost::numeric::ublas::matrix_row<mat> mat_row;
typedef boost::numeric::ublas::matrix_column<mat> mat_col;
typedef boost::numeric::ublas::mapped_matrix<float> mat_sparse;
typedef boost::numeric::ublas::matrix_row<mat_sparse> mat_sparse_row;
typedef boost::numeric::ublas::matrix_column<mat_sparse> mat_sparse_col;
typedef boost::numeric::ublas::vector_range<vec> vec_range;
typedef boost::bimaps::bimap<boost::bimaps::unordered_set_of<int>, boost::bimaps::unordered_set_of<int>> bimap;
typedef std::chrono::time_point<std::chrono::high_resolution_clock> timer;
typedef std::set<int> set;


// Prints vector or matrix type to standard out. Functions should
// only be used for debugging purposes.

void print(vec &v){
	printf("\nVECTOR (%zu) : ", v.size());
	for (size_t i = 0; i < v.size(); i++) printf("%f, ", v(i));
	printf("\n");
}
void print(mat &m){
	printf("\nMATRIX (%zuX%zu) : ", m.size1(), m.size2());
	for (size_t i = 0; i < m.size1(); i++) {
		for (size_t j = 0; j < m.size2(); j++) printf("%f, ", m(i, j));
		printf("\n");
	}
}


// Returns the current time in readable format (e.g., "Thu Jun  5 19:33:28 2014")
std::string time_now_str(){
	std::time_t t = std::time(NULL);
	char time_str[100];
	std::strftime(time_str, sizeof(time_str), "%c", std::localtime(&t));
	return std::string(time_str);
}

// Returns a timer object set to current time. Function is used to measure elapsed time
// from this point (see function 'time_elapsed').
inline timer time_init(){
	return std::chrono::high_resolution_clock::now();
}

// Returns elapsed seconds since time set on timer object.
inline float time_elapsed(timer start){
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
}

// Simplistic random number generator of floating-point number between [0,1]. 
inline float random01(){
	return rand() / float(RAND_MAX);
}


