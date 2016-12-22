#####Platform: 
* Linux 64-bit (tested on Centos 6.5)

#####Dependencies:
* Boost C++ Libraries
* MPI library (tested with OpenMPI)

#####Getting Started: 
1. Ensure dependencies are installed on your system. 
 * If Boost includes are not located in '/usr/include/boost/', then update the Makefile accordingly. 
 * Check that OpenMPI is installed and the mpic++ command is available in your path.

* Compile distributed Frank Wolfe (dFW) binary for Lasso Regression or Support Vector Machines (SVM). In the following, we prepare the dFW for SVM binary: 
 1. Navigate to the root directory of package and execute: **make frankwolfe\_svm**

 * We will use the provided sample classification data. To prepare your own data, see provided matlab function 'prepare\_data.m'. If your source data is already split into parts dedicated for each node, simple call the matlab function with 'split' argument set to '1'. Otherwise, if your source data is a single file, use 'splits' to partition the data for each node; several other options exist for this partitioning (see matlab code).

 * In the following command, the first two arguments specify the experiment name and input data path. Note that the second argument is the path *prefix* to the data; at run time, each node uses its' unique id to determine its respective data file (e.g., node 3 of 10 will append '.of10.3' to the path prefix argument). Executing the binary without arguments will display usage information for other required arguments. To run the code using two nodes, execute: **mpiexec -n 2 ./frankwolfe_svm test1 data.classification/test .0001 0.5 1000000 0.75 1.2** 

 * Observe output to standard out for program progress and node communication: 

      > [Sun Jan 25 03:12:01 2015 (0) - hostname (rank 0)]     Given input file: data.classification/test.of2.1
      
      > [Sun Jan 25 03:12:01 2015 (0) - hostname (rank 0)]     Reading atom matrix (120x3354) and label vector (3354)...
      
      > [Sun Jan 25 03:12:01 2015 (0) - hostname (rank 1)]     Given input file: data.classification/test.of2.2
      
      > [Sun Jan 25 03:12:01 2015 (0) - hostname (rank 1)]     Reading atom matrix (120x6670) and label vector (6670)...
      
      > [Sun Jan 25 03:12:02 2015 (548) - hostname (rank 0)]   Finished loading training data.
      
      > [Sun Jan 25 03:12:02 2015 (548) - hostname (rank 0)]   Preparing to run...
      
      > [Sun Jan 25 03:12:02 2015 (1016) - hostname (rank 1)]  Finished loading training data.
      
      > [Sun Jan 25 03:12:02 2015 (1017) - hostname (rank 1)]  Preparing to run...
      
      > [Sun Jan 25 03:12:02 2015 (1017) - hostname (rank 0)]  Sending local atom 0 for initialization...
      
      > [Sun Jan 25 03:12:02 2015 (1022) - hostname (rank 0)]  Running algorithm...
      
      > [Sun Jan 25 03:12:02 2015 (1027) - hostname (rank 1)]  Running algorithm...
      
      > [Sun Jan 25 03:12:02 2015 (1027) - hostname (rank 1)]  Sending local atom 4615 on iteration 0...
      
      > [Sun Jan 25 03:12:02 2015 (1031) - hostname (rank 0)]  Sending local atom 2322 on iteration 1...
      
      > [Sun Jan 25 03:12:02 2015 (1036) - hostname (rank 0)]  Sending local atom 3337 on iteration 2...
      
      > [Sun Jan 25 03:12:02 2015 (1043) - hostname (rank 0)]  Sending local atom 1650 on iteration 3...
      
      > [Sun Jan 25 03:12:02 2015 (1049) - hostname (rank 0)]  Sending local atom 2336 on iteration 4...
      
      > [Sun Jan 25 03:12:02 2015 (1055) - hostname (rank 0)]  Sending local atom 1357 on iteration 5...
      
      > [Sun Jan 25 03:12:02 2015 (1061) - hostname (rank 1)]  Sending local atom 6039 on iteration 6...
      
      > ...


 * Observe log file (found in the 'results' sub-directory) for optimization progress; format here is: *(1) iteration number, (2) seconds per iteration, (3) number of non-zero entries, (4) communication cost, (5) duality gap, (6) objective value*:

      >1,8,1,240,7.91043,3.33333
      
      >2,3,2,360,8.28475,3.33333
      
      >3,4,3,480,2.81655,1.1327
      
      >4,6,4,600,2.24387,0.840934
      
      >5,6,5,720,1.46405,0.566311
      
      >6,5,6,840,1.29286,0.474762
      
      >7,6,7,960,0.938121,0.374195
      
      >8,3,8,1080,0.956039,0.34767
      
      >9,5,9,1200,0.743864,0.284809
      
      >10,7,10,1320,0.723161,0.259852
      
      > ...


#####TODO:
- line-search to be implemented.
- file-io should be moved to common file
- better error checking on program arguments and input data.
- improve makefile definition
- change input data format from text to binary (for space efficiency)
- frankwolfe\_lasso does not print last attribute to log file (objective value).
- provide functionality to write out solution (index of non-zero alphas with respective values)

-----------------

#####For more information, refer to: 

A. Bellet, Y. Liang, A. Bagheri Garakani, M.-F. Balcan and F. Sha.
A Distributed Frank-Wolfe Algorithm for Communication-Efficient Sparse Learning.
SIAM International Conference on Data Mining (SDM), 2015.
