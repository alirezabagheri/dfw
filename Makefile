# MAKEFILE

FLAG_OPT = -O3 -D NDEBUG -D BOOST_UBLAS_NDEBUG
FLAG_INCL = -I/usr/include/boost/  -lboost_filesystem

# For debugging uncomment below
#FLAG_OPT = -g -D DEBUG_ENABLED -Wall

main: frankwolfe_svm frankwolfe_svm_commdrop frankwolfe_lasso

frankwolfe_svm:
	mpic++ frankwolfe_svm.cpp -std=c++0x ${FLAG_OPT} ${FLAG_INCL} -o frankwolfe_svm

frankwolfe_svm_commdrop:
	mpic++ frankwolfe_svm_commdrop.cpp -std=c++0x ${FLAG_OPT} ${FLAG_INCL} -o frankwolfe_svm_commdrop

frankwolfe_lasso:
	mpic++ frankwolfe_lasso.cpp -std=c++0x ${FLAG_OPT} ${FLAG_INCL} -o frankwolfe_lasso

clean:
	rm -f frankwolfe_svm frankwolfe_svm_commdrop frankwolfe_lasso
