#include "globals.h"


double randn( ) {
    random_device rd;
    mt19937 gen(rd());
    double mean = 0.0, stddev = 1.0, rval; 

    normal_distribution<double> distribution(mean,stddev);

    rval = distribution(gen);
    return rval;
}