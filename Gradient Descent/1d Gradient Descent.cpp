#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double function(double x)
{
	return x*x - 4*x+2;
}

double dfx(double x0, double x1)
{
	return (function(x1) - function(x0))/(x1-x0);
}

double onedGD(double x0, unsigned int max_iters, double lambda, double error)
{
	double x1 = x0 + 0.1;
    unsigned int iters = 0;
    double temp,c_error = error+1;
    while(error < c_error && iters < max_iters) {
        temp = x1;
        x1 -= dfx(temp, x0) * lambda;
	    c_error = fabs(x1-temp);
        iters++;
        x0 = temp;
    }
    printf("\n iterators times %d\n", iters);
    return x1;
}

int main() {
    double lambda = 0.1, error = 10e-8;
    int max_iters = 1000;
    onedGD(10.0, max_iters, lambda, error);
    printf("%f\n",onedGD(10.0, max_iters, lambda, error));
    return 0;
}
