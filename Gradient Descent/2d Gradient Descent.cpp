#include<stdio.h>
#include<stdlib.h>
#include<math.h>
double function(double x, double y)
{
	return x*x - 4*x+2 + y*y - 4*y + 2;
}

double dfx(double x0, double x1, double y0)
{
	return (function(x1, y0) - function(x0, y0))/(x1-x0);
}

double dfy(double x1, double y0, double y1)
{
	return (function(x1, y1) - function(x1, y0))/(y1-y0);
}

void twodGD(double x0, double y0, unsigned int max_iters, double lambda, double error)
{
	double x1 = x0 + 0.1;
	double y1 = y0 + 0.1;
    unsigned int iters = 0;
    double tempx, tempy ,x_error = error+1, y_error = error+1;
    while(error < x_error && error < y_error && iters < max_iters) {
        tempx = x1;
        x1 -= dfx(tempx, x0, y0) * lambda;
	    x_error = fabs(x1-tempx);
	    tempy = y1;
        y1 -= dfy(x1, tempy, y0) * lambda;
	    y_error = fabs(y1-tempy);
        printf("\nx_error %f, y_error %f\n", x_error, y_error);
        iters++;
        x0 = tempx;
        y0 = tempy;
    }
    printf("\n iterators times %d\n", iters);
    printf("%f	%f\n",x1, y1);
}

int main() {
    double lambda = 0.1, error = 10e-8;
    int max_iters = 1000;
    twodGD(10.0, 20.0, max_iters, lambda, error);
    return 0;
}
