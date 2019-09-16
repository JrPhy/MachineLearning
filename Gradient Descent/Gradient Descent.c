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

double GD(double x0, unsigned int max_iters, double lambda, double error)
{
	double x1 = x0 + 0.1;
    unsigned int iters = 0;
    double temp,c_error = error+1;
    while(error < c_error && iters < max_iters) {
        temp = x1;
        x1 -= dfx(temp, x0) * lambda;
	    c_error = fabs(x1-temp);
        printf("\nc_error %f\n", c_error);
        iters++;
        x0 = temp;
    }
    printf("\n iterators times %d\n", iters);
    return x1;
}

void LRGD(int max_iters, double lambda, double y[], double x[], int datalength)
//Linear Regression by Gradient Descent
{
	double b = 0, a = 0, tempb, tempa;
    int iters = 0;
    double totalData = (double) datalength;

    while(iters < max_iters)
    {
        for(int i=0;i<datalength;i++)
        {
            tempb -= (1/totalData)*(y[i]-(b*x[i]+a))*(-2*x[i]);
            tempa -= (1/totalData)*(y[i]-(b*x[i]+a))*(-2);
        }
        b = lambda*tempb;
        a = lambda*tempa;
        iters++;
        if(iters % 100 == 0 ) printf("\n iterators times %d\n", iters);
    }

    printf("y = %fx+%f\n",b,a);
}

int main() {
    int datalength = 100, i;
    double x[datalength],y[datalength];
    double lambda = 0.0001;
    int max_iters = 1000;

    FILE *fp;
    fp = fopen("data.csv","r");
    for (i=0;i<datalength;i++)fscanf(fp,"%lf,%lf\n",&x[i],&y[i]);
    fclose(fp);
    //LRGD(max_iters, lambda, y, x, datalength);
    //LUD(x, y, 100, 2);
    twoByTwoInverse(x, y, 100, 2);
    //printf("The local minimum is: %f, The value is: %f\n", x1,function(x1));
    return 0;
}
