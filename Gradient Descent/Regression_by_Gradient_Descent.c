#include<stdio.h>
#include<stdlib.h>
#include<math.h>
void LRGD(int max_iters, double lambda, double y[], double x[], int datalength)
//Linear Regression by Gradient Descent
{
	double b = 0, a = 0, tempb = 0, tempa = 0;
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

void PRGD(int max_iters, double lambda, double y[], double x[], int datalength, int degree)
//Polynomial Regression by Gradient Descent
{
	double coef[degree+1] = {0}, tempcoef[degree+1] = {0}, temp = 0;
	//double coef[degree+1] = {0}, tempcoef[degree+1], temp = 0;
    int iters = 0;
    double totalData = (double) datalength;
    while(iters < max_iters)
    {
        for(int i=0;i<datalength;i++)
        {
        	for(int j=0;j<degree+1;j++) temp += coef[j]*pow(x[i],j);
        	for(int j=0;j<degree+1;j++) tempcoef[j] -= (-2/totalData)*(y[i]-temp)*(pow(x[i],j));
        	temp = 0;
        }
        for(int j=0;j<degree+1;j++) coef[j] = lambda*tempcoef[j];
        iters++;
        if(iters % 100000 == 0 ) printf("\n iterators times %d\n", iters);
    }
	printf("y = ");
    for(int j=degree;j>=0;j--) 
    {
		if(j == 1) printf("%fx+",coef[j]);
		else if (j == 0) printf("%fx",coef[j]);
		else printf("%fx^%d+",coef[j], j);
	}
}

int main() {
    int datalength = 100, i, degree = 2;
    double x[datalength],y[datalength];
    double lambda = 0.0001;
    int max_iters = 1000;

    FILE *fp;
    fp = fopen("data.txt","r");
    for (i=0;i<datalength;i++)fscanf(fp,"%lf,%lf\n",&x[i],&y[i]);
    fclose(fp);
    LRGD(max_iters, lambda, y, x, datalength);
    //PRGD(max_iters, lambda, y, x, datalength, degree);
    return 0;
}
