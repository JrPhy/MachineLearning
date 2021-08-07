#include<stdio.h> 
#include<stdlib.h>
#include<math.h> 
#define DIMENSION 3
void UpdateWeight(float w[], float x[], int y)
{
	int i;
	for(i=0; i<DIMENSION; i++) w[i] = w[i] + x[i]*y;
}

double InnerProduct(float w[], float x[])
{
	int i;
	double result = 0;
	for(i=0; i<DIMENSION; i++) result += w[i]*x[i];
	return result;
}

int Sign(double x)
{
	//int y;
	if(x > 0) return 1;
	/*else if(fabs(x) < 1e-6) 
	{
		y = 0;
		return y;
	}*/
	else return -1;
}

void PLA(float data[][DIMENSION], int *label, int n)
{
	int correctNum = 0, index = 0, j, y, step = 0, i;
    bool isFinished = 0;
    float w[DIMENSION] = {0}, x[DIMENSION];
    while(!isFinished)
	{
		correctNum = 0;
		for(j=0; j<n; j++)
		{
			for(i=0; i<DIMENSION; i++) printf("%f	",w[i]);
			printf("\n");
			x[0] = 1.0;
			for(i=1; i<DIMENSION; i++) x[i] = data[j][i-1];
			y = label[j]; 
			if((int)Sign(InnerProduct(w,x)) != y ) 
			{
				UpdateWeight(w, x, y);
				step++;
				printf("%d\n",step);
				system("pause");
        	}
        	else correctNum++;
		}
		
		if(index == n-1)index = 0;
        else index++;
        if(correctNum == n)isFinished = 1;
	}
	printf("%d\n",step);
	for(i=0; i<DIMENSION; i++) printf("%f,",w[i]);
	printf("\n");
}

int main(){
	FILE *fp;
	int n = 4, label[n], j;
	float data[n][DIMENSION];
	fp = fopen("test1.txt","r");
	for(j=0; j<n; j++) fscanf(fp,"%f,%f,%d",&data[j][0], &data[j][1], &label[j]);
	//for(j=0; j<n; j++) fscanf(fp,"%f %f %f %f	%d",&data[j][0], &data[j][1], &data[j][2], &data[j][3], &label[j]);
	//for(j=0; j<n; j++) printf("%f,%f,%f,%f,%d\n",data[j][0], data[j][1], data[j][2], data[j][3], label[j]);
	fclose(fp);
	PLA(data, label, n);
	return 0;
}
