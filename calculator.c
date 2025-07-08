#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int plus(int num1, int num2)
{
    return num1+num2;
}

int minus(int num1, int num2)
{
    return num1-num2;
}


double inner_product(double* array1, double* array2, int size)
{
	double result=0;
	for(int i=0 ; i < size ; i++)
	{
		result += array1[i] * array2[i];
		
		//printf("%f * %f = %f\n", array1[i], array2[i], result);
	}
		
	//printf("%f\n", result);
	
	return result;
}

float matrix_product(float* array1, float* array2, int size1, int size2)
{
	float result = 0.0f;
	int seq=0;
	
	//result = malloc(sizeof(double) * size);
	
	for(int i=0 ; i < size1 ; i++)
	{
		for(int j=0 ; j < size2 ; j++)
		{
			seq = i * size2 + j;
			result += array1[seq] * array2[seq];
			
			//printf("%f * %f = %f\n", array1[seq], array2[seq], result);
		}
	}
	
	//printf("\n");
		
	return result;
}