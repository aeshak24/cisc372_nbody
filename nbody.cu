#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"

// represents the objects in the system.  Global variables
vector3 *hVel;//, *d_hVel;
vector3 *hPos;//, *d_hPos;
//vector3 *d_values, d_accels, *d_sum;
//double *d_mass;
double *mass;
//vector3 *t_accels;
//vector3 *values;
vector3 **accels;
//initHostMemory: Create storage for numObjects entities in our system
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the hVel, hPos, and mass global variables
void initHostMemory(int numObjects)
{
	//hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
	cudaMallocManaged(&hVel,sizeof(vector3)*numObjects);
	//hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
	cudaMallocManaged(&hPos,sizeof(vector3)*numObjects);
	//mass = (double *)malloc(sizeof(double) * numObjects);
	cudaMallocManaged(&mass,sizeof(double)*numObjects);
}

//freeHostMemory: Free storage allocated by a previous call to initHostMemory
//Parameters: None
//Returns: None
//Side Effects: Frees the memory allocated to global variables hVel, hPos, and mass.
void freeHostMemory()
{
	cudaFree(hVel);
	cudaFree(hPos);
	cudaFree(mass);
}

//planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
//Parameters: None
//Returns: None
//Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
void planetFill(){
	int i,j;
	double data[][7]={SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE};
	for (i=0;i<=NUMPLANETS;i++){
		for (j=0;j<3;j++){
			hPos[i][j]=data[i][j];
			hVel[i][j]=data[i][j+3];
		}
		mass[i]=data[i][6];
	}
}

//randomFill: FIll the rest of the objects in the system randomly starting at some entry in the list
//Parameters: 	start: The index of the first open entry in our system (after planetFill).
//				count: The number of random objects to put into our system
//Returns: None
//Side Effects: Fills count entries in our system starting at index start (0 based)
void randomFill(int start, int count)
{
	int i, j ;
	for (i = start; i < start + count; i++)
	{
		for (j = 0; j < 3; j++)
		{
			hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
			hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
			mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
		}
	}
}

//printSystem: Prints out the entire system to the supplied file
//Parameters: 	handle: A handle to an open file with write access to prnt the data to
//Returns: 		none
//Side Effects: Modifies the file handle by writing to it.
void printSystem(FILE* handle){
	int i,j;
	for (i=0;i<NUMENTITIES;i++){
		fprintf(handle,"pos=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hPos[i][j]);
		}
		printf("),v=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hVel[i][j]);
		}
		fprintf(handle,"),m=%lf\n",mass[i]);
	}
}

int main(int argc, char **argv)
{
	//cudaMallocManaged(&values,sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accelstmp=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (int i=0;i<NUMENTITIES;i++)
		cudaMalloc(&accelstmp[i],sizeof(vector3)*NUMENTITIES);
	cudaMalloc(&accels,sizeof(vector3*)*NUMENTITIES);
	cudaMemcpy(accels,accelstmp,sizeof(vector3*)*NUMENTITIES,cudaMemcpyHostToDevice);
	free(accelstmp);
  //      vector3** accelstmp=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
//        for (int i=0;i<NUMENTITIES;i++)
    //            accelstmp[i]=&values[i*NUMENTITIES];
//	cudaMalloc(&accels,sizeof(vector3*)*NUMENTITIES);
//	cudaMemcpy(&accels,accelstmp,sizeof(vector3*)*NUMENTITIES,cudaMemcpyHostToDevice);
//	free(accelstmp);
	clock_t t0=clock();
	int t_now;
	//srand(time(NULL));
	srand(1234);
	initHostMemory(NUMENTITIES);
	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);
	//now we have a system.
	#ifdef DEBUG
	printSystem(stdout);
	#endif
	//cudaMallocManaged(&values, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
       //cudaMallocManaged(&accels,sizeof(vector3*)*NUMENTITIES);
        //for (i=0;i<NUMENTITIES;i++)
        //        accels[i]=&values[i*NUMENTITIES];
	for (t_now=0;t_now<DURATION;t_now+=INTERVAL){
		compute();
	}
	clock_t t1=clock()-t0;
#ifdef DEBUG
	printSystem(stdout);
#endif
	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);
	//cudaFree(values);
//	vector3** t_accels = (vector3**)malloc(sizeof(vector3*) * NUMENTITIES);
  //  cudaMemcpy(t_accels, accels, sizeof(vector3*) * NUMENTITIES, cudaMemcpyDeviceToHost);

    // Free device acceleration matrix
   // for (int i = 0; i < NUMENTITIES; i++) {
     //   cudaFree(t_accels[i]);
   // }

//    free(t_accels);
  //  cudaFree(accels);

	accelstmp=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	//changed here
	cudaMemcpy(accelstmp, accels, sizeof(vector3*)*NUMENTITIES, cudaMemcpyDeviceToHost);
	int i;
	//free device acceleration matrix
	for (i=0;i<NUMENTITIES;i++){
		cudaFree(accelstmp[i]);
}
	free(accelstmp);                   
	cudaFree(accels);
	freeHostMemory();

	}
