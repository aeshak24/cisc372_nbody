#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
__global__
        void add(float**accels,float* hPos,float* mass);

__global__
        void sumcolumns( float **accels, vector3 *hPos, vector3 *hVel);
extern vector3 **accels;
extern vector3 *values;
//make cuda malloc, for loop w i, 
//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	dim3 size(16,16,3);
		int a= ((NUMENTITIES+15)/16);
	dim3 bk(a,a);
add<<<bk, size>>>(accels, hPos, mass);
		int b= ((NUMENTITIES+255/256);
		sumcolums<<256,b);
}
/*
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i,j,k;
	double x,y,z;
	//first compute the pairwise accelerations.  Effect is on the first argument.
	for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				accels[i * NUMENTITIES + j].x = 0.0;
       				accels[i * NUMENTITIES + j].y = 0.0;
       				accels[i * NUMENTITIES + j].z = 0.0;
				//FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
	accels[i * NUMENTITIES + j].x = accelmag * distance.x / magnitude;
        accels[i * NUMENTITIES + j].y = accelmag * distance.y / magnitude;
        accels[i * NUMENTITIES + j].z = accelmag * distance.z / magnitude;
			//	FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
		*/
	
	
__global__
	void add(float**accels,vector3* hPos,float* mass){
	int i= blockldx.x* blockDim.x + threadldx.x;
	int j= blockldx.y* blockDim.y + threadldx.y;
	int k= threadldx.z;
		if (i==j){
		accels[i][j][k]=0;
		}
	vector3 distance;
	distance[k]= hPos[i][k]-hPos[j][k];
	__syncthreads();
	double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
        double magnitude=sqrt(magnitude_sq);
        double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
	accels[i][j][k]==accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude;
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
}
__global__
	void sumcolumns( float **accels, vector3 *hPos, vector3 *hVel){
	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		overallAccels[i].x = 0.0;
        	overallAccels[i].y = 0.0;
        	overallAccels[i].z = 0.0;
		for (j=0;j<NUMENTITIES;j++){
			overallAccels[i].x += accels[j * NUMENTITIES + i].x;
           		overallAccels[i].y += accels[j * NUMENTITIES + i].y;
           		overallAccels[i].z += accels[j * NUMENTITIES + i].z;
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}
	
}
}
