# include <stdio.h>

__global__ void hello(){
	printf("gpu:hello cuda world\n");
}

int main(){
	printf("cpu:hello from cpu\n");
	hello<<<1,8>>>();
	cudaDeviceSynchronize();
	
	return 0;
}
