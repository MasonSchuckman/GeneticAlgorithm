//#ifndef debug
//#define DEBUG 1
//#endif


#include "cuda_runtime.h"
#include "math.h"
#include <math.h>

#include "device_launch_parameters.h"

#include <stdio.h>


void sharedTest();



//This is basically a #define. Needed for compile time understanding of certain variable definitions (particularly those that use numLayers for the size of an array)
constexpr auto numLayers = 3; //Number of layers;

constexpr auto bpt = 1; // Stands for "bots per thread" This allows for easy adjusting of the simulation if we wanna do something funky.

__host__ __device__ void printNet(float* activs, int* layerShapes) {
	printf("Activations:\n");
	int AO = 0; // "activs offset"

	for (int layer = 0; layer < numLayers; layer++) {
		printf("Layer %d, size = %d, AO = %d\n", layer, layerShapes[layer], AO);

		for (int i = 0; i < layerShapes[layer]; i++) {
			printf("%f, ", activs[AO + i]);
		}
		AO += layerShapes[layer];
		printf("\n");
	}
	printf("\n");
}


/**
 * Perform forward propagation of a dense neural network
 *
 * @param input     input data to the network, a float array of size input_size
 * @param weights   weight matrix of the network, a float array of size input_size * output_size
 * @param biases    bias vector of the network, a float array of size output_size
 * @param output    output of the network, a float array of size output_size
 * @param input_size    size of the input data
 * @param output_size   size of the output data
 */
__device__ void forward_propagation(const float* inputs, const float* weights, const float* biases, float* output, int input_size, int output_size)
{
	int stride = blockDim.x;
	int tid = threadIdx.x;
#ifdef DEBUG
	if (threadIdx.x == 0) {
		printf("Biases : ");
		for (int i = 0; i < output_size; i++) {
			printf("%f, ", biases[i]);
		}
		printf("\n");
	}
#endif
	// Initialize output to biases
	for (int i = threadIdx.x; i < output_size; i += stride) {
		output[i] = biases[i];
	}
	
	// Compute dot product of input and weights
#pragma unroll 4
	for (int i = 0; i < input_size; i++) {
		for (int j = tid; j < output_size; j += stride) {
			output[j] += inputs[i] * weights[i * output_size + j];
		}
	}


	

#ifdef DEBUG
	if (threadIdx.x == 0 && blockIdx.x == 0) {

		printf("Activs : ");
		for (int i = 0; i < output_size; i++)
			printf("%f, ", output[i]);
		printf("\n\n");
	}
#endif // DEBUG

	

	//TODO: Look into using different activation functions for different layers. (output should probably be sigmoid, others maybe ReLU)
	// Apply activation function (sigmoid in this case)	
	__syncthreads();
	for (int i = tid; i < output_size; i += stride) 
		output[i] = 1 / (1 + expf(-output[i]));
	
}


// this kernel divides the work into blocks rather than each thread works alone.
// Reason for doing this is to hopefully make better use of cache and reduce memory stalls.
// (Hopefully this will lead to higher FLOPS).
__global__ void simulateShared(const int n, const float* allWeights, const float* allBiases, const int* layerShapes, const float* startingParams, float* output) {

	int gid = threadIdx.x + blockIdx.x * blockDim.x; //global id
	int tid = threadIdx.x; //thread id (within a block)

	int block = blockIdx.x;
	int stride = blockDim.x;

	//prevent OOB errors
	if (block < n) {


		//calc the number of neurons per bot.
		int totalNeurons = 0;
		int totalWeights = 0;
		for (int i = 0; i < numLayers; i++) {
			if (i != numLayers - 1)
				totalWeights += layerShapes[i] * layerShapes[i + 1];
			totalNeurons += layerShapes[i];
		}

		//shared mem layout is w1,w2...,w_bpt,b1,b2,...,b_bpt,a_1,a_2,...,a_bpt
		//declare our block of shared memory
		extern __shared__ float s[];

		//split our shared memory block into the weights, biases, and activations
		float* weights = s;
		float* biases = weights + totalWeights * bpt;
		float* activations = biases + totalNeurons * bpt;

#ifdef DEBUG
		printf("Weights = %p\n", weights);
		printf("biases  = %p\n", biases);
		printf("activs  = %p\n", activations);
#endif

		//Copy this block's weights and biases to the shared arrays.
		for (int i = 0; i < totalWeights * bpt; i += stride) {
			weights[i] = (allWeights)[block * totalWeights + i];
		}
		for (int i = 0; i < totalNeurons * bpt; i += stride) {
			biases[i] = (allBiases)[block * totalNeurons + i];
		}


		//Seperate the bot(s) data
		const float* ws[bpt]; //abbreviation for "bot weights"
		const float* bs[bpt]; //abbreviation for "bot biases"
		float* activs[bpt]; //abbreviation for "bot activations"



		//Populate the arrays created above
		for (int i = 0; i < bpt; i++) {
			ws[i] = weights + totalWeights * i;
			bs[i] = biases + totalNeurons * i;
			activs[i] = activations + totalNeurons * i;
		}
		__syncthreads();

		int maxIters = 500;
		bool finished = false;

		int iter = 0; //current timestep of simulation we're on

		//run the simulation loop.
		while (!finished) {
			//Determine inputs for the bot(s)
			for (int i = 0; i < bpt; i++) {
				for (int j = tid; j < layerShapes[0]; j += stride) {
					//This line is a placeholder for now.
					activs[i][j] = 0.5f;
				}
			}


			//It's important to remember that activs and nns are essentially 2d arrays. That's why indexing them is tricky and weird.
			//Poll the NN for actions.
			for (int bot = 0; bot < bpt; bot++) {
				// All of these offsets are to account for the multiple layers in the network.
				int WO = 0; // weights offset
				int BO = 0; // biases offset
				int AO = 0; // activs offset
				int numBiases;
				int numWeights;
				for (int layer = 0; layer < numLayers - 1; layer++) {
					numBiases = layerShapes[layer];
					numWeights = numBiases * layerShapes[layer + 1];
#ifdef DEBUG
					if (tid == 0) {
						printf("Weights of layer %d:\n", layer);
						for (int k = 0; k < numBiases; k++) {
							for (int l = 0; l < layerShapes[layer + 1]; l++) {
								printf("%f, ", (nns[i] + LO + numBiases)[k * layerShapes[layer + 1] + l]);
							}
							printf("\n");
						}
						int cc = 0;
						for (int k = 0; k < numWeights; k++) {
							printf("%f, ", (nns[i] + LO + numBiases)[k]);
							cc++;
							if (cc % layerShapes[layer + 1] == 0)
								printf("\n");
						}
						printf("\n");
					}
					
#endif
					//forward_propagation(float* input, float* weights, float* biases, float* output, int input_size, int output_size)
					forward_propagation(activs[bot] + AO, ws[bot] + WO, bs[bot] + numBiases + BO, activs[bot] + AO + numBiases, numBiases, layerShapes[layer + 1]);

					AO += numBiases;
					WO += numWeights;
					BO += numBiases;
				}
			}

			//update simulation/game state based on bot actions




			//do simulation/game logic



			//if(checkWinCondition(<something>)
			//	finished = true;

			iter++;
			if (iter >= maxIters) {
				finished = true;
			}
		}
#ifdef DEBUG
		if (tid == 0 && blockIdx.x == 0) {
			printf("Activations:\n");
			int AO = 0; // "activs offset"
			for (int layer = 0; layer < numLayers; layer++) {
				printf("Layer %d, size = %d, AO = %d\n", layer, layerShapes[layer], AO);
				for (int i = 0; i < layerShapes[layer]; i++) {
					printf("%f, ", activs[0][AO + i]);
				}
				AO += layerShapes[layer];
				printf("\n");
			}
			printf("\n");
		}
#endif
	}
	return;
}


/*
* Note, I'll refer to the population as "bots"
* Brief description:
*   Using a dense neural network (NN)
*   There are N threads, corresponding to bpt * N bots. (these N's are lowercase in code)
*   networks memory layout in order:
		bpt * N * (b_ij + w_ij), where i=0...k, where k=numLayers is the number of layers
			and j=0...m, where m is the number of neurons in any layer.

			The effect of this memory layout is that each thread's NN data is contiguous in memory, rather than the data being
			interlaced.
*		Although the input layer doesn't have any biases, we will simply store them and have them set to 0 for code simplicity/cleanness
*
*   layerShapes is an array containing the size of each layer. There are numLayers layers.
*
*	The activations array is essentially scratch paper for all the intermediate steps while performing forward propagation.
*		NOTE: This scratch paper includes the inputs for each bot. This is to make the code for foward propagation cleaner.
*
*   Output will likely be an array of 2N length, with each entry corresponding to a bot's score. The host (CPU) will then
*       parse through the output and determine winners.
*
*/


//Defining the interface
using game_logic_func = int (*) (int);

//Actual function def here
__device__ int game1(int input){
	printf("in game\n");
	return 0;
}

// Required for functional pointer argument in kernel function
// Static pointers to device functions
__device__ game_logic_func game1_d = game1;



__global__ void game_kernel(game_logic_func game_logic) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //output[tid] = game_logic(input[tid]);
	printf("%d\n", (*game_logic)(tid));	
	return;
}

void functionPointerTest(){


	cudaSetDevice(0);
	game_logic_func game1_h;
	cudaMemcpyFromSymbol(&game1_h, game1_d, sizeof(game_logic_func));

	game_kernel<<<2, 32>>>(game1_h);
	cudaDeviceSynchronize();

	printf("done\n");
}

int main()
{
	//functionPointerTest();

	sharedTest();


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	
	return 0;
}

void sharedTest() {
	
	//int numThreads = 32;
	int totalBots =	1024 * 32;
	int numStartingParams = 1;
	int* layerShapes_h = new int[numLayers];
	float* startingParams_h = new float[numStartingParams];
	float* output_h = new float[totalBots];
	float* weights_h;
	float* biases_h;

	int numConnections = 0;
	int numNeurons = 0;

	// Determine the network configuration
	layerShapes_h[0] = 8;
	layerShapes_h[1] = 32;
	layerShapes_h[2] = 8;

	// Calculate how many connections and neurons there are based on layerShapes_h so we can create the networks_h array.
	for (int i = 0; i < numLayers; i++) {
		if(i != numLayers - 1)
			numConnections += layerShapes_h[i] * layerShapes_h[i + 1]; //effectively numWeights
		numNeurons += layerShapes_h[i];	//effectively numBiases
	}
	int botNetSize = (numConnections + numNeurons); //how many indices a single bot uses in the networks_h array.
	weights_h = new float[numConnections * totalBots];
	biases_h = new float[numNeurons * totalBots];


	printf("Total network size = %d KB\n", numConnections * sizeof(float) / (2 << 10));
	
	//initialize networks_h with random stuff for testing.

	for (int i = 0; i < totalBots; i++) {
		//printf("bot %d\n", i);
		int WO = 0;
		int BO = 0;
		for (int j = 0; j < numLayers; j++) {		
			//printf("\tlayer %d\n", j);
			for (int k = 0; k < layerShapes_h[j]; k++) {
				//printf("\t\tNode %d: ", k);
				//set the biases
				if (j == 0) {
					//input layer biases are 0
					biases_h[i * numNeurons + BO + k] = 0;
				}
				else {
					//other layers get a bias = layer number.
					biases_h[i * numNeurons + BO + k] = j;
				}
				//printf("bias = %f, weights: ", networks_h[i * botNetSize + LO + k]);
				if (j != numLayers - 1) {
					for (int l = 0; l < layerShapes_h[j + 1]; l++) {
						//set the weights. all layers get a weight of layerNum+1
						weights_h[i * numConnections + WO + k * layerShapes_h[j + 1] + l] = j + 1;
						//printf("%f, ", networks_h[i * botNetSize + LO + layerShapes_h[j] + k * layerShapes_h[j + 1] + l]);
					}
				}
				
				//printf("\n");
				
			}
			if (j != numLayers - 1) {
				BO += layerShapes_h[j];
				WO += layerShapes_h[j] * layerShapes_h[j + 1];
			}
			//printf("\n");
		}
		//printf("\n");
	}

	//printNet(networks_h, layerShapes_h); //this doesn't work I think

	//Create pointers for device data
	
	int* layerShapes_d;

	float* startingParams_d;
	float* output_d;
	float* weights_d;
	float* biases_d;


	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&layerShapes_d, numLayers * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

	cudaMalloc((void**)&startingParams_d, numStartingParams * sizeof(float));
	cudaMalloc((void**)&output_d, totalBots * sizeof(float));
	cudaMalloc((void**)&weights_d, numConnections * totalBots * sizeof(float));
	cudaMalloc((void**)&biases_d, totalBots * numNeurons * sizeof(float));


	//copy data over to GPU
	cudaMemcpy(layerShapes_d, layerShapes_h, numLayers * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(startingParams_d, startingParams_h, numStartingParams * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(output_d, output_h, totalBots * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(weights_d, weights_h, totalBots * numConnections * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(biases_d, biases_h, totalBots * numNeurons * sizeof(float), cudaMemcpyHostToDevice);



	int tpb = 32; //threads per block
	int numBlocks = (totalBots / bpt);
	printf("Num blocks = %d\n", numBlocks);

	int sharedMemNeeded = (numConnections + numNeurons * 2) * bpt;
	printf("Shared mem needed per block = %d KB\n", sharedMemNeeded * sizeof(float) / (2 << 10));
	// Launch a kernel on the GPU with one thread for each element.
	//__global__ void simulateShared(const int n, const float* allWeights, const float* allBiases, const int* layerShapes, const float* startingParams, float* output)
	simulateShared <<<numBlocks, tpb, sharedMemNeeded * sizeof(float)>>> (numBlocks, weights_d, biases_d, layerShapes_d, startingParams_d, output_d);


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching simulate!\n", cudaStatus);
		goto Error;
	}


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(output_h, output_d, totalBots * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//clean up memory.
Error:
	cudaFree(layerShapes_d);
	cudaFree(startingParams_d);
	cudaFree(output_d);
	cudaFree(weights_d);
	cudaFree(biases_d);

	
	delete[] layerShapes_h;
	delete[] startingParams_h;
	delete[] output_h;
	delete[] weights_h;
	delete[] biases_h;


}