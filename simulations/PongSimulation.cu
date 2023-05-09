#include "PongSimulation.cuh"
#include <random>
#define WIDTH 640.0f
#define HEIGHT 480.0f
#define PADDLE_WIDTH 10.0f
#define PADDLE_HEIGHT 50.0f
#define BALL_RADIUS 10.0f
#define BALL_SPEED 7.0f
#define PADDLE_SPEED 5.0f
#define SPEED_UP_RATE 1.00f // Ball will increase in speed by x % after every paddle hit

// Compile command:
// nvcc -rdc=true -lineinfo -o runner .\biology\Genome.cpp .\Simulator.cu .\Runner.cu .\Kernels.cu .\simulations\BasicSimulation.cu .\simulations\TargetSimulation.cu .\simulations\MultibotSimulation.cu .\simulations\AirHockeySimulation.cu .\simulations\PongSimulation.cu
/*
gamestate[0]        // ball x
gamestate[1]        // ball y
gamestate[2]        // ball vx
gamestate[3]        // ball vy
gamestate[4]        // left paddle x
gamestate[5]        // left paddle y
gamestate[6]        // right paddle x
gamestate[7]        // right paddle y
gamestate[8]        // iter
gamestate[9]        // left score
gamestate[10]       // right score
gamestate[11]       // generation number
*/
extern __device__ float rng(float a, float b, unsigned int seed);


// The starting parameters are the initial positions and velocities of the ball and the paddles
__host__ void PongSimulation::getStartingParams(float *startingParams)
{
    static int iterationsCompleted = 0;

    startingParams[0] = WIDTH / 2;  // ball x
    startingParams[1] = HEIGHT / 2; // ball y
    startingParams[2] = BALL_SPEED; // ball vx
    if (iterationsCompleted % 2 == 0)
        startingParams[2] *= -1;

    startingParams[3] = (((double)rand() / RAND_MAX) - 0.5) * BALL_SPEED * 2; // ball vy
    startingParams[4] = PADDLE_WIDTH / 2;                                     // left paddle x
    startingParams[5] = HEIGHT / 2;                                           // left paddle y
    startingParams[6] = PADDLE_WIDTH / 2 + WIDTH - PADDLE_WIDTH;              // right paddle x
    startingParams[7] = HEIGHT / 2;                                           // right paddle y
    startingParams[8] = iterationsCompleted;
    iterationsCompleted++;
}

// The gamestate is an array of floats that stores the current positions and velocities of the ball and the paddles
__device__ void PongSimulation::setupSimulation(const float *startingParams, float *gamestate)
{
    int tid = threadIdx.x;

    if (tid < 8)
        gamestate[tid] = startingParams[tid];
    if (tid < 3)
        gamestate[tid + 8] = 0;
    if(tid == 0)
        gamestate[11] = startingParams[8]; // Generation number

    __syncthreads();
}

__constant__ float Limits[8] = {WIDTH, HEIGHT, BALL_SPEED, BALL_SPEED, WIDTH, HEIGHT, WIDTH, HEIGHT};

// The activations are the inputs to the neural networks that control the paddles. They are the normalized positions and velocities of the ball and the paddles
__device__ void PongSimulation::setActivations(float *gamestate, float **activs, int iter)
{
    int tid = threadIdx.x;

    // Have the first 8 threads update bot 1's activations
    // if (tid < 8)
    //     activs[0][tid] = gamestate[tid] / Limits[tid];

    // // Have thread 0 update bot 2's activations (since it's more complicated due to reversing some things)
    // if (tid == 0)
    // {
    //     activs[0][0] = abs(gamestate[4] - gamestate[0]) / WIDTH;

    //     activs[1][0] = abs(gamestate[6] - gamestate[0]) / WIDTH;       // ball x
    //     activs[1][1] = gamestate[1] / HEIGHT;      // ball y
    //     activs[1][2] = -gamestate[2] / BALL_SPEED; // ball vx (inverted for right paddle)
    //     activs[1][3] = gamestate[3] / BALL_SPEED;  // ball vy
    //     activs[1][4] = gamestate[6] / WIDTH;       // right paddle x
    //     activs[1][5] = gamestate[7] / HEIGHT;      // right paddle y
    //     activs[1][6] = gamestate[4] / WIDTH;       // left paddle x
    //     activs[1][7] = gamestate[5] / HEIGHT;      // left paddle y
    // } 

    if (tid == 0)
    {
        float rand = 0;
        //float randomMagnitude = 400.0f / logf(gamestate[11] + 2.0f) + 400.0f / logf((float)(iter + 2));
        float randomMagnitude = 400.0f / logf((float)(iter + 2));

        for (int i = 0; i < 4; i++)
        {
            if (i == 0)
            {
                activs[0][i] = fabsf(gamestate[4] - gamestate[0]) / Limits[i];
                activs[1][i] = fabsf(gamestate[6] - gamestate[0]) / Limits[i];
            }
            activs[0][i] = gamestate[i] / Limits[i];
            activs[1][i] = gamestate[i] / Limits[i];
        }
        activs[1][2] *= -1;

        rand = rng(-randomMagnitude, randomMagnitude, tid + iter + blockIdx.x ^ (int)gamestate[11]);        

        activs[0][4] = 0;
        activs[0][5] = gamestate[5] / HEIGHT; // left paddle y
        activs[0][6] = 1;
        activs[0][7] = (gamestate[7] + rand)  / HEIGHT; // right paddle y //skew the other bot's pos a little

        rand = rng(-randomMagnitude, randomMagnitude, tid + iter + blockIdx.x ^ (int)gamestate[11]);        

        activs[1][4] = 0;
        activs[1][5] = gamestate[7] / HEIGHT; // left paddle y
        activs[1][6] = 1;
        activs[1][7] = (gamestate[5] + rand) / HEIGHT; // right paddle y
    }

    __syncthreads();
}

// The actions are the outputs of the neural networks that control the paddles. They are the normalized velocities of the paddles in the y direction
__device__ void PongSimulation::eval(float **actions, float *gamestate)
{
    if (threadIdx.x == 0)
    {
        // if(blockIdx.x == 0 && gamestate[8] > 9990){
        //     printf("Gamestate:\n");
        //     for(int i = 0; i < 8; i++){
        //         printf("%f, ", gamestate[i]);
        //     }
        //     printf("\nActions: %f, %f\n", actions[0][0], actions[1][0]);

        // }
        // Update the ball position and velocity based on physics
        gamestate[0] += gamestate[2]; // ball x += ball vx
        gamestate[1] += gamestate[3]; // ball y += ball vy

        // Check for collisions with walls and paddles and bounce accordingly

        if (gamestate[1] < BALL_RADIUS || gamestate[1] > HEIGHT - BALL_RADIUS)
        {                       // top or bottom wall collision
            gamestate[3] *= -1; // invert ball vy
        }

        // calculate the ball's new vx and vy after a collision with the left paddle
        if (gamestate[0] - BALL_RADIUS <= gamestate[4] + PADDLE_WIDTH &&
            gamestate[1] >= gamestate[5] &&
            gamestate[1] <= gamestate[5] + PADDLE_HEIGHT &&
            gamestate[2] < 0)
        {
            // if(blockIdx.x == 0 && gamestate[8] > 500)
            //     printf("Hit left paddle on iter %f\n", gamestate[8]);
            gamestate[2] = -gamestate[2] * SPEED_UP_RATE;                                                         // reverse the ball's horizontal direction
            gamestate[3] += (gamestate[1] - gamestate[5] - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED; // adjust the ball's vertical speed based on where it hit the paddle

            // Update the ball position and velocity based on physics
            gamestate[0] += gamestate[2]; // ball x += ball vx
            gamestate[1] += gamestate[3]; // ball y += ball vy
        }

        // calculate the ball's new vx and vy after a collision with the right paddle
        if (gamestate[0] + BALL_RADIUS >= gamestate[6] &&
            gamestate[1] >= gamestate[7] &&
            gamestate[1] <= gamestate[7] + PADDLE_HEIGHT &&
            gamestate[2] > 0)
        {
            // if(blockIdx.x == 0 && gamestate[8] > 500)
            //     printf("Hit right paddle on iter %f\n", gamestate[8]);

            gamestate[2] = -gamestate[2] * SPEED_UP_RATE;                                                         // reverse the ball's horizontal direction
            gamestate[3] += (gamestate[1] - gamestate[7] - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED; // adjust the ball's vertical speed based on where it hit the paddle

            // Update the ball position and velocity based on physics
            gamestate[0] += gamestate[2]; // ball x += ball vx
            gamestate[1] += gamestate[3]; // ball y += ball vy
        }

        // Update the paddle positions based on actions and physics
        gamestate[5] += fminf(1.0, fmaxf(-1.0, actions[0][0])) * PADDLE_SPEED; // left paddle y += action * paddle speed
        gamestate[7] += fminf(1.0, fmaxf(-1.0, actions[1][0])) * PADDLE_SPEED; // right paddle y += action * paddle speed

        // Clamp the paddle positions to the screen boundaries
        if (gamestate[5] < PADDLE_HEIGHT / 2)
        {
            gamestate[5] = PADDLE_HEIGHT / 2;
        }
        if (gamestate[5] > HEIGHT - PADDLE_HEIGHT / 2)
        {
            gamestate[5] = HEIGHT - PADDLE_HEIGHT / 2;
        }
        if (gamestate[7] < PADDLE_HEIGHT / 2)
        {
            gamestate[7] = PADDLE_HEIGHT / 2;
        }
        if (gamestate[7] > HEIGHT - PADDLE_HEIGHT / 2)
        {
            gamestate[7] = HEIGHT - PADDLE_HEIGHT / 2;
        }

        gamestate[8]++;
    }

    __syncthreads();
}

// The simulation is finished when the ball goes out of bounds on either side
__device__ int PongSimulation::checkFinished(float *gamestate)
{
    if (gamestate[0] < 0 || gamestate[0] > WIDTH)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

// The output is an array of floats that stores the score of each paddle. The score is 1 if the paddle won, -1 if it lost, and 0 if it tied

__device__ void PongSimulation::setOutput(float *output, float *gamestate, const float *startingParams_d)
{
    if (threadIdx.x == 0)
    {
        // if(gamestate[8] > 4990)
        //     printf("Timeout %d\n", blockIdx.x);
        // output[blockIdx.x * 2 + 0] = gamestate[9];
        // output[blockIdx.x * 2 + 1] = gamestate[10];

        // if (gamestate[0] < 0)
        // { // left paddle lost
        //     output[blockIdx.x * 2 + 0] = -1 + gamestate[8] / 2;
        //     output[blockIdx.x * 2 + 1] = 1 + gamestate[8];
        // }
        // else if (gamestate[0] > WIDTH)
        // {
        //     // right paddle lost
        //     output[blockIdx.x * 2 + 0] = 1 + gamestate[8];
        //     output[blockIdx.x * 2 + 1] = -1 + gamestate[8] / 2;
        // }
        // else
        // { // tie
        //     output[blockIdx.x * 2 + 0] = 1 + gamestate[8];
        //     output[blockIdx.x * 2 + 1] = 1 + gamestate[8];
        // }

        // TESTING NO TIES:
        if (gamestate[0] < 0)
        { // left paddle lost
            output[blockIdx.x * 2 + 0] = -1;
            output[blockIdx.x * 2 + 1] = 1;
        }
        else // if (gamestate[0] > WIDTH)
        {
            // right paddle lost
            output[blockIdx.x * 2 + 0] = 1;
            output[blockIdx.x * 2 + 1] = -1;
        }
        // else
        // { // tie
        //     output[blockIdx.x * 2 + 0] = 1;// + gamestate[8];
        //     output[blockIdx.x * 2 + 1] = 1;// + gamestate[8];
        // }
    }
}

// The ID is a unique identifier for this simulation type
__host__ int PongSimulation::getID()
{
    return 4; // just a random number
}