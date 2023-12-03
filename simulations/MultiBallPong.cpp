#include "MultiBallPong.h"
#include <random>
#define WIDTH 640.0f
#define HEIGHT 480.0f
#define PADDLE_WIDTH 10.0f
#define PADDLE_HEIGHT 50.0f
#define BALL_RADIUS 5.0f
#define BALL_SPEED 5.0f
#define PADDLE_SPEED 10.0f
#define SPEED_UP_RATE 1.00f // Ball will increase in speed by x % after every paddle hit
#define ball2 12
// Compile command:
// nvcc -rdc=true -lineinfo -o runner .\biology\Genome.cpp .\Simulator.cu .\Runner.cu .\Kernels.cu .\simulations\BasicSimulation.cu .\simulations\TargetSimulation.cu .\simulations\MultibotSimulation.cu .\simulations\AirHockeySimulation.cu .\simulations\PongSimulation2.cu
/*
gamestate[0]        // ball1 x
gamestate[1]        // ball1 y
gamestate[2]        // ball1 vx
gamestate[3]        // ball1 vy
gamestate[4]        // left paddle x
gamestate[5]        // left paddle y
gamestate[6]        // right paddle x
gamestate[7]        // right paddle y
gamestate[8]        // iter
gamestate[9]        // left touches
gamestate[10]       // right touches
gamestate[11]       // generation number

gamestate[12]        // ball2 x
gamestate[13]        // ball2 y
gamestate[14]        // ball2 vx
gamestate[15]        // ball2 vy

gamestate[16]        // left score
gamestate[17]        // right score

gamestate[18] // prev ball left got scored on with
gamestate[19] // prev ball right got scored on with

*/

float LimitsMultiball[8] = {WIDTH, HEIGHT, BALL_SPEED, BALL_SPEED, WIDTH, HEIGHT, WIDTH, HEIGHT};


// The starting parameters are the initial positions and velocities of the ball and the paddles
void MultiBallPong::getStartingParams(float *startingParams)
{
    static int iterationsCompleted = 0;

    startingParams[0] = WIDTH / 2;  // ball x
    startingParams[1] = HEIGHT / 2; // ball y
    startingParams[2] = BALL_SPEED; // ball vx
    // if (iterationsCompleted % 2 == 0)
    if ((double)rand() / RAND_MAX > 0.5)
        startingParams[2] *= -1;

    startingParams[3] = (((double)rand() / RAND_MAX) - 0.5) * BALL_SPEED * 1.2; // ball vy
    startingParams[4] = PADDLE_WIDTH / 2;                                     // left paddle x
    startingParams[5] = HEIGHT / 2;                                           // left paddle y
    startingParams[6] = PADDLE_WIDTH / 2 + WIDTH - PADDLE_WIDTH;              // right paddle x
    startingParams[7] = HEIGHT / 2;                                           // right paddle y
    startingParams[8] = iterationsCompleted;
    iterationsCompleted++;
}

// The gamestate is an array of floats that stores the current positions and velocities of the ball and the paddles
void MultiBallPong::setupSimulation(int tid, int block, const float *startingParams, float *gamestate)
{
    
    for(tid = 0; tid < 8; tid++)
        gamestate[tid] = startingParams[tid];

    for(tid = 0; tid < 3; tid++)
        gamestate[tid + 8] = 0;

    
    gamestate[11] = startingParams[8]; // Generation number

    // Set ball 2's info
    gamestate[ball2 + 0] = startingParams[0];
    gamestate[ball2 + 1] = startingParams[1] - BALL_RADIUS * 2.5;
    gamestate[ball2 + 2] = -startingParams[2];
    gamestate[ball2 + 3] = -startingParams[3];

    gamestate[16] = 0;
    gamestate[17] = 0;
    gamestate[18] = -1;
    gamestate[19] = -1;

    

    
}

// The activations are the inputs to the neural networks that control the paddles. They are the normalized positions and velocities of the ball and the paddles
void MultiBallPong::setActivations(int tid, int block, float *gamestate, float **activs, int iter)
{
        float rand = 0;
        // float randomMagnitude = 400.0f / logf(gamestate[11] + 2.0f) + 400.0f / logf((float)(iter + 2));
        //float randomMagnitude = 400.0f / logf((float)(iter + 2));
        float randomMagnitude = 0;

        for (int i = 0; i < 4; i++)
        {
            if(iter % 2 == 0){
                if (i == 0)
                {
                    // Ball 1
                    activs[0][i] = fabsf(gamestate[4] - gamestate[0]) / LimitsMultiball[i];
                    activs[1][i] = fabsf(gamestate[6] - gamestate[0]) / LimitsMultiball[i];

                    // Ball 2
                    activs[0][i + 6] = fabsf(gamestate[4] - gamestate[ball2]) / LimitsMultiball[i];
                    activs[1][i + 6] = fabsf(gamestate[6] - gamestate[ball2]) / LimitsMultiball[i];
                }
                else
                {
                    // Ball 1
                    activs[0][i] = gamestate[i] / LimitsMultiball[i];
                    activs[1][i] = gamestate[i] / LimitsMultiball[i];

                    // Ball 2
                    activs[0][i + 6] = gamestate[i + ball2] / LimitsMultiball[i];
                    activs[1][i + 6] = gamestate[i + ball2] / LimitsMultiball[i];
                }
            }else{
                if (i == 0)
                {
                    // Ball 1
                    activs[0][i + 6] = fabsf(gamestate[4] - gamestate[0]) / LimitsMultiball[i];
                    activs[1][i + 6] = fabsf(gamestate[6] - gamestate[0]) / LimitsMultiball[i];

                    // Ball 2
                    activs[0][i] = fabsf(gamestate[4] - gamestate[ball2]) / LimitsMultiball[i];
                    activs[1][i] = fabsf(gamestate[6] - gamestate[ball2]) / LimitsMultiball[i];
                }
                else
                {
                    // Ball 1
                    activs[0][i + 6] = gamestate[i] / LimitsMultiball[i];
                    activs[1][i + 6] = gamestate[i] / LimitsMultiball[i];

                    // Ball 2
                    activs[0][i] = gamestate[i + ball2] / LimitsMultiball[i];
                    activs[1][i] = gamestate[i + ball2] / LimitsMultiball[i];
                }
            }
        }
        activs[1][2] *= -1;
        activs[1][ball2 + 2] *= -1;
        // rand = rng(-randomMagnitude, randomMagnitude, tid + iter + blockIdx.x ^ (int)gamestate[11]);

        activs[0][4] = gamestate[5] / HEIGHT; // left paddle y
        // activs[0][5] = (gamestate[7] + rand) / HEIGHT; // right paddle y //skew the other bot's pos a little
        activs[0][5] = 0;
        // rand = rng(-randomMagnitude, randomMagnitude, tid + iter + blockIdx.x ^ (int)gamestate[11]);

        activs[1][4] = gamestate[7] / HEIGHT; // right paddle y
        // activs[1][5] = (gamestate[5] + rand) / HEIGHT; // left paddle y
        activs[1][5] = 0;
    

    
}

// The actions are the outputs of the neural networks that control the paddles. They are the normalized velocities of the paddles in the y direction
void MultiBallPong::eval(int tid, int block, float **actions, float *gamestate)
{
    
        float randDirRate = 0.9;
        int balls = 2;
        int ballIdx[2] = {0, ball2};


        // Update the paddle positions based on actions and physics
        //printf("output : %f %f %f\n", actions[0][0], actions[0][1], actions[0][2]);

        // action 0 = go up, 1 = stay still, 2 = go down
        // for(int bot = 0; bot < 2; bot++){
        //     float max = actions[bot][0];
        //     int choice = 0;
            
        //     for(int action = 1; action < 3; action++){
        //         if(actions[bot][action] > max)
        //         {
        //             max = actions[bot][action];
        //             choice = action;
        //         }
        //     }
        //     //printf("Max val = %f, choice = %d\n", max, choice);
        //     // Update bot's position
        //     gamestate[5 + bot * 2] += (choice - 1) * PADDLE_SPEED; // left paddle y += action * paddle speed

        // }
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



        for (int i = 0; i < balls; i++)
        {
            // Update the ball position and velocity based on physics
            gamestate[ballIdx[i] + 0] += gamestate[ballIdx[i] + 2]; // ball x += ball vx
            gamestate[ballIdx[i] + 1] += gamestate[ballIdx[i] + 3]; // ball y += ball vy

            // Check for collisions with walls and paddles and bounce accordingly

            if (gamestate[ballIdx[i] + 1] < BALL_RADIUS || gamestate[ballIdx[i] + 1] > HEIGHT - BALL_RADIUS)
            {                                    // top or bottom wall collision
                gamestate[ballIdx[i] + 3] *= -1; // invert ball vy
            }

            // calculate the ball's new vx and vy after a collision with the left paddle
            if (gamestate[ballIdx[i] + 0] - BALL_RADIUS <= gamestate[4] + PADDLE_WIDTH &&
                gamestate[ballIdx[i] + 1] >= gamestate[5] &&
                gamestate[ballIdx[i] + 1] <= gamestate[5] + PADDLE_HEIGHT &&
                gamestate[ballIdx[i] + 2] < 0)
            {
                // if(blockIdx.x == 0 && gamestate[8] > 500)
                //     printf("Hit left paddle on iter %f\n", gamestate[8]);
                gamestate[ballIdx[i] + 2] = -gamestate[ballIdx[i] + 2] * SPEED_UP_RATE;                                                         // reverse the ball's horizontal direction
                
                if((double)rand() / RAND_MAX > randDirRate)
                    gamestate[ballIdx[i] + 3] += (gamestate[ballIdx[i] + 1] - gamestate[5] - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED; // adjust the ball's vertical speed based on where it hit the paddle
                else
                    gamestate[ballIdx[i] + 3] = (((double)rand() / RAND_MAX) - 0.5) * BALL_SPEED * 2;
                // Update the ball position and velocity based on physics
                gamestate[ballIdx[i] + 0] += gamestate[ballIdx[i] + 2]; // ball x += ball vx
                gamestate[ballIdx[i] + 1] += gamestate[ballIdx[i] + 3]; // ball y += ball vy

                gamestate[9]++;
            }

            // calculate the ball's new vx and vy after a collision with the right paddle
            if (gamestate[ballIdx[i] + 0] + BALL_RADIUS >= gamestate[6] &&
                gamestate[ballIdx[i] + 1] >= gamestate[7] &&
                gamestate[ballIdx[i] + 1] <= gamestate[7] + PADDLE_HEIGHT &&
                gamestate[ballIdx[i] + 2] > 0)
            {
                // if(blockIdx.x == 0 && gamestate[8] > 500)
                //     printf("Hit right paddle on iter %f\n", gamestate[8]);

                gamestate[ballIdx[i] + 2] = -gamestate[ballIdx[i] + 2] * SPEED_UP_RATE;                                                         // reverse the ball's horizontal direction
                
                if((double)rand() / RAND_MAX > randDirRate)
                    gamestate[ballIdx[i] + 3] += (gamestate[ballIdx[i] + 1] - gamestate[7] - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED; // adjust the ball's vertical speed based on where it hit the paddle
                else
                    gamestate[ballIdx[i] + 3] = (((double)rand() / RAND_MAX) - 0.5) * BALL_SPEED * 2;

                // Update the ball position and velocity based on physics
                gamestate[ballIdx[i] + 0] += gamestate[ballIdx[i] + 2]; // ball x += ball vx
                gamestate[ballIdx[i] + 1] += gamestate[ballIdx[i] + 3]; // ball y += ball vy

                gamestate[10]++;
            }
        }


        
        gamestate[8]++;
    

    
}

// The simulation is finished when the ball goes out of bounds on either side
int MultiBallPong::checkFinished(int tid, int block, float *gamestate)
{
    // Right now we just check if a goal was scored, and if so, we reset that ball back to the center
    int balls = 2;
    int ballIdx[2] = {0, ball2};

    for (int i = 0; i < balls; i++)
    {
        if (gamestate[ballIdx[i] + 0] < 0 || gamestate[ballIdx[i] + 0] > WIDTH){
            // Adjust score
            if (gamestate[ballIdx[i] + 0] < 0){
                gamestate[17]++;

                // De-incentivize letting the same ball score repeatedly
                // if(gamestate[18] == i)
                //     gamestate[16]--;
                gamestate[18] = i;
            }
            else{
                gamestate[16]++;

                // if(gamestate[19] == i)
                //     gamestate[17]--;
                gamestate[19] = i;
            }

            //Reverse x velocity and change vy
            gamestate[ballIdx[i] + 2] *= -1;
            //gamestate[ballIdx[i] + 3] *= -1;
            gamestate[ballIdx[i] + 3] = (((double)rand() / RAND_MAX) - 0.5) * BALL_SPEED * 2;

            //Move ball a bit away from where it hit
            gamestate[ballIdx[i] + 0] += gamestate[ballIdx[i] + 2] * 2;
            gamestate[ballIdx[i] + 1] += gamestate[ballIdx[i] + 3];
            //gamestate[ballIdx[i] + 1] = HEIGHT / 2 + (int)((gamestate[8] + gamestate[11]) * 10) % (int)(HEIGHT - BALL_RADIUS) + (int)BALL_RADIUS * 2;

        }
    }
    int max = 4;
    if(gamestate[16] > max || gamestate[17] > max)
        return 1;
    // if (gamestate[0] < 0 || gamestate[0] > WIDTH)
    // {
    //     return 1;
    // }
    // else
    // {
    //     return 0;
    // }
    return 0;
}

// The output is an array of floats that stores the score of each paddle. The score is 1 if the paddle won, -1 if it lost, and 0 if it tied

void MultiBallPong::setOutput(int tid, int block, float *output, float *gamestate, const float *startingParams_d)
{
    
        if (gamestate[10] > gamestate[9]) // was gamestate[0] < 0
        {                                 // left paddle lost
            output[block * 2 + 0] = -1;
            output[block * 2 + 1] = gamestate[10];
        }
        else // if (gamestate[0] > WIDTH)
        {
            // right paddle lost
            output[block * 2 + 0] = gamestate[9];
            output[block * 2 + 1] = -1;
        }
        if (block == 0 && (int)startingParams_d[8] % 25 == 0)
            printf("Touches: %d, %d\n", (int)gamestate[9], (int)gamestate[10]);
    
}

Eigen::MatrixXd MultiBallPong::getState(int& action, float & reward, float *gamestate)
{
	Eigen::MatrixXd state(1,10);
	return state;
}


// The ID is a unique identifier for this simulation type
int MultiBallPong::getID()
{
    return 7;
}