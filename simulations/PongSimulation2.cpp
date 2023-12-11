#include "PongSimulation2.h"
#include <random>
#define WIDTH 640.0f
#define HEIGHT 480.0f
#define PADDLE_WIDTH 10.0f
#define PADDLE_HEIGHT 50.0f
#define BALL_RADIUS 10.0f
#define BALL_SPEED 8.0f
#define PADDLE_SPEED 6.5f
#define SPEED_UP_RATE 1.00f // Ball will increase in speed by x % after every paddle hit

// Compile command:
// nvcc -rdc=true -lineinfo -o runner .\biology\Genome.cpp .\Simulator.cu .\Runner.cu .\Kernels.cu .\simulations\BasicSimulation.cu .\simulations\TargetSimulation.cu .\simulations\MultibotSimulation.cu .\simulations\AirHockeySimulation.cu .\simulations\PongSimulation2.cu
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
gamestate[12]       // left action
gamestate[13]       // right action
gamestate[14]       // left hit?
gamestate[15]       // right hit?
gamestate[16]       // left delta y
gamestate[17]       // right delta y
gamestate[18]       // prev left pos
gamestate[19]       // prev right pos
gamestate[20]       // winner (-1 for game not over, 0 left, 1 right)
gamestate[21]       // prev ball vy (used for reward shaping, giving more reward to hitting fast moving balls)
*/

unsigned int xorshift(unsigned int x)
{
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 0x2545F491;
}


float rng(float a, float b, unsigned int seed)
{
    unsigned int r = xorshift(seed);
    static const float m = 4294967296.0f; // 2^32
    return a + (b - a) * (static_cast<float>(r) / m);
}

float Limits[8] = {WIDTH, HEIGHT, BALL_SPEED, BALL_SPEED, WIDTH, HEIGHT, WIDTH, HEIGHT};

// The starting parameters are the initial positions and velocities of the ball and the paddles
void PongSimulation2::getStartingParams(float *startingParams)
{
    static int iterationsCompleted = 0;

    startingParams[0] = WIDTH / 2;  // ball x
    startingParams[1] = HEIGHT / 2; // ball y
    startingParams[2] = BALL_SPEED; // ball vx
    // if (iterationsCompleted % 2 == 0)
    if ((double)rand() / RAND_MAX > 0.5)
        startingParams[2] *= -1;

    double speedCoef = fmin(1.5f, (0.5 + 0.0002 * iterationsCompleted));
    startingParams[3] = (((double)rand() / RAND_MAX) - 0.5) * BALL_SPEED * speedCoef; // ball vy
    startingParams[4] = PADDLE_WIDTH / 2;                                 // left paddle x
    startingParams[5] = HEIGHT / 2 + (((double)rand() / RAND_MAX) - 0.5) * HEIGHT / 8; // right paddle y
    startingParams[6] = PADDLE_WIDTH / 2 + WIDTH - PADDLE_WIDTH;          // right paddle x
    startingParams[7] = HEIGHT / 2 + (((double)rand() / RAND_MAX) - 0.5) * HEIGHT / 8; // right paddle y
    startingParams[8] = iterationsCompleted;
    iterationsCompleted++;
}

// The gamestate is an array of floats that stores the current positions and velocities of the ball and the paddles
void PongSimulation2::setupSimulation(int tid, int block, const float *startingParams, float *gamestate)
{

    for (tid = 0; tid < 8; tid++)
        gamestate[tid] = startingParams[tid];
    for (tid = 0; tid < 3; tid++)
        gamestate[tid + 8] = 0;

    gamestate[11] = startingParams[8]; // Generation number
    gamestate[20] = -1;
}

// The activations are the inputs to the neural networks that control the paddles. They are the normalized positions and velocities of the ball and the paddles
void PongSimulation2::setActivations(int tid, int block, float *gamestate, float **activs, int iter)
{

    float rand = 0;
    // float randomMagnitude = 400.0f / logf(gamestate[11] + 2.0f) + 400.0f / logf((float)(iter + 2));
    float randomMagnitude = 400.0f / logf((float)(iter + 2));
    randomMagnitude = 0;

    for (int i = 0; i < 4; i++)
    {
        if (i == 0) //horizontal dist from ball
        {
            activs[0][i] = fabsf(gamestate[4] - gamestate[0]) / Limits[i];
            activs[1][i] = fabsf(gamestate[6] - gamestate[0]) / Limits[i];
        }
        else //ball y and velocity
        {
            activs[0][i] = gamestate[i] / Limits[i];
            activs[1][i] = gamestate[i] / Limits[i];
        }
    }
    activs[1][2] *= -1;

    rand = 0; // rng(-randomMagnitude, randomMagnitude, tid + iter + 0 ^ (int)gamestate[11]);

    activs[0][4] = gamestate[5] / HEIGHT; // left paddle y
    activs[0][5] = (gamestate[7]) / HEIGHT; // right paddle y //skew the other bot's pos a little
    activs[0][5] = rand / HEIGHT;
    
    rand = 0; // rng(-randomMagnitude, randomMagnitude, tid + iter + 0 ^ (int)gamestate[11]);
    activs[1][4] = gamestate[7] / HEIGHT; // right paddle y
    activs[1][5] = (gamestate[5]) / HEIGHT; // left paddle y
    activs[1][5] = rand / HEIGHT;
}


Eigen::MatrixXd PongSimulation2::getStateP1(int& action, float& reward, float** activs)
{
    Eigen::MatrixXd state(6, 1);
    for (int i = 0; i < 6; i++)
        state(i, 0) = activs[0][i];

    return state;

}

Eigen::MatrixXd PongSimulation2::getStateP2(int& action, float& reward, float** activs)
{
    Eigen::MatrixXd state(6, 1);
    for (int i = 0; i < 6; i++)
        state(i, 0) = activs[1][i];

    return state;
}

const float actionEffects[3] = { PADDLE_SPEED, -PADDLE_SPEED, 0 };

// The actions are the outputs of the neural networks that control the paddles. They are the normalized velocities of the paddles in the y direction
void PongSimulation2::eval(int tid, int block, float **actions, float *gamestate)
{

    // if(blockIdx.x == 0 && gamestate[8] > 9990){
    //     printf("Gamestate:\n");
    //     for(int i = 0; i < 8; i++){
    //         printf("%f, ", gamestate[i]);
    //     }
    //     printf("\nActions: %f, %f\n", actions[0][0], actions[1][0]);

    // }

    // Update the paddle positions based on actions and physics
    //gamestate[5] += fminf(1.0, fmaxf(-1.0, actions[0][0])) * PADDLE_SPEED; // left paddle y += action * paddle speed
    //gamestate[7] += fminf(1.0, fmaxf(-1.0, actions[1][0])) * PADDLE_SPEED; // right paddle y += action * paddle speed
    gamestate[14] = 0;
    gamestate[15] = 0;

    gamestate[21] = gamestate[3];

    // Actions:
    // 0 : Go up
    // 1 : Go down
    // 2 : Don't move
    for (int bot = 0; bot < 2; bot++)
    {
        int chosenAction = 0;
        float max = actions[bot][0];

        for (int action = 1; action < 3; action++)
        {
            if (actions[bot][action] > max)
            {
                max = actions[bot][action];
                chosenAction = action;
            }
        }
        // Record action taken this time step
        gamestate[12 + bot] = chosenAction;

        // Update previous position
        gamestate[18 + bot] = gamestate[5 + bot * 2];


        // Update paddle position
        gamestate[5 + bot * 2] += actionEffects[chosenAction];

        // Clamp paddle position to screen dims
        if (gamestate[5 + bot * 2] < PADDLE_HEIGHT / 2)
            gamestate[5 + bot * 2] = PADDLE_HEIGHT / 2;

        else if (gamestate[5 + bot * 2] > HEIGHT - PADDLE_HEIGHT / 2)
            gamestate[5 + bot * 2] = HEIGHT - PADDLE_HEIGHT / 2;
       
        // set delta y
        gamestate[16 + bot] = gamestate[18 + bot] - gamestate[5 + bot * 2];
    }


    /*gamestate[5] += (actions[0][0] < actions[0][1]) ? PADDLE_SPEED : -PADDLE_SPEED;
    gamestate[7] += (actions[1][0] < actions[1][1]) ? PADDLE_SPEED : -PADDLE_SPEED;


    gamestate[12] = (actions[0][0] < actions[0][1]) ? 0 : 1;
    gamestate[13] = (actions[1][0] < actions[1][1]) ? 0 : 1;*/

    
    // Update the ball position and velocity based on physics
    gamestate[0] += gamestate[2]; // ball x += ball vx
    gamestate[1] += gamestate[3]; // ball y += ball vy

    // Check for collisions with walls and paddles and bounce accordingly

    if (gamestate[1] < BALL_RADIUS || gamestate[1] > HEIGHT - BALL_RADIUS)
    {                       // top or bottom wall collision
        gamestate[3] *= -1; // invert ball vy
        gamestate[1] += gamestate[3]; // ball y += ball vy
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
        //gamestate[3] += (gamestate[1] - gamestate[5] - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED; // adjust the ball's vertical speed based on where it hit the paddle
        gamestate[3] = (((double)rand() / RAND_MAX) - 0.5) * BALL_SPEED * 2;
        // Update the ball position and velocity based on physics
        gamestate[0] += gamestate[2]; // ball x += ball vx
        gamestate[1] += gamestate[3]; // ball y += ball vy

        gamestate[9]++;
        gamestate[14] = 1;
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
        //gamestate[3] += (gamestate[1] - gamestate[7] - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED; // adjust the ball's vertical speed based on where it hit the paddle
        gamestate[3] = (((double)rand() / RAND_MAX) - 0.5) * BALL_SPEED * 2;
        // Update the ball position and velocity based on physics
        gamestate[0] += gamestate[2]; // ball x += ball vx
        gamestate[1] += gamestate[3]; // ball y += ball vy

        gamestate[10]++;
        gamestate[15] = 1;
    }

    // Clamp ball vertical speed
    gamestate[3] = fmin(BALL_SPEED * 2, fmax(-BALL_SPEED * 2, gamestate[3]));

    gamestate[8]++;
}

// The simulation is finished when the ball goes out of bounds on either side
int PongSimulation2::checkFinished(int tid, int block, float *gamestate)
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

void PongSimulation2::setOutput(int tid, int block, float *output, float *gamestate, const float *startingParams_d)
{

    // if (gamestate[10] > gamestate[9])
    if (gamestate[0] < 0)
    { // left paddle lost
        output[block * 2 + 0] = -abs(gamestate[5] - gamestate[1]) / 100; // The bot who loses gets a score of negative <dist to ball> (closer to ball the better)
        output[block * 2 + 1] = gamestate[10] + 1;
        gamestate[20] = 1;
    }
    else // if (gamestate[0] > WIDTH)
    {
        // right paddle lost
        output[block * 2 + 0] = gamestate[9] + 1;
        output[block * 2 + 1] = -abs(gamestate[7] - gamestate[1]) / 100;
        gamestate[20] = 0;
    }
    if (block == 0 && (int)startingParams_d[8] % 25 == 0)
        printf("Touches: %d, %d\t", (int)gamestate[9], (int)gamestate[10]);

    
}

Eigen::MatrixXd PongSimulation2::getState(int& action, float & reward, float *gamestate)
{
	Eigen::MatrixXd state(1,1);

    reward = gamestate[14] / 5.0f; //* fmax(abs(gamestate[21]) / BALL_SPEED, 1);
    if (gamestate[0] < 0)
    {
        float dist = abs(abs(gamestate[5] - gamestate[1]) - PADDLE_HEIGHT / 2);
        dist = fmin(dist, 150);
        reward = fmin(-0.2, -dist / 100.f);
    }
        
    else if (gamestate[0] > WIDTH)
        reward = 1;



   /* if (abs(gamestate[16]) < 0.1)
        reward -= 0.02;
    else
    {
        reward += 0.02;
    }*/
    /*if (gamestate[12] == 2)
        reward -= 0.0001;*/
    
    // If the ball is moving away from the left paddle, give negative reward proportional to dist from middle and dist from paddle
    //      (if ball is moving away but we basically just hit it, give less negative reward...give time for bot to reset position)
    //if (gamestate[2] > 0)
    //    reward -= fmax(abs(gamestate[5] - HEIGHT / 2) / HEIGHT / 20000 * gamestate[0] / WIDTH, 0);

    
    //printf("reward = %f\n", reward);
    action = gamestate[12];




	return state;
}



// The ID is a unique identifier for this simulation type
int PongSimulation2::getID()
{
    return 6;
}