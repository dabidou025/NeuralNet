#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double ReLu (double x) {
    return (x > 0) ? x : 0.;
}

double ReLu_derivative (double x) {
    return (x > 0) ? 1 : 0;
}

double Sigmoid (double x) {
    return 1 / (1 + exp(-x));
}

double Sigmoid_derivative (double x) {
    return Sigmoid(x) * (1 - Sigmoid(x));
}

double Loss_function (double* output, double* label, int output_size) {
    double ret = 0.;
    for (int i = 0; i < output_size; i++)
        ret += pow(output[i] - label[i], 2.);
    return 0.5*ret;
}

double Loss_function_derivative (double output, double label) {
    return output - label;
}

// NETWORK SHAPE

#define num_layers 4

#define K 10
#define N 2
#define M 2

#define bias 1

int dims[num_layers] = {N, 4, 5, M};

typedef double (*f)(double);
f functions[num_layers] = {&Sigmoid, &Sigmoid, &Sigmoid, &Sigmoid};
f functions_derivatives[num_layers] = {&Sigmoid_derivative, &Sigmoid_derivative, &Sigmoid_derivative, &Sigmoid_derivative};

//

int RandomInt (int min, int max) {
    return rand() % (max - min) + min;
}

double RandomDouble (double min, double max) {
    return ((double) rand() / RAND_MAX)*(max-min) + min;
}

typedef struct {
    int size;
    double *output, *error;
    double** weights;
    double (*activation_function)(double x);
    double (*activation_function_derivative)(double x);
} LAYER;

typedef struct {
    LAYER *input, *output;
    LAYER** layers;
    double error, eta;
} NET;

// INITIALIZATION

void CreateNet (NET* Net) {

    Net->layers = calloc(num_layers, sizeof(LAYER*));
    
    for (int l = 0; l < num_layers; l++) {

        Net->layers[l] = malloc(sizeof(LAYER));

        Net->layers[l]->size = dims[l];
        Net->layers[l]->output = calloc(dims[l]+1, sizeof(double));
        Net->layers[l]->output[0] = bias;
        Net->layers[l]->error = calloc(dims[l]+1, sizeof(double));
        Net->layers[l]->weights = calloc(dims[l]+1, sizeof(double*));
        Net->layers[l]->activation_function = functions[l];
        Net->layers[l]->activation_function_derivative = functions_derivatives[l];
        
        if (l > 0) {
            for (int i = 1; i <= dims[l]; i++) {
                Net->layers[l]->weights[i] = calloc(dims[i-1]+2, sizeof(double));
            }
        }
        Net->input = Net->layers[0];
        Net->output = Net->layers[num_layers-1];
        Net->eta = 0.09;
    }
}

void InitInput (NET* Net, double* Input) {
    for (int i = 1; i <= Net->input->size; i++)
        Net->input->output[i] = Input[i-1]; 
}

void GetOutput (NET* Net, double* Output) {
    for (int i = 1; i <= Net->output->size; i++)
        Output[i-1] = Net->output->output[i];
}

void InitWeights (NET* Net) {

    printf(" ici11 %i \n",Net->layers[3]->size);
    for (int l = 1; l < num_layers; l++) {
        printf("la %i %i \n",l,Net->layers[l]->size);
        for (int i = 1; i <= Net->layers[l]->size; i++) {
            for (int j = 0; j <= Net->layers[l-1]->size; j++) {
                Net->layers[l]->weights[i][j] = RandomDouble(-0.5,0.5);
                printf("%i %i %i %f\n",l,i,j,Net->layers[l]->weights[i][j]);
                printf("%i %i %i\n",l,i,j);
            }
        }
    }
    printf("cc");
}

// FORWARD

void ForwardLayer (LAYER* Lower, LAYER* Upper) {
    for (int i = 1; i <= Upper->size; i++) {
        double sum = 0;
        for (int j = 0; j <= Lower->size; j++)
            sum += Upper->weights[i][j] * Lower->output[j];
        Upper->output[i] = Upper->activation_function(sum);
    }
}

void ForwardNet (NET* Net) {
    for (int l = 0; l < num_layers-1; l++) {
        ForwardLayer(Net->layers[l], Net->layers[l+1]);
    }
}

// BACKPROPAGATION

void ComputeOutputError (NET* Net, double* Label) {
    for (int i = 1; i <= Net->output->size; i++) {
        double out = Net->output->output[i];
        double err = Loss_function_derivative(out, Label[i-1]);
        Net->output->error[i] = err * Net->output->activation_function_derivative(out); 
    }
    Net->error = Loss_function(Net->output->output, Label, Net->output->size);
}

void BackpropagateLayer (NET* Net, LAYER* Upper, LAYER* Lower) {
    for (int i = 1; i <= Lower->size; i++) {
        double out = Lower->output[i];
        double err = 0.;
        for (int j = 1; j <= Upper->size; j++)
            err += Upper->weights[j][i] * Upper->error[j];
        Lower->error[i] = err * Lower->activation_function_derivative(out); 
    }
}

void BackpropagateNet (NET* Net) {
    for (int l = num_layers - 1; l > 1; l--)
        BackpropagateLayer(Net, Net->layers[l], Net->layers[l-1]);
}

void AdjustWeights (NET* Net) {
    for (int l = 1; l < num_layers; l++) {
        for (int i = 1; i <= Net->layers[l]->size; i++) {
            for (int j = 0; j <= Net->layers[l-1]->size; j++) {
                double out = Net->layers[l-1]->output[j];
                double err = Net->layers[l]->error[i];
                Net->layers[l]->weights[i][j] -= Net->eta * out * err;
            }
        }
    }
}

void SimulateNet (NET* Net, double* Input, double* Output, double* Label, int training) {
    InitInput(Net, Input);
    ForwardNet(Net);
    GetOutput(Net, Output);

    ComputeOutputError(Net, Label);
    if (training) {
        BackpropagateNet(Net);
        AdjustWeights(Net);
    }
}

void TrainNet (NET* Net, double** TrainingSet, double** Labels, int epochs) {
    double Output[M];
    for (int i = 1; i <= epochs; i++) {
        printf("Epoch : %i, Error = %f \n", i, Net->error);
        int k = RandomInt(0, K-1);
        double* Input = TrainingSet[k];
        double* Label = Labels[k];
        SimulateNet(Net, Input, Output, Label, 1);
    }
}


void TestNet (NET* Net, double** TrainingSet, double** Labels) {
    double Output[M];
    for (int i = 0; i < K; i++) {
        double* Input = TrainingSet[i];
        double* Label = Labels[i];
        SimulateNet(Net, Input, Output, Label, 0);
        printf("i = %i, Output = { ", i);
        for (int i = 0; i < M; i++) {
            printf("%f ", Output[i]);
        }
        printf("}\n");
        printf("     , Label  = { ", i);
        for (int i = 0; i < M; i++) {
            printf("%f ", Label[i]);
        }
        printf("}\n\n");
    }
    printf("\n");
}

int main () {
    
    printf("START\n");

    NET Net;
    int epochs = 20;
    
    time_t t;
    srand((unsigned) time(&t));
    
    printf("START1\n");
    CreateNet(&Net);
    printf("START2\n");
    InitWeights(&Net);
    printf("START3\n");

    double ts[K][N] = {
        {1.3,1.2}, {2.2,1.1}, {1.1,2.1}, {3.5,1.3}, {1.5,3.5},
        {-1.1,-1.2}, {-2.2,-1.5}, {-1.4,-2.2}, {-3.05,-1.1}, {-1.2,-3.6},
    };

    double ls[K][M] = {
        {1,0}, {1,0}, {1,0}, {1,0}, {1,0},
        {0,1}, {0,1}, {0,1}, {0,1}, {0,1},
    };


    double* TrainingSet[K] = { ts[0], ts[1], ts[2] , ts[3], ts[4],
                               ts[5], ts[6], ts[7], ts[8], ts[9] };
    
    double* Labels[K] = { ls[0], ls[1], ls[2] , ls[3], ls[4],
                          ls[5], ls[6], ls[7], ls[8], ls[9] };

    printf("START4\n");
    TrainNet(&Net, TrainingSet, Labels, epochs);
    printf("TRAINING DONE\n");
    TestNet(&Net, TrainingSet, Labels);
    printf("START5\n");

    return 0;
}
