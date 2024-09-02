#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ARRAY_SIZE_LAYER1 5
#define ARRAY_SIZE_LAYER2 3
#define TRAINING_ARRAYS 4 
#define MAX_ITERATIONS 10
#define A 0.5
#define LAMBDA 0.5


// Function to initialize the array with 0 or 1 randomly
void initialize_array(float arr[], int size) {
    // set the first element as 1 (bias)
    arr[0] = 1;
    // initialize the array with 0 or 1 randomly using a random number generator
    for (int i = 1; i < size; i++) {
        arr[i] = (float)rand() / RAND_MAX; // generate a random float between 0 and 1
        if (arr[i] < 0.5) {
            arr[i] = 0.0f;
        } else {
            arr[i] = 1.0f;
        }
    }
}

void print_array(float arr[], int size) {
    // print the array
    for (int i = 0; i < size; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

float calculate_Sum_Arrays(float delta, float W[], int size) {
    float result = 0;
    for (int i = 1; i < size; i++) {
        result += delta * W[i];
    }
    printf("V = %.3f\n", result);
    return result;
}

float calculate_y(float X[], float W[], int size) {
    float result = 0;
    for (int i = 0; i < size; i++) {
        result += X[i] * W[i];
    }
    printf("V = %.3f\n", result);

    // apply the activation function
    float y = 1 / (1 + exp(-A * result));
    printf("y = %.3f\n", y);
    return y;
}
float calculate_average_array ( float quadratic_error_array[], int size){
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += (quadratic_error_array[i]* quadratic_error_array[i]);
    }
    return sum / size;

}
int main() {
    srand(time(NULL)); // seed the random number generator
    
    float X[4][ARRAY_SIZE_LAYER1] = {
        {1, 1, 0, 1, 0},  // X1
        {1, 0, 1, 0, 1},  // X2
        {1, 0, 0, 1, 0},  // X3
        {1, 0, 0, 0, 1}   // X4
    };
    float yd[4][2] = {
        {1, 0},  // yd1
        {0, 1},  // yd2
        {1, 0},  // yd3
        {0, 1}   // yd4
    };
    float wi1[ARRAY_SIZE_LAYER1], wi2[ARRAY_SIZE_LAYER1];
    float wj1[ARRAY_SIZE_LAYER2], wj2[ARRAY_SIZE_LAYER2];
    float dwj1[ARRAY_SIZE_LAYER2], dwj2[ARRAY_SIZE_LAYER2];
    float dwi1[ARRAY_SIZE_LAYER1], dwi2[ARRAY_SIZE_LAYER1];
    float Y1 [ARRAY_SIZE_LAYER2];
    float yi1, yi2, yj1, yj2, erroj1, erroj2, quadraticError, Age_quadratic_error, gradientj1, gradientj2, gradient_i1, gradient_i2;
    float quadratic_error_array[TRAINING_ARRAYS];
    
    printf("Training arrays: \n");
    for (int count = 0; count < TRAINING_ARRAYS; count++){
        printf("X: %d\t", count+1);
        print_array(X[count], ARRAY_SIZE_LAYER1);
        printf("Desired Y: %.0f %.0f\n", yd[count][0], yd[count][1]);
    }

    // initialize the arrays randomly
    initialize_array(wi1, ARRAY_SIZE_LAYER1); 
    initialize_array(wi2, ARRAY_SIZE_LAYER1); 
    initialize_array(wj1, ARRAY_SIZE_LAYER2); 
    initialize_array(wj2, ARRAY_SIZE_LAYER2); 

  
    printf("Initialized arrays: \n");
    print_array(wi1, ARRAY_SIZE_LAYER1);
    print_array(wi2, ARRAY_SIZE_LAYER1);
    print_array(wj1, ARRAY_SIZE_LAYER2);
    print_array(wj2, ARRAY_SIZE_LAYER2);

    for (int t_atual = 0; t_atual < TRAINING_ARRAYS; t_atual++){
        printf("\n\n\nTreinando usando o X[%d]\n", t_atual+1);
        // calculate the y for the first layer
        yi1 = calculate_y(X[t_atual], wi1, ARRAY_SIZE_LAYER1);
        yi2 = calculate_y(X[t_atual], wi2, ARRAY_SIZE_LAYER1);

        // calculate the y for the second layer
        Y1[0] = 1;
        Y1[1] = yi1;
        Y1[2] = yi2;
        yj1 = calculate_y(Y1, wj1, ARRAY_SIZE_LAYER2);
        yj2 = calculate_y(Y1, wj2, ARRAY_SIZE_LAYER2);

        // calculate error for the second layer
        erroj1 = yd[t_atual][0] - yj1;
        erroj2 = yd[t_atual][1] - yj2;
        printf("Erros %f %f\n", erroj1, erroj2);

        // calculate the quadratic error for this Training Array
        quadraticError = 0.5 * pow(erroj1, 2) + 0.5 * pow(erroj2, 2);
        quadratic_error_array[t_atual] = quadraticError;
        printf("Quadratic Error: %.3f\n", quadraticError);

        // calculate the gradient for the local error for the second layer
        gradientj1 = yj1 * (1 - yj1) * erroj1 * A;
        gradientj2 = yj2 * (1 - yj2) * erroj2 * A;
        printf("Gradient for j1: %.3f, Gradient for j2: %.3f\n", gradientj1, gradientj2);

        // calculate the gradient for the first layer
        gradient_i1 = A * yi1 * (1 - yi1) * calculate_Sum_Arrays(gradientj1, wj1, 3);
        gradient_i2 = A * yi2 * (1 - yi2) * calculate_Sum_Arrays(gradientj2, wj2, 3);
        printf("Gradient for i1: %f, Gradient for i2: %f\n", gradient_i1, gradient_i2);

        //update the weights for the second layer
        printf("Wheight arrays of the SECOND layer BEFORE the update: \n");
        print_array(wj1, ARRAY_SIZE_LAYER2);
        print_array(wj2, ARRAY_SIZE_LAYER2);
        for (int i = 0; i < ARRAY_SIZE_LAYER2; i++) {
            dwj1[i] = LAMBDA * gradientj1 * Y1[i];
            dwj2[i] = LAMBDA * gradientj2 * Y1[i];
            wj1[i] += dwj1[i];
            wj2[i] += dwj2[i];
        //printf("dwj1[%d] = %.3f, dwj2[%d] = %.3f, wj1[%d] = %.3f, wj2[%d] = %.3f\n", i, dwj1[i], i, dwj2[i], i, wj1[i], i, wj2[i]);
        }
        printf("Wheight arrays of the SECOND layer AFTER the update: \n");
        print_array(wj1, ARRAY_SIZE_LAYER2);
        print_array(wj2, ARRAY_SIZE_LAYER2);

        // update the weights for the first layer
        printf("Wheight arrays of the FIRST layer BEFORE the update: \n");
        print_array(wi1, ARRAY_SIZE_LAYER1);
        print_array(wi2, ARRAY_SIZE_LAYER1);
        for (int i = 0; i < ARRAY_SIZE_LAYER1; i++) {
            dwi1[i] = LAMBDA * gradient_i1 * X[t_atual][i];
            dwi2[i] = LAMBDA * gradient_i2 * X[t_atual][i];
            wi1[i] += dwi1[i];
            wi2[i] += dwi2[i];
            //printf("dwi1[%d] = %.3f, dwi2[%d] = %.3f, wi1[%d] = %.3f, wi2[%d] = %.3f\n", i, dwi1[i], i, dwi2[i], i, wi1[i], i, wi2[i]);
        }
        printf("Wheight arrays of the FIRST layer AFTER the update: \n");
        print_array(wi1, ARRAY_SIZE_LAYER1);
        print_array(wi2, ARRAY_SIZE_LAYER1);

    }

    // calculate the average quadratic error for all training arrays
    Age_quadratic_error = calculate_average_array(quadratic_error_array, TRAINING_ARRAYS);
    printf("Average Age Quadratic Error: %.3f\n", Age_quadratic_error);
    return 0;
}