#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to multiply 2 matrices
void matrix_multiply(double** A, int rowsA, int colsA, double** B, int rowsB, int colsB, double** C) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            C[i][j] = 0;
            for (int k = 0; k < colsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to find the transpose of a matrix
void matrix_transpose(double** A, int rowsA, int colsA, double** A_T) {
    for (int i = 0; i < colsA; i++) {
        A_T[i] = (double*)malloc(rowsA * sizeof(double));
    }

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsA; j++) {
            A_T[j][i] = A[i][j];
        }
    }
}

// Function to find the inverse of a matrix through Gauss-Jordan elimination
void invert(int n, double** dest, double** src) {
    double** N = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        N[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
             if (i == j) {
                N[i][j] = 1.0;
            } else {
                N[i][j] = 0.0;
            }
        }
    }

    for (int p = 0; p < n; p++) {
        double f = src[p][p];
        for (int i = 0; i < n; i++) {
            src[p][i] /= f;
            N[p][i] /= f;
        }
        for (int i = 0; i < n; i++) {
            if (i != p) {
                f = src[i][p];
                for (int j = 0; j < n; j++) {
                    src[i][j] -= src[p][j] * f;
                    N[i][j] -= N[p][j] * f;
                }
            }
        }
    }

    for (int p = n - 1; p >= 0; p--) {
        for (int i = p - 1; i >= 0; i--) {
            double f = src[i][p];
            for (int j = 0; j < n; j++) {
                src[i][j] -= src[p][j] * f;
                N[i][j] -= N[p][j] * f;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dest[i][j] = N[i][j];
        }
    }

    for (int i = 0; i < n; i++) {
        free(N[i]);
    }
    free(N);
}



int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Two input files are required.\n");
        return 1;
    }

/* --------------------------Input Train File-------------------------- */
    int kTrain, nTrain;
    double** trainingData;
    double** X; 
    double** Y; 

    FILE* trainingFile = fopen(argv[1], "r");
    if (trainingFile == NULL) {
        printf("Error opening file: %s\n", argv[1]);
        return 1; 
    }

    // Skip the first line in trainingData which says "train"
    char buffer[100];
    if (fgets(buffer, sizeof(buffer), trainingFile) == NULL) {
        fclose(trainingFile);
        return 1;
    }

    // Scan first 2 integers (k and n) from the 2nd and 3rd lines of the data file
    if (fscanf(trainingFile, "%d", &kTrain) != 1 || fscanf(trainingFile, "%d", &nTrain) != 1) {
        fclose(trainingFile);
        return 1;
    }

    // Creating training data matrix
    trainingData = (double**)malloc(nTrain * sizeof(double*));
    for (int i = 0; i < nTrain; i++) {
        trainingData[i] = (double*)malloc((kTrain + 1) * sizeof(double)); 
        for (int j = 0; j <= kTrain; j++) {
            if (fscanf(trainingFile, "%lf", &trainingData[i][j]) != 1) {
                fclose(trainingFile);
                return 1;
            }
        }
    }


    // Creating matrices X and Y from trainingData
    X = (double**)malloc(nTrain * sizeof(double*));
    Y = (double**)malloc(nTrain * sizeof(double*));

    for (int i = 0; i < nTrain; i++) {
        X[i] = (double*)malloc((kTrain + 1) * sizeof(double)); 
        Y[i] = (double*)malloc(sizeof(double));

        X[i][0] = 1.0;

        for (int j = 1; j <= kTrain; j++) {
            X[i][j] = trainingData[i][j - 1]; 
        }

        Y[i][0] = trainingData[i][kTrain];

    }

        fclose(trainingFile);


    /* --------------------------Input Data File-------------------------- */
        int kInput, mInput;
        double** inputData;

        FILE* inputFile = fopen(argv[2], "r");
        if (inputFile == NULL) {
            printf("Error opening file: %s\n", argv[2]);
            return 1; 
        }

        // Skip the first line in inputData which says "data"
        if (fgets(buffer, sizeof(buffer), inputFile) == NULL) {
            fclose(inputFile);
            return 1;
        }

        // Sscan first 2 integers (k and m) from the 2nd and 3rd lines of the data file
        if (fscanf(inputFile, "%d", &kInput) != 1 || fscanf(inputFile, "%d", &mInput) != 1) {
            fclose(inputFile);
            return 1;
        }

        // Creating inputData matrix
        inputData = (double**)malloc(mInput * sizeof(double*));
        for (int i = 0; i < mInput; i++) {
            inputData[i] = (double*)malloc(kInput * sizeof(double));
            for (int j = 0; j < kInput; j++) {
                if (fscanf(inputFile, "%lf", &inputData[i][j]) != 1) {
                    fclose(inputFile);
                    return 1;
                }
            }
        }

        // Creating matrix X_data
        double** X_data = (double**)malloc(mInput * sizeof(double*));
        for (int i = 0; i < mInput; i++) {
            X_data[i] = (double*)malloc((kTrain + 1) * sizeof(double)); 
            X_data[i][0] = 1.0;
            for (int j = 1; j <= kTrain; j++) {
                X_data[i][j] = inputData[i][j - 1]; 
            }
        }

        fclose(inputFile);

/* --------------------------Calculating W-------------------------- */
        // Creating and calculating XT
        double** Xt;
        Xt = (double**)malloc((kTrain + 1) * sizeof(double*)); 
        matrix_transpose(X, nTrain, kTrain + 1, Xt);
        

        //Creating and calculating XtX
        double** XtX;
        XtX = (double**)malloc((kTrain + 1) * sizeof(double*));
        for (int i = 0; i < kTrain + 1; i++) {
            XtX[i] = (double*)malloc((kTrain + 1) * sizeof(double));
        }
        matrix_multiply(Xt, kTrain + 1, nTrain, X, nTrain, kTrain + 1, XtX);


        // Creating and calculating XtX_inverse (inverse of XtX)
        double** XtX_inverse;
        XtX_inverse = (double**)malloc((kTrain + 1) * sizeof(double*));
        for (int i = 0; i < kTrain + 1; i++) {
            XtX_inverse[i] = (double*)malloc((kTrain + 1) * sizeof(double));
        }
        invert(kTrain + 1, XtX_inverse, XtX);
        
        
        // Creating and calculating XtX_inv * Xt to get the matrix Xi
        double** Xi;
        Xi = (double**)malloc((kTrain + 1) * sizeof(double*));
        for (int i = 0; i < kTrain + 1; i++) {
            Xi[i] = (double*)malloc(nTrain * sizeof(double));
        }
        matrix_multiply(XtX_inverse, kTrain + 1, kTrain + 1, Xt, kTrain + 1, nTrain, Xi);


        // Creating and calculating W by multiplying Xi with Y
        double** W;
        W = (double**)malloc((kTrain + 1) * sizeof(double*));
        for (int i = 0; i < kTrain + 1; i++) {
            W[i] = (double*)malloc(sizeof(double));
        }
        matrix_multiply(Xi, kTrain + 1, nTrain, Y, nTrain, 1, W);

/* --------------------------Calculating Predicted Prices-------------------------- */
    // Creating and calculating Y_PredPrice by multiplying X_data with W
    double** Y_PredPrice;
    Y_PredPrice = (double**)malloc(mInput * sizeof(double*));
    for (int i = 0; i < mInput; i++) {
        Y_PredPrice[i] = (double*)malloc(1 * sizeof(double));
    }
    matrix_multiply(X_data, mInput, kTrain + 1, W, kTrain + 1, 1, Y_PredPrice);


    for (int i = 0; i < mInput; i++) {
        printf("%.0f\n", Y_PredPrice[i][0]);
    }

/* --------------------------Freeing allocated memory-------------------------- */
    for (int i = 0; i < nTrain; i++) {
        free(trainingData[i]);
        free(X[i]);
        free(Y[i]);
    }
    free(trainingData);
    free(X);
    free(Y);

    for (int i = 0; i < mInput; i++) {
        free(inputData[i]);
        free(X_data[i]);
        free(Y_PredPrice[i]);
    }
    free(inputData);
    free(X_data);
    free(Y_PredPrice);

    for (int i = 0; i < kTrain + 1; i++) {
        free(Xt[i]);
        free(XtX[i]);
        free(XtX_inverse[i]);
        free(Xi[i]);
        free(W[i]);
    }
    free(Xt);
    free(XtX);
    free(XtX_inverse);
    free(Xi);
    free(W);

    return 0;
}