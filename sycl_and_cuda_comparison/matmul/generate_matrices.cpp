#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

// Define the value of N using a macro
#define N 64

int main()
{
    // Seed the random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Create two matrices of size N x N and populate them with random values
    int **matrixA, **matrixB;
    matrixA = (int **)malloc(N * sizeof(int *));

    // Allocate memory for each row (an array of integers)
    for (int i = 0; i < N; i++)
    {
        matrixA[i] = (int *)malloc(N * sizeof(int));
    }

    matrixB = (int **)malloc(N * sizeof(int *));

    // Allocate memory for each row (an array of integers)
    for (int i = 0; i < N; i++)
    {
        matrixB[i] = (int *)malloc(N * sizeof(int));
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrixA[i][j] = std::rand() % 100; // Generate random values between 0 and 99
            matrixB[i][j] = std::rand() % 100;
        }
    }

    // Open a file for writing (overwrite if it already exists)
    std::ofstream outputFile("output.txt");

    if (!outputFile)
    {
        std::cerr << "Error opening the output file." << std::endl;
        return 1;
    }

    // Write N to the file
    outputFile << N << " " << N << "\n";

    // Write the values of matrix A to the file
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            outputFile << matrixA[i][j] << " ";
        }
    }
    outputFile << "\n";

    // Write the values of matrix B to the file
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            outputFile << matrixB[i][j] << " ";
        }
    }

    // Close the output file
    outputFile.close();

    std::cout << "Matrices written to 'output.txt'" << std::endl;

    return 0;
}
