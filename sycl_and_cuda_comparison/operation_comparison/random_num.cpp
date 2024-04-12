#include <iostream>
#include <fstream>
#include <random>

#define SIZE 65536

// Function to generate random numbers and save them in a text file
void generateRandomNumbers(int size) {
    // Open the file for writing
    std::ofstream outFile("random_numbers.txt");

    // Check if the file is opened successfully
    if (!outFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return;
    }

    // Seed for random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, size - 1);

    // Generate and save random numbers
    for (int i = 0; i < size; ++i) {
        int randomNumber = dis(gen);
        outFile << randomNumber << std::endl;
    }

    // Close the file
    outFile.close();

    std::cout << "Random numbers saved to random_numbers.txt." << std::endl;
}

int main() {
    int size = SIZE;

    // Generate and save random numbers
    generateRandomNumbers(size);

    return 0;
}
