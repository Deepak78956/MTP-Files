#include <iostream>
#include <fstream>
#include <random>
#include <unordered_set>
#define size (1 << 18)

// Function to fill the array with unique random numbers
void fillArrayUniqueRandom() {
    // Open the file for writing
    std::ofstream outFile("random_numbers.txt");

    // Check if the file is opened successfully
    if (!outFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return;
    }

    unsigned stridedArray[size];
    stridedArray[0] = 0;

    for (int i = 1; i < size; i++) {
        stridedArray[i] = 1024 + stridedArray[i - 1];
    }

    printf("%u\n", stridedArray[size - 1]);

    std::unordered_set<int> uniqueNumbers;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, size - 1);

    for (int i = 0; i < size; ++i) {
        int randNum;
        do {
            randNum = dis(gen); // Generate a random number within the range
        } while (uniqueNumbers.count(randNum) > 0); // Check if it's already generated

        outFile << stridedArray[randNum] << std::endl;
        uniqueNumbers.insert(randNum); // Insert the number into the set
    }

    // Close the file
    outFile.close();

    std::cout << "Random numbers saved to random_numbers.txt." << std::endl;
}

int main() {
    fillArrayUniqueRandom();

    return 0;
}
