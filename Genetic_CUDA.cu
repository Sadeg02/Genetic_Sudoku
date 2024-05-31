#include <iostream>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <random>
#include <omp.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

using namespace std;

const int POPULATION_SIZE = 500;
const int GENERATIONS = 1000;
const int BEST_COUNT = 50;
const double MUTATION_RATE = 0.2;
const int BOARD_SIZE = 9;
const int SUBGRID_SIZE = 3;
const int NUM_RUNS = 1;
const int THREADS = 10;

struct Individual {
    int board[BOARD_SIZE][BOARD_SIZE];
    int quality;
};

// Pomocnicza funkcja do drukowania planszy Sudoku
__host__ void print_board(const int board[BOARD_SIZE][BOARD_SIZE]) {
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            cout << board[i][j] << " ";
        }
        cout << endl;
    }
}

// Funkcja zwracająca dostępne liczby dla danego pola
__device__ void get_possible_numbers(const int grid[BOARD_SIZE][BOARD_SIZE], int row, int col, bool possible_numbers[BOARD_SIZE + 1]) {
    for (int i = 0; i <= BOARD_SIZE; ++i) {
        possible_numbers[i] = true;
    }

    // Sprawdzanie dostępnych liczb w wierszu
    for (int i = 0; i < BOARD_SIZE; ++i) {
        possible_numbers[grid[row][i]] = false;
    }

    // Sprawdzanie dostępnych liczb w kolumnie
    for (int i = 0; i < BOARD_SIZE; ++i) {
        possible_numbers[grid[i][col]] = false;
    }

    // Sprawdzanie dostępnych liczb w kwadracie 3x3
    int start_row = row / SUBGRID_SIZE * SUBGRID_SIZE;
    int start_col = col / SUBGRID_SIZE * SUBGRID_SIZE;
    for (int i = start_row; i < start_row + SUBGRID_SIZE; ++i) {
        for (int j = start_col; j < start_col + SUBGRID_SIZE; ++j) {
            possible_numbers[grid[i][j]] = false;
        }
    }
   
}

// Funkcja do wypełniania pojedynczego Sudoku na podstawie dostępnych liczb
__device__ void shuffle_cell_order(int cell_order[BOARD_SIZE * BOARD_SIZE], curandState* state) {
    int remaining = BOARD_SIZE * BOARD_SIZE;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
        int j = curand_uniform(state) * remaining; // Losujemy indeks z pozostałych elementów
        int temp = cell_order[i];
        cell_order[i] = cell_order[j];
        cell_order[j] = temp;
        remaining--;
    }
}

__device__ void fill_sudoku(const int initial_board[BOARD_SIZE][BOARD_SIZE], int grid[BOARD_SIZE][BOARD_SIZE], curandState* state) {
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            grid[i][j] = initial_board[i][j];
        }
    }

    // Losowo ustawiamy kolejność próby wstawienia liczby w każdą komórkę
    int cell_order[BOARD_SIZE * BOARD_SIZE];
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
        cell_order[i] = i;
    }
    shuffle_cell_order(cell_order, state);

    for (int k = 0; k < BOARD_SIZE * BOARD_SIZE; ++k) {
        int row = cell_order[k] / BOARD_SIZE;
        int col = cell_order[k] % BOARD_SIZE;

        if (grid[row][col] == 0) {
            bool possible_numbers[BOARD_SIZE + 1];
            get_possible_numbers(grid, row, col, possible_numbers);

            int num_choices = 0;
            for (int i = 1; i <= BOARD_SIZE; ++i) {
                if (possible_numbers[i]) {
                    num_choices++;
                }
            }

            if (num_choices == 0) {
                break; // Przerwij próbę wypełnienia, jeśli brak dostępnych liczb
            }

            int num_index = curand(state) % num_choices;
            int num_count = 0;
            for (int i = 1; i <= BOARD_SIZE; ++i) {
                if (possible_numbers[i]) {
                    if (num_count == num_index) {
                        grid[row][col] = i;
                        break;
                    }
                    num_count++;
                }
            }
        }
    }
}

// Funkcja obliczająca ilość pustych pól w planszy
__device__ int count_empty_cells(int board[BOARD_SIZE][BOARD_SIZE]) {
    int empty_cells = 0;
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            if (board[i][j] == 0) {
                empty_cells++;
            }
        }
    }
    return empty_cells;
}

__global__ void generate_first_population_kernel(const int initial_board[BOARD_SIZE][BOARD_SIZE], Individual population[POPULATION_SIZE]) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < POPULATION_SIZE) {
        curandState state;
        curand_init(id, 0, 0, &state);
        fill_sudoku(initial_board, population[id].board, &state); // Wypełnij planszę Sudoku dla nowego osobnika
        population[id].quality = count_empty_cells(population[id].board); // Oblicz jakość planszy dla nowego osobnika
    }
}

void generate_first_population_CUDA(const int initial_board[BOARD_SIZE][BOARD_SIZE], Individual population[POPULATION_SIZE]) {
    // Alokacja pamięci na GPU
    int(*d_initial_board)[BOARD_SIZE];
    cudaMalloc((void**)&d_initial_board, BOARD_SIZE * BOARD_SIZE * sizeof(int));
    Individual* d_population;
    cudaMalloc((void**)&d_population, POPULATION_SIZE * sizeof(Individual));

    // Kopiowanie danych z hosta do urządzenia
    cudaMemcpy(d_initial_board, initial_board, BOARD_SIZE * BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Wywołanie kernela
    int threadsPerBlock = THREADS;
    int blocksPerGrid = (POPULATION_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    generate_first_population_kernel << <threadsPerBlock, blocksPerGrid>> > (d_initial_board, d_population);

    // Kopiowanie wyników z urządzenia do hosta
    cudaMemcpy(population, d_population, POPULATION_SIZE * sizeof(Individual), cudaMemcpyDeviceToHost);

    // Zwolnienie pamięci na GPU
    cudaFree(d_initial_board);
    cudaFree(d_population);
}

// Funkcja wybierająca najlepsze osobniki z populacji na podstawie jakości planszy
__host__ void select_best_individuals(const Individual population[POPULATION_SIZE], Individual best_individuals[BEST_COUNT]) {
    // Tworzymy kopię populacji, aby nie zmieniać kolejności oryginalnej
    Individual sorted_population[POPULATION_SIZE];
    memcpy(sorted_population, population, sizeof(Individual) * POPULATION_SIZE);

    // Sortujemy populację według jakości planszy (rosnąco)
    sort(sorted_population, sorted_population + POPULATION_SIZE, [](const Individual& a, const Individual& b) {
        return a.quality < b.quality;
        });

    // Wybieramy najlepsze jednostki
    for (int i = 0; i < BEST_COUNT && i < POPULATION_SIZE; ++i) {
        best_individuals[i] = sorted_population[i];
    }
}

__device__ void create_child(const int initial_board[BOARD_SIZE][BOARD_SIZE], const Individual& parent1, const Individual& parent2, Individual& child, curandState* state) {
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            child.board[i][j] = initial_board[i][j];
        }
    }

    int cell_order[BOARD_SIZE * BOARD_SIZE];
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
        cell_order[i] = i;
    }
    shuffle_cell_order(cell_order, state);

    // Losowo przemieszaj kolejność komórek dziecka
    shuffle_cell_order(cell_order, state);

    for (int k = 0; k < BOARD_SIZE * BOARD_SIZE; ++k) {
        int row = cell_order[k] / BOARD_SIZE;
        int col = cell_order[k] % BOARD_SIZE;

        if (child.board[row][col] == 0) {
            // Sprawdzenie mutacji
            if (curand_uniform(state) < 0.25) {
                bool possible_numbers[BOARD_SIZE + 1];
                get_possible_numbers(child.board, row, col, possible_numbers);
                int num_choices = 0;
                for (int i = 1; i <= BOARD_SIZE; ++i) {
                    if (possible_numbers[i]) {
                        num_choices++;
                    }
                }
                if (num_choices > 0) {
                    int num_index = curand(state) % num_choices;
                    int num_count = 0;
                    for (int i = 1; i <= BOARD_SIZE; ++i) {
                        if (possible_numbers[i]) {
                            if (num_count == num_index) {
                                child.board[row][col] = i;
                                break;
                            }
                            num_count++;
                        }
                    }
                }
            }
            else {
                // Sprawdź dostępność wartości od obu rodziców
                bool possible_numbers[BOARD_SIZE + 1];
                get_possible_numbers(child.board, row, col, possible_numbers);

                // Sprawdź, czy wartość komórki rodzica może wystąpić w komórce dziecka
                bool value_can_be_in_child = possible_numbers[parent1.board[row][col]] || possible_numbers[parent2.board[row][col]];

                // Jeśli wartość może wystąpić w dziecku, przypisz ją
                if (value_can_be_in_child) {
                    if (possible_numbers[parent1.board[row][col]]) {
                        child.board[row][col] = parent1.board[row][col];
                    }
                    else {
                        child.board[row][col] = parent2.board[row][col];
                    }
                }
                else {
                    // W przypadku braku możliwości wystąpienia wartości w dziecku, pozostaw komórkę dziecka pustą
                    child.board[row][col] = 0;
                }
            }
        }
    }
}


// Funkcja tworząca nowe osobniki w populacji na podstawie najlepszych osobników
__global__ void create_new_population_kernel(const int initial_board[BOARD_SIZE][BOARD_SIZE], const Individual best_individuals[BEST_COUNT], Individual population[POPULATION_SIZE], float mutation_rate) {
    int id = threadIdx.x + (blockIdx.x * blockDim.x);

    if (id < POPULATION_SIZE) {
        curandState state;
        curand_init(id, 0, 0, &state);

        int parent1_idx = curand(&state) % BEST_COUNT;
        int parent2_idx = curand(&state) % BEST_COUNT;
        create_child(initial_board, best_individuals[parent1_idx], best_individuals[parent2_idx], population[id], &state);

        population[id].quality = count_empty_cells(population[id].board);
    }
}

void create_new_population_CUDA(const int initial_board[BOARD_SIZE][BOARD_SIZE], const Individual best_individuals[BEST_COUNT], Individual population[POPULATION_SIZE], float mutation_rate) {
    // Alokacja pamięci na GPU
    int(*d_initial_board)[BOARD_SIZE];
    cudaMalloc((void**)&d_initial_board, BOARD_SIZE * BOARD_SIZE * sizeof(int));
    Individual* d_best_individuals;
    cudaMalloc((void**)&d_best_individuals, BEST_COUNT * sizeof(Individual));
    Individual* d_population;
    cudaMalloc((void**)&d_population, POPULATION_SIZE * sizeof(Individual));

    // Kopiowanie danych z hosta do urządzenia
    cudaMemcpy(d_initial_board, initial_board, BOARD_SIZE * BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_best_individuals, best_individuals, BEST_COUNT * sizeof(Individual), cudaMemcpyHostToDevice);

    // Wywołanie kernela
    int threadsPerBlock = THREADS;
    int blocksPerGrid = (POPULATION_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    create_new_population_kernel << <blocksPerGrid, threadsPerBlock >> > (d_initial_board, d_best_individuals, d_population, mutation_rate);

    // Kopiowanie wyników z urządzenia do hosta
    cudaMemcpy(population, d_population, POPULATION_SIZE * sizeof(Individual), cudaMemcpyDeviceToHost);

    // Zwolnienie pamięci na GPU
    cudaFree(d_initial_board);
    cudaFree(d_best_individuals);
    cudaFree(d_population);
}

// Funkcja sprawdzająca, czy plansza Sudoku spełnia wszystkie reguły gry
bool is_valid_board(int board[BOARD_SIZE][BOARD_SIZE]) {
    // Sprawdzamy wiersze i kolumny
    for (int i = 0; i < BOARD_SIZE; ++i) {
        bool row_nums[BOARD_SIZE + 1] = { false };
        bool col_nums[BOARD_SIZE + 1] = { false };
        for (int j = 0; j < BOARD_SIZE; ++j) {
            // Sprawdzamy wiersz
            if (board[i][j] == 0 || row_nums[board[i][j]]) {
                return false; // Powtórzona liczba w wierszu lub zero
            }
            row_nums[board[i][j]] = true;
            // Sprawdzamy kolumnę
            if (board[j][i] == 0 || col_nums[board[j][i]]) {
                return false; // Powtórzona liczba w kolumnie lub zero
            }
            col_nums[board[j][i]] = true;
        }
    }

    // Sprawdzamy kwadraty 3x3
    for (int i = 0; i < BOARD_SIZE; i += 3) {
        for (int j = 0; j < BOARD_SIZE; j += 3) {
            bool square_nums[BOARD_SIZE + 1] = { false };
            for (int k = i; k < i + 3; ++k) {
                for (int l = j; l < j + 3; ++l) {
                    if (board[k][l] == 0 || square_nums[board[k][l]]) {
                        return false; // Powtórzona liczba w kwadracie 3x3 lub zero
                    }
                    square_nums[board[k][l]] = true;
                }
            }
        }
    }

    return true; // Plansza jest poprawna
}

int main() {
    int initial_board[BOARD_SIZE][BOARD_SIZE] = {
        {8, 0, 0, 0, 0, 0, 1, 4, 9},
        {0, 0, 0, 5, 0, 0, 0, 0, 2},
        {0, 0, 0, 0, 0, 7, 5, 0, 0},
        {0, 0, 0, 9, 0, 0, 7, 0, 5},
        {0, 0, 9, 0, 4, 0, 2, 0, 0},
        {1, 0, 2, 0, 0, 6, 0, 0, 0},
        {0, 0, 4, 6, 0, 0, 0, 0, 0},
        {3, 0, 0, 0, 0, 9, 0, 0, 0},
        {9, 2, 5, 0, 0, 0, 0, 0, 4}
    };

    Individual population[POPULATION_SIZE];
    Individual best_individuals[BEST_COUNT];

    auto start = std::chrono::steady_clock::now();
    // Generowanie początkowej populacji
    generate_first_population_CUDA(initial_board, population);
    
    for (int generation = 0; generation < GENERATIONS; ++generation) {
        // Wybieranie najlepszych osobników
        select_best_individuals(population, best_individuals);
        cout << "Generacja: " << generation << endl;
        print_board(best_individuals[0].board);
        cout << "Liczba pustych miejsc: " << best_individuals[0].quality << endl;
        if (best_individuals[0].quality == 0) {
            break;
        }
        // Tworzenie nowej populacji
        create_new_population_CUDA(initial_board, best_individuals, population, MUTATION_RATE);
    }
    Individual result = best_individuals[0];
    auto end = std::chrono::steady_clock::now();
    cout << "Is final board valid? " << (is_valid_board(result.board) ? "Yes" : "No") << endl;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double duration_seconds = static_cast<double>(duration.count()) / 1000.0; // Konwersja z milisekund na sekundy

    // Teraz możesz wyświetlić czas wykonania
    std::cout << "Time taken: " << duration_seconds << " seconds" << std::endl;

    return 0;
}

