#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <unordered_set>
#include <set>
#include <chrono>

using namespace std;

const int POPULATION_SIZE = 300;
const int GENERATIONS = 10000;
const int BEST_COUNT = 30;
const double MUTATION_RATE = 0.2;
const int BOARD_SIZE = 9;
const int SUBGRID_SIZE = 3;
const int NUM_RUNS = 10;

struct Individual {
    vector<vector<int>> board;
    int quality;
};

// Pomocnicza funkcja do drukowania planszy Sudoku
void print_board(const vector<vector<int>>& board) {
    for (const auto& row : board) {
        for (const auto& cell : row) {
            cout << cell << " ";
        }
        cout << endl;
    }
}

// Funkcja zwracająca dostępne liczby dla danego pola
set<int> get_possible_numbers(const vector<vector<int>>& grid, int row, int col) {
    set<int> possible_numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Usuwanie liczb, które są już w danym wierszu
    for (int i = 0; i < BOARD_SIZE; ++i) {
        if (grid[row][i] != 0) {
            possible_numbers.erase(grid[row][i]);
        }
    }

    // Usuwanie liczb, które są już w danej kolumnie
    for (int i = 0; i < BOARD_SIZE; ++i) {
        if (grid[i][col] != 0) {
            possible_numbers.erase(grid[i][col]);
        }
    }

    // Usuwanie liczb, które są już w danym kwadracie 3x3
    int start_row = row / 3 * 3;
    int start_col = col / 3 * 3;
    for (int i = start_row; i < start_row + 3; ++i) {
        for (int j = start_col; j < start_col + 3; ++j) {
            if (grid[i][j] != 0) {
                possible_numbers.erase(grid[i][j]);
            }
        }
    }

    return possible_numbers;
}

vector<vector<set<int>>> calculate_possible_numbers(const vector<vector<int>>& grid) {
    vector<vector<set<int>>> possible_numbers_grid(BOARD_SIZE, vector<set<int>>(BOARD_SIZE));

    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            if (grid[row][col] == 0) {
                possible_numbers_grid[row][col] = get_possible_numbers(grid, row, col);
            }
        }
    }

    return possible_numbers_grid;
}


// Funkcja do wypełniania pojedynczego Sudoku na podstawie dostępnych liczb
vector<vector<int>> fill_sudoku(const vector<vector<int>>& initial_board) {
    vector<vector<int>> grid = initial_board; // Skopiowanie planszy początkowej
    vector<vector<set<int>>> possible_numbers_grid = calculate_possible_numbers(grid);

    // Losowo ustawiamy kolejność próby wstawienia liczby w każdą komórkę
    vector<pair<int, int>> cell_order;
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            cell_order.emplace_back(row, col);
        }
    }
    shuffle(cell_order.begin(), cell_order.end(), std::mt19937(std::random_device()()));

    for (const auto& [row, col] : cell_order) {
        if (grid[row][col] == 0) {
            if (possible_numbers_grid[row][col].empty()) {
                break; // Przerwij próbę wypełnienia, jeśli brak dostępnych liczb
            }

            // Losowanie liczby z dostępnych rozwiązań w danym miejscu
            int num_index = rand() % possible_numbers_grid[row][col].size();
            auto it = possible_numbers_grid[row][col].begin();
            advance(it, num_index);
            int num = *it;

            grid[row][col] = num;

            // Aktualizacja możliwych liczb po wstawieniu nowej liczby
            for (int i = 0; i < BOARD_SIZE; ++i) {
                possible_numbers_grid[row][i].erase(num);
                possible_numbers_grid[i][col].erase(num);
            }
            int start_row = row / 3 * 3;
            int start_col = col / 3 * 3;
            for (int i = start_row; i < start_row + 3; ++i) {
                for (int j = start_col; j < start_col + 3; ++j) {
                    possible_numbers_grid[i][j].erase(num);
                }
            }
        }
    }

    return grid; // Zwróć wypełnioną planszę
}


// Funkcja obliczająca ilość pustych pól w planszy
int count_empty_cells(const vector<vector<int>>& board) {
    int empty_cells = 0;
    for (const auto& row : board) {
        for (int cell : row) {
            if (cell == 0) {
                empty_cells++;
            }
        }
    }
    return empty_cells;
}

// Funkcja do generowania populacji o określonym rozmiarze
vector<Individual> generate_first_population(const vector<vector<int>>& initial_board) {
    cout<<"Intial Board: "<<endl;
    print_board(initial_board);
    cout<<endl;
    vector<Individual> population;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        Individual individual;
        individual.board = fill_sudoku(initial_board); // Wypełnij planszę Sudoku dla nowego osobnika
        individual.quality = count_empty_cells(individual.board); // Oblicz jakość planszy dla nowego osobnika
        population.push_back(individual); // Dodaj nowego osobnika do populacji
    }
    return population;
}


// Funkcja wybierająca najlepsze osobniki z populacji na podstawie jakości planszy
vector<Individual> select_best_individuals(const vector<Individual>& population) {
    vector<Individual> best_individuals;

    // Tworzymy kopię populacji, aby nie zmieniać kolejności oryginalnej
    vector<Individual> sorted_population = population;

    // Sortujemy populację według jakości planszy (rosnąco)
    sort(sorted_population.begin(), sorted_population.end(), [](const Individual& a, const Individual& b) {
        return a.quality < b.quality;
    });

    // Wybieramy najlepsze jednostki
    for (int i = 0; i < BEST_COUNT && i < sorted_population.size(); ++i) {
        best_individuals.push_back(sorted_population[i]);
    }

    return best_individuals;
}


vector<vector<int>> create_child(const vector<vector<int>>& initial_board,const vector<vector<int>>& parent1,const vector<vector<int>>& parent2) {
    vector<vector<int>> grid = initial_board; // Skopiowanie planszy początkowej
    vector<vector<set<int>>> possible_numbers_grid = calculate_possible_numbers(grid);

    // Losowo ustawiamy kolejność próby wstawienia liczby w każdą komórkę
    vector<pair<int, int>> cell_order;
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            cell_order.emplace_back(row, col);
        }
    }
    shuffle(cell_order.begin(), cell_order.end(), std::mt19937(std::random_device()()));

    for (const auto& [row, col] : cell_order) {
        if (grid[row][col] == 0) {
            if (possible_numbers_grid[row][col].empty()) {
                break; // Przerwij próbę wypełnienia, jeśli brak dostępnych liczb
            }

            // Losowanie liczby z dostępnych rozwiązań w danym miejscu
            int num;
            if (rand() < MUTATION_RATE * RAND_MAX) {
                // Mutacja: losujemy nową liczbę z dostępnych dla tego miejsca
                int num_index = rand() % possible_numbers_grid[row][col].size();
                auto it = possible_numbers_grid[row][col].begin();
                advance(it, num_index);
                num = *it;
            } else {
                // Wybieramy jedną z liczb dostępnych dla tego miejsca, którą posiadają rodzice
                if (possible_numbers_grid[row][col].count(parent1[row][col]) && possible_numbers_grid[row][col].count(parent2[row][col])) {
                    if (rand() % 2 == 0) {
                        num = parent1[row][col];
                    } else {
                        num = parent2[row][col];
                    }
                } else if (possible_numbers_grid[row][col].count(parent1[row][col])) {
                    num = parent1[row][col];
                } else if (possible_numbers_grid[row][col].count(parent2[row][col])) {
                    num = parent2[row][col];
                } else {
                    num = 0;
                }
            }

            grid[row][col] = num;

            // Aktualizacja możliwych liczb po wstawieniu nowej liczby
            for (int i = 0; i < BOARD_SIZE; ++i) {
                possible_numbers_grid[row][i].erase(num);
                possible_numbers_grid[i][col].erase(num);
            }
            int start_row = row / 3 * 3;
            int start_col = col / 3 * 3;
            for (int i = start_row; i < start_row + 3; ++i) {
                for (int j = start_col; j < start_col + 3; ++j) {
                    possible_numbers_grid[i][j].erase(num);
                }
            }
        }
    }

    return grid; // Zwróć wypełnioną planszę
}

vector<Individual> generate_population(const vector<vector<int>>& initial_board, const vector<Individual>& best_individuals) {
    vector<Individual> population;

    // Generujemy dzieci z najlepszych jednostek populacji
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        // Losowo wybieramy parę rodziców spośród najlepszych jednostek
        const auto& parent1 = best_individuals[rand() % BEST_COUNT];
        const auto& parent2 = best_individuals[rand() % BEST_COUNT];

        Individual individual;
        individual.board = create_child(initial_board, parent1.board, parent2.board); // Generujemy dziecko z rodziców
        individual.quality = count_empty_cells(individual.board); // Obliczamy jakość planszy dla nowego osobnika
        population.push_back(individual); // Dodajemy nowego osobnika do populacji
    }

    return population;
}

vector<vector<int>> genetic_algorithm(const vector<vector<int>>& initial_board) {
    vector<Individual> first_population = generate_first_population(initial_board); // Generujemy początkową populację
    vector<Individual> best_individuals = select_best_individuals(first_population); // Przechowujemy najlepsze jednostki z pierwszej
    vector<Individual> next_population;

    for (int generation = 0; generation < GENERATIONS; ++generation) {
        // Wyświetlamy informacje o aktualnej generacji
        cout << "Generation: " << generation + 1 << ", Best Quality: " << best_individuals[0].quality << endl;
        cout << "Best Quality Board" << endl;
        print_board(best_individuals[0].board);

        //przerywamy jesli osisagniemy pełne sudoku
        if(best_individuals[0].quality==0){
            break;
        }else {
            next_population.clear();
        }
        // Tworzymy nową populację z najlepszych jednostek
        next_population = generate_population(initial_board, best_individuals);
        best_individuals = select_best_individuals(next_population);
    }
    return best_individuals[0].board;
}




// Funkcja sprawdzająca, czy plansza Sudoku spełnia wszystkie reguły gry
bool is_valid_board(const vector<vector<int>>& board) {
    // Sprawdzamy wiersze i kolumny
    for (int i = 0; i < BOARD_SIZE; ++i) {
        set<int> row_nums;
        set<int> col_nums;
        for (int j = 0; j < BOARD_SIZE; ++j) {
            // Sprawdzamy wiersz
            if (board[i][j] != 0 && row_nums.count(board[i][j])) {
                return false; // Powtórzona liczba w wierszu
            }
            row_nums.insert(board[i][j]);
            // Sprawdzamy kolumnę
            if (board[j][i] != 0 && col_nums.count(board[j][i])) {
                return false; // Powtórzona liczba w kolumnie
            }
            col_nums.insert(board[j][i]);
        }
    }

    // Sprawdzamy kwadraty 3x3
    for (int i = 0; i < BOARD_SIZE; i += 3) {
        for (int j = 0; j < BOARD_SIZE; j += 3) {
            set<int> square_nums;
            for (int k = i; k < i + 3; ++k) {
                for (int l = j; l < j + 3; ++l) {
                    if (board[k][l] != 0 && square_nums.count(board[k][l])) {
                        return false; // Powtórzona liczba w kwadracie 3x3
                    }
                    square_nums.insert(board[k][l]);
                }
            }
        }
    }

    return true; // Plansza jest poprawna
}

int main() {
    // Przykładowa plansza Sudoku (0 oznacza puste miejsce)
    /*
    vector<vector<int>> initial_board = {
            {5, 3, 0, 0, 7, 0, 0, 0, 0},
            {6, 0, 0, 1, 9, 5, 0, 0, 0},
            {0, 9, 8, 0, 0, 0, 0, 6, 0},
            {8, 0, 0, 0, 6, 0, 0, 0, 3},
            {4, 0, 0, 8, 0, 3, 0, 0, 1},
            {7, 0, 0, 0, 2, 0, 0, 0, 6},
            {0, 6, 0, 0, 0, 0, 2, 8, 0},
            {0, 0, 0, 4, 1, 9, 0, 0, 5},
            {0, 0, 0, 0, 8, 0, 0, 7, 9}
    };
    */

/*
    vector<vector<int>> initial_board = {
            {5, 0, 0, 0, 7, 0, 0, 0, 0},
            {6, 0, 0, 1, 0, 5, 0, 0, 0},
            {0, 9, 0, 0, 0, 0, 0, 6, 0},
            {8, 0, 0, 0, 6, 0, 0, 0, 3},
            {0, 0, 0, 8, 0, 3, 0, 0, 1},
            {7, 0, 0, 0, 2, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 2, 8, 0},
            {0, 0, 0, 4, 1, 0, 0, 0, 5},
            {0, 0, 0, 0, 8, 0, 0, 7, 9}
    };
    */

    vector<vector<int>> initial_board = {
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


    auto start = std::chrono::steady_clock::now();
    vector<vector<int>> result = genetic_algorithm(initial_board);
    auto end = std::chrono::steady_clock::now();


    cout << "Is final board valid? " << (is_valid_board(result) ? "Yes" : "No") << endl;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double duration_seconds = static_cast<double>(duration.count()) / 1000.0; // Konwersja z milisekund na sekundy

// Teraz możesz wyświetlić czas wykonania
    std::cout << "Time taken: " << duration_seconds << " seconds" << std::endl;

    return 0;
}