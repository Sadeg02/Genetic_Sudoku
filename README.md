Genetic_Sudoku
===============

Projekt uczelniany: porównanie implementacji algorytmu genetycznego rozwiązującego Sudoku — wersje: single-thread, OpenMP i CUDA.

Opis projektu
-------------
Repo zawiera trzy implementacje algorytmu genetycznego, każda w osobnym pliku źródłowym:

- `Genetic.cpp` — implementacja jednowątkowa (C++17).
- `Genetic_OpenMP.cpp` — wersja równoległa z użyciem OpenMP.
- `Genetic_CUDA.cu` — wersja używająca CUDA (kernels tworzą populacje i generują potomków na GPU).

Każda implementacja generuje populację wypełnionych (częściowo) plansz Sudoku, selekcjonuje najlepsze osobniki i tworzy nowe pokolenia, próbując osiągnąć pełne, poprawne Sudoku.

Główne parametry (warto sprawdzić w plikach źródłowych)
------------------------------------------------------
- POPULATION_SIZE — rozmiar populacji (różni się między plikami, np. 300/500).
- GENERATIONS — maksymalna liczba generacji (np. 1000/10000).
- BEST_COUNT — liczba najlepszych osobników używanych do kreacji nowej populacji.
- MUTATION_RATE — prawdopodobieństwo mutacji.
- THREADS — liczba wątków dla implementacji OpenMP.

Pliki pomocnicze
-----------------
- `CMakeLists.txt` — prosty CMake tworzący target `Geny` wskazujący obecnie na `Genetic.cpp` (uwaga: `Genetic_OpenMP.cpp` jest zakomentowany). Brak domyślnego targetu CUDA w tym CMake.
- `sudoku.txt` — przykładowa plansza sudoku (9 wierszy po 9 liczb).
- `wynik.txt` — (istnieje w repo; sprawdź zawartość lokalnie).

Znane problemy i uwagi
----------------------
- W repo są drobne różnice w parametrach pomiędzy implementacjami — przed porównaniem warto je ujednolicić.
- `CMakeLists.txt` nie buduje OpenMP/CUDA automatycznie. Proponuję dodać trzy targety: `geny_single`, `geny_openmp` i `geny_cuda` i opcję `ENABLE_CUDA`.
- W `Genetic_OpenMP.cpp` użycie `rand()` nie jest thread-safe. Lepiej użyć `std::mt19937` z ziarnem zależnym od numeru wątku.
- W `Genetic_CUDA.cu` zauważyłem błędne parametry wywołania kernela (kolejność bloków/threads) i potencjalne ryzyko przy kopiowaniu tablic 2D — wymagane poprawki i sprawdzenie błędów CUDA (cudaGetLastError()).
- Na macOS dostępność CUDA może być ograniczona (brak wsparcia dla nowych GPU/sterowników). Sprawdź, czy posiadasz kompatybilny sprzęt i zainstalowany NVIDIA CUDA Toolkit.

Instrukcje instalacji (macOS)
-----------------------------
1. Zainstaluj Homebrew jeśli go nie masz:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Zainstaluj CMake:

```bash
brew install cmake
```

3. (Opcjonalnie) Jeśli chcesz kompilować wersję OpenMP i masz GCC z OpenMP: zainstaluj gcc:

```bash
brew install gcc
```

4. (Opcjonalnie) Aby użyć CUDA: zainstaluj CUDA Toolkit i upewnij się, że twoja maszyna ma kompatybilną kartę NVIDIA (uwaga: wiele Maców nie ma wsparcia CUDA).

Szybkie polecenia kompilacji i uruchomień
----------------------------------------
Poniższe komendy są minimalne; w razie potrzeby można rozszerzyć `CMakeLists.txt`.

Kompilacja jednowątkowa (g++):

```bash
g++ -std=c++17 -O2 -pthread Genetic.cpp -o geny_single
./geny_single
```

Kompilacja OpenMP (jeśli kompilator wspiera OpenMP):

```bash
g++ -std=c++17 -O2 -fopenmp Genetic_OpenMP.cpp -o geny_openmp
./geny_openmp
```

Kompilacja CUDA (jeśli masz nvcc i kompatybilne GPU):

```bash
nvcc -std=c++14 -O2 Genetic_CUDA.cu -o geny_cuda
./geny_cuda
```

Plan eksperymentów (szybka propozycja)
-------------------------------------
Cele: porównać czas wykonania i skuteczność (ile pustych pól pozostaje) między trzema implementacjami.

Metryki:
- time_sec — czas wykonania programu
- quality — liczba pustych pól w finalnym rozwiązaniu
- valid — czy finalne sudoku jest poprawne

Sugerowane zmienne eksperymentalne:
- POPULATION_SIZE: {100, 300, 500}
- GENERATIONS: {1000, 5000, 10000}
- MUTATION_RATE: {0.05, 0.2}
- THREADS (OpenMP): {1,2,4,8}
- powtórzenia: 5 uruchomień na kombinację

Wyjście: CSV z kolumnami: mode,population,generations,mutation,threads,run,time_sec,quality,valid
