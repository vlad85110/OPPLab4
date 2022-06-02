#include <stdio.h>
#include "iterations.h"

int main() {
    MPI_Init(NULL, NULL);
    iterations();
    MPI_Finalize();
    return 0;
}
