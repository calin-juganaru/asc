#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>     // provides int8_t, uint8_t, int16_t etc.
#include <stdlib.h>

#define DIE(assertion, call_description) \
	do                                   \
    {								     \
		if ((assertion))                 \
        {				                 \
			fprintf(stderr, "(%s, %d): ",\
					__FILE__, __LINE__); \
			perror(call_description);    \
			exit(1);				     \
		}						         \
	}                                    \
    while (0)

typedef struct
{
    int8_t v_x, v_y, v_z;
} particle;

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("apelati cu %s <n>\n", argv[0]);
        return -1;
    }

    long n = atol(argv[1]);

    // TODO - alocati dinamic o matrice de n x n elemente de tip struct particle
    // verificati daca operatia a reusit
    particle* mat = malloc(n * n * sizeof(particle*));
    DIE(mat == NULL, "matrix malloc");

    // TODO - populati matricea alocata astfel:
    // *liniile pare contin particule cu toate componentele vitezei pozitive
    //   -> folositi modulo 128 pentru a limita rezultatului lui rand()
    // *liniile impare contin particule cu toate componentele vitezi negative
    //   -> folositi modulo 129 pentru a limita rezultatului lui rand()
    for (int i = 0; i < n * n; ++i)
        for (int j = 0; j < n; ++j)
        {
            if (i % 2)
            {
                mat[n * i + j].v_x = (uint8_t)rand() % 129;
                mat[n * i + j].v_y = (uint8_t)rand() % 129;
                mat[n * i + j].v_z = (uint8_t)rand() % 129;
            }
            else
            {
                mat[n * i + j].v_x = (uint8_t)rand() % 128;
                mat[n * i + j].v_y = (uint8_t)rand() % 128;
                mat[n * i + j].v_z = (uint8_t)rand() % 128;
            }
        }

    // TODO
    // scalati vitezele tuturor particulelor cu 0.5
    //   -> folositi un cast la int8_t* pentru a parcurge vitezele fara
    //      a fi nevoie sa accesati individual componentele v_x, v_y, si v_z
    int8_t* aux = (int8_t*)mat;
    for (int i = 0; i < n * n; ++i, ++aux)
        *aux >>= 1;

    // compute max particle speed
    float max_speed = 0.0f;
    for(long i = 0; i < n * n; ++i)
    {
        float speed = sqrt(mat[i].v_x * mat[i].v_x +
                           mat[i].v_y * mat[i].v_y +
                           mat[i].v_z * mat[i].v_z);
        if(max_speed < speed) max_speed = speed;
    }

    // print result
    printf("viteza maxima este: %f\n", max_speed);

    free(mat);

    return 0;
}

