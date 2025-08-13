#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define NUM_ATR 3

int main()
{
    int N_visitantes = 50000;
    int minutos_simulacion = 2000;
    double precio_ticket = 10.0;

    int capacidad_atr[NUM_ATR] = {24, 40, 18};
    double prob_falla_base[NUM_ATR] = {0.02, 0.01, 0.015};

    double ingresos_totales = 0.0;
    double satisfaccion_total = 0.0;
    int fallas_totales = 0;
    int atendidos_totales = 0;
    int fallas_por_atr[NUM_ATR] = {0};
    double satisf_por_atr[NUM_ATR] = {0.0};

    double t0 = omp_get_wtime();
    printf("Hilos (max): %d\n", omp_get_max_threads());

#pragma omp parallel sections shared(N_visitantes, minutos_simulacion, precio_ticket, capacidad_atr, prob_falla_base, ingresos_totales, satisfaccion_total, fallas_totales, atendidos_totales)
    {
// taquilla
#pragma omp section
        {
#pragma omp parallel for reduction(+ : ingresos_totales, atendidos_totales)
            for (int i = 0; i < N_visitantes; i++)
            {
                ingresos_totales += precio_ticket;
                atendidos_totales += 1;
            }
        }

// Operación de atracciones
#pragma omp section
        {
            unsigned base_seed_op = (unsigned)time(NULL) + 123u;
#pragma omp parallel shared(capacidad_atr, prob_falla_base, fallas_por_atr, satisf_por_atr, minutos_simulacion)
            {
                unsigned myseed = base_seed_op + 1009u * (unsigned)omp_get_thread_num();
                int fallas_local[NUM_ATR] = {0};
                double satisf_local[NUM_ATR] = {0.0};

                for (int minuto = 0; minuto < minutos_simulacion; minuto++)
                {
#pragma omp for reduction(+ : satisfaccion_total, fallas_totales)
                    for (int j = 0; j < NUM_ATR; j++)
                    {
                        int base = capacidad_atr[j];
                        int vari = (rand_r(&myseed) % (base / 2)) - (base / 4); // +/-25%
                        int served = base + vari;
                        if (served < 0)
                            served = 0;

                        int r = rand_r(&myseed);
                        double p = (double)r / (double)RAND_MAX;
                        int fallo = p < prob_falla_base[j];
                        double delta = fallo ? (-0.5 * served) : (0.8 * served);

                        if (fallo)
                        {
                            fallas_totales += 1;
                            fallas_local[j] += 1;
                        }
                        satisfaccion_total += delta;
                        satisf_local[j] += delta;
                    }
                }
#pragma omp critical
                {
                    for (int j = 0; j < NUM_ATR; j++)
                    {
                        fallas_por_atr[j] += fallas_local[j];
                        satisf_por_atr[j] += satisf_local[j];
                    }
                }
            }
        }

// Mantenimiento / Limpieza
#pragma omp section
        {
            int rondas = 40;
            unsigned base_seed_m = (unsigned)time(NULL) + 321u;
            for (int r = 0; r < rondas; r++)
            {
#pragma omp parallel
                {
                    unsigned myseed = base_seed_m + 2003u * (unsigned)omp_get_thread_num() + (unsigned)r;
#pragma omp for
                    for (int k = 0; k < NUM_ATR; k++)
                    {
                        int rr = rand_r(&myseed);
                        double mejora = 0.000005 + ((double)rr / (double)RAND_MAX) * 0.00002;

#pragma omp critical
                        {
                            prob_falla_base[k] -= mejora;
                            if (prob_falla_base[k] < 0.00005)
                                prob_falla_base[k] = 0.00005;
                        }
                    }
                }
            }
        }
    }

    double t1 = omp_get_wtime();

    printf("=== Resultados ===\n");
    printf("Visitantes atendidos: %d\n", atendidos_totales);
    printf("Ingresos totales:     %.2f\n", ingresos_totales);
    printf("Satisfaccion total:   %.2f\n", satisfaccion_total);
    printf("Fallas totales:       %d\n", fallas_totales);
    printf("Tiempo (s):           %.3f\n", t1 - t0);

    for (int j = 0; j < NUM_ATR; j++)
    {
        printf("Prob. falla final atraccion %d: %.6f\n", j, prob_falla_base[j]);
    }
    printf("\n--- Desglose por atracción ---\n");
    for (int j = 0; j < NUM_ATR; j++)
    {
        printf("Atracción %d -> Fallas: %d, Satisfacción: %.2f\n", j, fallas_por_atr[j], satisf_por_atr[j]);
    }
    return 0;
}