#include <mpich/mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>
#include "./src/mb.h"
#include "./src/globals.h"
#include "./src/bayes.h"
#include "./src/command.h"
#include "./src/mcmc.h"
//
// Created by bili on 2020/12/4.
//

int main(int argc, char *argv[]){
    int i;
#	if defined (MPI_ENABLED)
    int		ierror;
#	endif
    nBitsInALong = sizeof(long) * 8;
    if (nBitsInALong > 32) /* Do not use more than 32 bits until we    */
        nBitsInALong = 32; /* understand how 64-bit longs are handled. */
#	if defined (MPI_ENABLED)
        ierror = MPI_Init(&argc, &argv);
    if (ierror != MPI_SUCCESS)
    {
        MrBayesPrint ("%s   Problem initializing MPI\n", spacer);
        exit (1);
    }
    ierror = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (ierror != MPI_SUCCESS)
    {
        MrBayesPrint ("%s   Problem getting the number of processors\n", spacer);
        exit (1);
    }
    ierror = MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    if (ierror != MPI_SUCCESS)
    {
        MrBayesPrint ("%s   Problem getting processors rank\n", spacer);
        exit (1);
    }
#   endif
    /* Set up parameter table. */
    SetUpParms ();

    /* initialize seed using current time */
    GetTimeSeed ();

    /* Initialize the variables of the program. */
    InitializeMrBayes ();

    i = CommandLine (argc, argv);
#	if defined (MPI_ENABLED)
    /*Stop the MPI process*/
    MPI_Finalize();
#   endif
    if (i == ERROR)
        return (1);
    else
        return (0);
}




