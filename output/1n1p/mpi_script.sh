#!/bin/bash
#
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=1n1p.out
#SBATCH --error=1n1p.err
#SBATCH -p ctessum 

mpiexec -n 1 python3 ../mpi.py