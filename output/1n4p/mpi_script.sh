#!/bin/bash
#
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=1n4p.out
#SBATCH --error=1n4p.err
#SBATCH -p ctessum 

mpiexec -n 4 python3 ../mpi.py