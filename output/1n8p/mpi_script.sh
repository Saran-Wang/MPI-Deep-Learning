#!/bin/bash
#
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=1n8p.out
#SBATCH --error=1n8p.err
#SBATCH -p ctessum 

mpiexec -n 8 python3 ../mpi.py