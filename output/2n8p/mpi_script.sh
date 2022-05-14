#!/bin/bash
#
#SBATCH -t 48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --output=2n8p.out
#SBATCH --error=2n8p.err
#SBATCH -p ctessum 

mpiexec -n 8 python3 ../mpi.py

