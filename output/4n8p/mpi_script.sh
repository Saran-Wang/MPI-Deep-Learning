#!/bin/bash
#
#SBATCH -t 48:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --output=4n8p.out
#SBATCH --error=4n8p.err
#SBATCH -p ctessum 

mpiexec -n 8 python3 ../mpi.py