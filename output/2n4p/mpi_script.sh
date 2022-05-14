#!/bin/bash
#
#SBATCH -t 48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --output=2n4p.out
#SBATCH --error=2n4p.err
#SBATCH -p ctessum 

mpiexec -n 4 python3 ../mpi.py