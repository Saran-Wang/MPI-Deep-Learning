#!/bin/bash
#
#SBATCH -t 48:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=4n4p.out
#SBATCH --error=4n4p.err
#SBATCH -p ctessum 

mpiexec -n 4 python3 ../mpi.py