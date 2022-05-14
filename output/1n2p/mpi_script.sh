#!/bin/bash
#
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=1n2p.out
#SBATCH --error=1n2p.err
#SBATCH -p ctessum 

mpiexec -n 2 python3 ../mpi.py