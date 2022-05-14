#!/bin/bash
#
#SBATCH -t 48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=2n2p.out
#SBATCH --error=2n2p.err
#SBATCH -p ctessum 

mpiexec -n 2 python3 ../mpi.py