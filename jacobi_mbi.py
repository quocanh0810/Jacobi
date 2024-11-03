import numpy as np
import math
import matplotlib.pyplot as plt

# import MPI
from mpi4py import MPI

import time
start = time.time()

# NxN matrix
MAX_N = 40

# Matrix for jacobi calculation input and output
A = np.zeros((MAX_N-2, MAX_N-2))
A = np.pad(A, pad_width=1, mode='constant', constant_values=1)

# Matrix for jacobi calculation output temp
(row_num, col_num) = A.shape
B = np.zeros((row_num, col_num))

# MPI communication object, get information
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print('Num of process', size, ', current rank', rank)

# for each process, allocate A matrix divided by size
if rank == (size-1):
    local_row_num = int(MAX_N / size) + (MAX_N % size)
else:
    local_row_num = int(MAX_N / size)

# allocate local memory for each process
# range = 'local_first_row' to 'local_last_row' - 1
local_first_row = rank * int(MAX_N / size)
local_last_row = local_first_row + local_row_num 
A_local = A[local_first_row:local_last_row, :]
B_local = B[local_first_row:local_last_row, :]

#Jacobi loop
converge = False
iteration_num = 0
while (converge == False):
    iteration_num = iteration_num+1
    diffnorm = 0.0

    A_local_padding = np.pad(A_local, pad_width=1, mode='constant', constant_values=0)

    # Send and Recv border region of A
    if rank > 0:
        comm.Send(A_local[0, :], dest = rank-1, tag = 11)

        tmp = np.empty(MAX_N, dtype = A_local_padding.dtype)
        comm.Recv(tmp, source = rank - 1, tag = 22)
        A_local_padding[0, 1:MAX_N+1] = tmp
    if rank < (size - 1):
        comm.Send(A_local[local_row_num - 1, :], dest = rank + 1, tag = 22)

        tmp = np.empty(MAX_N, dtype = A_local_padding.dtype)
        comm.Recv(tmp, source = rank + 1, tag = 11)
        A_local_padding[local_row_num +1, 1:MAX_N+1] = tmp

    # Jacobi process:
    (row_num, col_num) = B_local.shape
    for i in range(row_num):
        for j in range(col_num):
            # because we do padding, index changed
            idx_i_A = i + 1
            idx_j_A = j + 1
            # Do jacobi
            B_local[i][j] = 0.25*(A_local_padding[idx_i_A+1, idx_j_A]
                                + A_local_padding[idx_i_A-1, idx_j_A]
                                + A_local_padding[idx_i_A, idx_j_A+1]
                                + A_local_padding[idx_i_A, idx_j_A-1])
            # simple converge test
            diffnorm += math.sqrt((B_local[i, j] - A_local[i, j])*(B_local[i, j] - A_local[i, j]))
    A_local = np.copy(B_local)

    # check converge
    diffnorm_glov = comm.allreduce(diffnorm, op = MPI.SUM)
    if diffnorm_glov <= 0.0001:
        print('Converge, iteration : %d per process' % iteration_num)
        print('Error : %f' % diffnorm_glov)
        converge = True

# Gather to root process and show
comm.Gatherv(A_local, [A, MPI.DOUBLE], root=0)

if rank == 0:
    end = time.time()
    print('execution time : ')
    print(end - start)
    plt.imshow(A, cmap='gray', interpolation='nearest')
    plt.show()