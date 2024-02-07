from asyncore import write
import numpy as np
import colorama
from colorama import Fore

TOKENS = 8
SA_SIZE = 4


def linearize(matrix):
    ref_reshaped = []
    for diag in range(TOKENS-SA_SIZE+1):
        for i in range(diag, diag+SA_SIZE):
            for j in range(diag, diag+SA_SIZE):
                if(diag == 0):
                    ref_reshaped.append(matrix[i][j])
                else:
                    if(j==diag+SA_SIZE-1):
                        ref_reshaped.append(matrix[i][j])
                    elif(i==diag+SA_SIZE-1):
                        if(j==diag):
                            ref_reshaped.append(matrix[i][diag+SA_SIZE-1])
                        ref_reshaped.append(matrix[i][j])
    return np.array(ref_reshaped).reshape(-1,SA_SIZE)


def maskout(matrix, TOKENS, SA_SIZE):
    for i in range(TOKENS):
        for j in range(TOKENS):
            if(i >= j+SA_SIZE):
                matrix[i][j] = 0
            if(j >= i+SA_SIZE):
                matrix[i][j] = 0
    return matrix


def print_matrix(matrix):
    rows,cols = matrix.shape
    for r in range(rows):
        for c in range(cols):
            col = Fore.RED if (matrix[r,c] == 0) else Fore.BLUE
            print(col + "{:.0f}".format(matrix[r,c]), end=' ')
        print()
    # print()

if __name__ == '__main__':
    matrix = np.loadtxt("hls.txt").reshape(TOKENS, TOKENS)
    DTYPE = np.float32
    matrix = np.where(matrix==0., -np.inf, matrix)
    matrix = np.exp(matrix)
    matrix = np.clip(matrix, -128, 128-2**-8)
    sums = matrix.sum(axis=-1)
    matrix /= sums.reshape(-1,1)
    matrix = np.round(matrix*2**8)/2**8
    
    # matrix = np.random.randint(1,9, (TOKENS,TOKENS))
    # DTYPE = np.int32
    
    maskout(matrix)
    print("\nOrignal Matrix: \n")
    print_matrix(matrix)
    print("\n")
    linear_matrix = linearize(matrix)
    # print("\nlinearized Matrix: \n")
    # print_matrix(linear_matrix)
    total_read_cycles = 2*TOKENS-SA_SIZE
    assert total_read_cycles == linear_matrix.shape[0]
    buffer_capacity = SA_SIZE+SA_SIZE-2
    buffer = np.zeros((buffer_capacity,SA_SIZE), dtype=DTYPE)
    sum_buffer = np.zeros((1,SA_SIZE+SA_SIZE-1), dtype=DTYPE)
    sums = np.zeros((TOKENS,), dtype=DTYPE)

    stream_read_idx = 0
    buffer_store_idx = 0
    buffer_read_idx = 0
    sum_buffer_idx = 0
    on_odd = 1
    shift = False
    write_output = 0
    col_elem_counters = np.zeros(shape=(SA_SIZE-1,), dtype=np.int32)
    col_ind_counters = np.zeros(shape=(SA_SIZE-1,), dtype=np.int32)
    for i in range(SA_SIZE-1):
        col_elem_counters[i] = -i-1
        idx = (SA_SIZE + 2*i)
        if(idx >= buffer_capacity):
            idx -= buffer_capacity
        col_ind_counters[i] = idx

    for i in range(total_read_cycles+(SA_SIZE-1)):    
        if(i<total_read_cycles):
            # storing the values in buffer
            elem = linear_matrix[stream_read_idx]
            stream_read_idx+=1
            buffer[buffer_store_idx] = elem
            buffer_store_idx+=1
            if buffer_store_idx == buffer_capacity:
                buffer_store_idx = 0 
            if(sum_buffer_idx==0):
                write_output=1
        else:
            # assign columns
            write_output = 1
            for s in range(SA_SIZE-1):#UNROLL
                elem_idx = col_elem_counters[s]
                elem_idx += 1
                col_elem_counters[s] = elem_idx
                if(elem_idx == SA_SIZE-1):
                    shift = True

            # move columns if necessry
            if (shift):
                for sh in range(SA_SIZE-1): #UNROLL
                    col_elem_counters[sh] = col_elem_counters[sh+1] if sh < SA_SIZE-2 else -SA_SIZE
                    col_ind_counters[sh]  = col_ind_counters[sh+1]  if sh < SA_SIZE-2 else 0
                shift = False
            
            buffer_read_idx += 2
            if (buffer_read_idx >= buffer_capacity):
                buffer_read_idx -= buffer_capacity

        if(i>SA_SIZE-1 and i<total_read_cycles):
            if(on_odd%2 == 1):
                write_output = 1
                for s in range(SA_SIZE-1):#UNROLL
                    elem_idx = col_elem_counters[s]
                    elem_idx += 1
                    col_elem_counters[s] = elem_idx
                    if(elem_idx == SA_SIZE-1):
                        shift = True

                # move columns if necessry
                if (shift):
                    for sh in range(SA_SIZE-1): #UNROLL
                        col_elem_counters[sh] = col_elem_counters[sh+1] if sh < SA_SIZE-2 else 0
                        idx = (col_ind_counters[sh] + 2)
                        if(idx >= buffer_capacity):
                            idx -= buffer_capacity
                        col_ind_counters[sh]  = col_ind_counters[sh+1]  if sh < SA_SIZE-2 else idx
                    shift = False                
                
                if(sum_buffer_idx > SA_SIZE-1):
                    buffer_read_idx += 2
                    if (buffer_read_idx >= buffer_capacity):
                        buffer_read_idx -= buffer_capacity
                else:
                    buffer_read_idx +=1
            on_odd += 1

        # assign row
        row = buffer[buffer_read_idx,:]
        sum_buffer[:,:SA_SIZE] = row

        # assign columns
        for s in range(SA_SIZE-1): # UNROLL?
            elem_idx = col_elem_counters[s]
            col_idx = col_ind_counters[s]
            sum_buffer[:,SA_SIZE+s] = 0 if elem_idx < 0 else buffer[col_idx,elem_idx]

        if(write_output):
            write_output = 0
            print_matrix(sum_buffer)
            sums[sum_buffer_idx] = sum_buffer.sum()
            sum_buffer_idx += 1

    ref = matrix.sum(axis=-1)
    # print_matrix(ref[None, ...])
    # print_matrix(sums[None, ...])
    assert np.allclose(ref, sums)
    
            

        







