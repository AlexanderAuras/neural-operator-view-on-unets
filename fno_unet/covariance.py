import numpy as np

def generate_covmatrix(size):
    
    K = (size-1)//2
    odd_matrix = np.ones((2*K+1, 2*K+1))/(2*K+1)**2
    
    for j in np.arange(1,K+1):
        start = int(j)
        end = int(2*K+1 - j)
        odd_matrix[start:end, start:end] = 1/(2*(K-j)+1)**2

    if size%2 == 0:
        even_matrix = np.ones((size,size))/(size+1)**2
        even_matrix[1:,1:] = odd_matrix
        matrix = even_matrix
    else:
        matrix = odd_matrix

    return np.fft.ifftshift(matrix)