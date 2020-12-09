import numpy as np

def DFT(img:np.ndarray):
    M = img.shape[0]
    N = img.shape[1]
    ans = np.zeros(img.shape,dtype=np.complex)
    for u in range(0,M):
        for v in range(0,N):
            sum = 0
            for x in range(0,M):
                for y in range(0,N):
                    sum += img[x][y]* np.exp(-1j * 2 * np.pi*(u*x/M + v* y/N ))
            ans[u][v] = sum

    return ans


def IDFT(img: np.ndarray):
    M = img.shape[0]
    N = img.shape[1]
    ans = np.zeros(img.shape,dtype=np.complex)
    for u in range(0, M):
        for v in range(0, N):
            sum = 0
            for x in range(0, M):
                for y in range(0, N):
                    sum += img[x][y] * np.exp(1j * 2 * np.pi*(u * x / M + v * y / N))
            ans[u][v] = sum

    return ans * (1/(M*N))

x = np.eye(8)
y = IDFT(DFT(np.array(x)))
print(y)