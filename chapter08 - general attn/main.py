import numpy as np
import scipy.special

def main():
    words = np.array([[1,0,0], [0,1,0],[1,1,0],[0,0,1]])

    np.random.seed(42)
    w_q = np.random.randint(3, size=(3,3))
    w_k = np.random.randint(3, size=(3,3))
    w_v = np.random.randint(3, size=(3,3))

    q = words @ w_q
    k = words @ w_k
    v = words @ w_v

    scores = q @ k.T

    wei = scipy.special.softmax(scores / k.shape[1] ** 0.5, axis=1)

    ctx = wei @ v
    print(ctx)

    return


if __name__ == '__main__':
    main()

