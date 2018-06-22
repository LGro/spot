from numba import jit


@jit(nopython=True)
def xcorr_list(in1, in2):
    """List of all time delays from a full cross correlation of the two inputs

    Parameters
    ----------
    in1 : numpy.ndarray
        Occurence times / indices
    in2 : numpy.ndarray
        Occurence times / indices
    """

    n1 = len(in1)
    n2 = len(in2)

    C = [0.0]*(n1*n2)
    for i in range(n1):
        for j in range(n2):
            C[i*n2+j] = in2[j] - in1[i]

    return C