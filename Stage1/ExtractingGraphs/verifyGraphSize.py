from scipy import sparse

def verify_A_size(size, path, RAISE=True):
    A_csr = sparse.load_npz(path)
    A_coo = A_csr.tocoo()
    if RAISE and size != A_coo.shape[0]:
        raise Exception(F"The no. of html tags, {size}, does not match the length of the Adj matrix, {A_coo.shape[0]}")
    return size == A_coo.shape[0]