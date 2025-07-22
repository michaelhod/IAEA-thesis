import numpy as np

def npz_equal(path1, path2, atol=0.0, rtol=0.0):
    """
    Return True if the two .npz files store the same data.

    Parameters
    ----------
    path1, path2 : str | Path
        Filenames of the .npz archives.
    atol, rtol : float
        Absolute / relative tolerances used when comparing
        floating-point arrays (`np.allclose`).  Set both to 0
        for exact equality.

    Notes
    -----
    * Dictionary keys (`npz.files`) must match *exactly*.
    * Each corresponding array must have the same dtype & shape.
    * Floating-point arrays are compared with `np.allclose`,
      everything else with `np.array_equal`.
    """
    with np.load(path1, allow_pickle=False) as f1, \
         np.load(path2, allow_pickle=False) as f2:

        # 1. Same set of array names?
        if set(f1.files) != set(f2.files):
            print("Files: ", f1.files, f2.files)
            return False

        # 2. Compare every array
        for key in f1.files:
            a, b = f1[key], f2[key]

            # Same shape & dtype?
            if a.shape != b.shape or a.dtype != b.dtype:
                print("Shape: ", a.shape, b.shape)
                return False

            # Numerical or exact comparison
            if np.issubdtype(a.dtype, np.inexact):
                if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                    print("value: ", a, b)
                    return False
            else:
                if not np.array_equal(a, b):
                    print("value: ", a, b)
                    return False
    return True

print(npz_equal("data/swde_HTMLgraphs/nbaplayer/nbaplayer/nbaplayer-foxsports(425)/0006/A.npz", "data/swde_HTMLgraphs/nbaplayer/nbaplayer/nbaplayer-foxsports(425)/0000/A.npz"))