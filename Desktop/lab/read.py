import numpy as np
def read_csv(csv_file_path):
    """
        Given a path to a csv file, return a matrix (list of lists)
        in row major.
    """    
    f1=open(csv_file_path,"rb")
    data=np.loadtxt(f1,delimiter=',',skiprows=0)
    f1.close()
    matrix=np.array(data)
    return matrix
