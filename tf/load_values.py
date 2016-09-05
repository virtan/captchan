import numpy as np
import gc

def load_data_n_values(filename, value_size = 1, data_size = 60*60):
    print "Loading values ..."
    raw_values = np.loadtxt(filename + '_values.gz')
    amount = raw_values.size / value_size
    print amount, "loaded"
    print "Reshaping ..."
    values = np.reshape(raw_values, (amount, value_size))
    raw_values = 0
    gc.collect()
    print "Loading data ..."
    raw_data = np.loadtxt(filename + '_data.gz')
    print amount, "loaded"
    print "Reshaping ..."
    data = np.reshape(raw_data, (amount, data_size))
    raw_data = 0
    gc.collect()
    print "Done"
    return values, data, amount
