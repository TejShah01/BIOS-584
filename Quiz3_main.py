from self_py_fun.Quiz3Fun import *

# You can use this .py script to perform a debugging task.
sample_arr_1 = np.array([1,2,3,4,5])
signal_diff_one = sample_arr_1[:-1] - sample_arr_1[1:]
D_val = np.sum(np.sqrt(1+signal_diff_one**2))
print(D_val)
#d_1 = compute_D_partial(sample_arr_1)




# The correct d_1 should be 5.66.
