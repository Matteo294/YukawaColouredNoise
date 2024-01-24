import ctypes

lib_path = "./"

class ACreturn(ctypes.Structure):
	_fields_ = [ ("average", ctypes.c_double), ("sigmaF", ctypes.c_double), ("sigmaF_error", ctypes.c_double), ("ACtime", ctypes.c_double), ("ACtimeError", ctypes.c_double) ]


def ComputeStatistics(data):
	ft_lib = ctypes.cdll.LoadLibrary(lib_path + "libac.so")

	arg = (ctypes.c_double * len(data))(*data)
	avg = ft_lib.Wrapper
	avg.restype = ACreturn

	return avg(arg, len(data))

# example usage:
# data = [...] #some data
# res = ComputeStatistics(data)
# print(res.average) #same for the other elements of ACreturn
