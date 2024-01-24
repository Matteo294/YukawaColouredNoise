#pragma once

#include <cuda_runtime.h>

template <typename T>
class ManagedVector {
public:
	ManagedVector(int const size_)
		: ptr{[size_](){T* tmp; cudaMallocManaged(&tmp, sizeof(T) * size_); return tmp;}()}
		, size_{size_}
	{}
	~ManagedVector() { cudaFree(ptr); }

	inline T& operator[](int const i) { return ptr[i]; }
	inline T operator[](int const i) const { return ptr[i]; }

	inline T*const& data() const { return ptr; }
	//inline void** blob() { return reinterpret_cast<void**>(&std::remove_const(ptr)); }
	inline void** blob() { return (void**)&ptr; }

	inline int size() const { return size_; }

//	using iter = std::;//std::vector<myType>::iterator;

	// I couldn't figure out how to do it in a simple way using proper iterators
	// TODO: try again to use iterators here
	T* begin() { return ptr; }
	T* end()   { return ptr+size_; }

private:
	T * const ptr;
	int const size_;
};

