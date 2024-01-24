#include "params.h"
#include "Laplace.h"

inline auto MulComponents = [](dimArray const& arr, int const startPosition)
{ int out = 1; for (auto it = arr.cbegin() + startPosition; it < arr.cend(); ++it) out *= *it;
	return out; };

dimArray IndexToCoords(int idx) {
	int i = nDim-1;
	std::array<int, nDim> out;
	while (i >= 0) {
		out[i] = idx % Sizes[i];
		idx /= Sizes[i];
		i--;
	}
	return out;
}

int CoordToIndex(dimArray const& coords) {
	int out = 0;
	for (int mu = 0; mu < nDim; ++mu) {
		out += coords[mu] * MulComponents(Sizes, mu+1);
		//out += coords[mu] * std::pow(Nx, nDim-mu-1);
	}
	return out;
}

enum class Direction { Positive, Negative };
dimArray ShiftCoord(dimArray coords, int const shiftDir, Direction const dir) {
	coords[shiftDir] += (dir == Direction::Positive ? 1 : -1);
	if (coords[shiftDir] < 0) coords[shiftDir] += Sizes[shiftDir];
	else if (coords[shiftDir] >= Sizes[shiftDir]) coords[shiftDir] %= Sizes[shiftDir];
	return coords;
}


Laplace::Laplace(int const n)
	: N{n}
	, row_indices{std::vector<int>{}}
	, col_indices{std::vector<int>{}}
	, vals{std::vector<myType>{}}
	, I{[n](){int *tmp; checkCuda(cudaMallocManaged(&tmp, sizeof(int)*(n+1))); return tmp;}()}
	, J{[n](){int *tmp; checkCuda(cudaMallocManaged(&tmp, sizeof(int)*(n * nElements))); return tmp;}()}
	, cval{[n](){myType *tmp; checkCuda(cudaMallocManaged(&tmp, sizeof(myType)*(n * nElements))); return tmp;}()}
{
	row_indices.resize(N+1);
	col_indices.resize(N * nElements);
	vals.resize(N * nElements);

//	cudaMallocManaged(&I, sizeof(int) * (N+1));
//	cudaMallocManaged(&J, sizeof(int) * N * nElements);
//	cudaMallocManaged(&cval, sizeof(myType) * N * nElements);
}

Laplace::~Laplace() {
	checkCuda(cudaFree(I));
	checkCuda(cudaFree(J));
	checkCuda(cudaFree(cval));
}

void Laplace::func2() {
	std::array<int, nElements> tmp_col_indices;

	auto row_it = row_indices.begin();
	*row_it = 0; ++row_it;

	auto col_it = col_indices.begin();
	auto val_it = vals.begin();

	for (int r = 0; r < N; ++r, ++row_it) {
		auto tmp_it = tmp_col_indices.begin();

		auto const coo = IndexToCoords(r);

		*tmp_it = r;  ++tmp_it;
		for (int dir = 0; dir < nDim; ++dir) {
			auto const cooP = ShiftCoord(coo, dir, Direction::Positive);
			auto const cooM = ShiftCoord(coo, dir, Direction::Negative);

			int const cP = CoordToIndex(cooP);
			int const cM = CoordToIndex(cooM);
			*tmp_it = cP; ++tmp_it;
			*tmp_it = cM; ++tmp_it;
		}

		std::sort(tmp_col_indices.begin(), tmp_col_indices.end());

		auto tmp_cit = tmp_col_indices.cbegin();
		for (; tmp_cit != tmp_col_indices.cend(); ++tmp_cit, ++col_it, ++val_it) {
			*val_it = (*tmp_cit != r ? 1.0 : -2.0*nDim);
			*col_it = *tmp_cit;
		}

		*row_it = nElements * (r+1);
	}

	std::copy(row_indices.cbegin(), row_indices.cend(), I);
	std::copy(col_indices.cbegin(), col_indices.cend(), J);
	std::copy(vals.cbegin(), vals.cend(), cval);
}
