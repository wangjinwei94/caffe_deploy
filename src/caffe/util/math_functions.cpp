#include <random>
#include <cmath>
#include <limits>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// MKL_ALTERNATE
#ifdef USE_MKL

#include <mkl.h>

#else  // If use MKL, simply include the MKL header

#ifdef USE_EIGEN

#include <Eigen/Core>

typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> DynamicStride;

inline void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
		const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha,
		const float  *A, const int lda, const float  *B, const int ldb, const float beta, float  *C, const int ldc) {
	int mat_a_row=M;
	int mat_a_col=K;
	if(TransA!=CblasNoTrans) {
		std::swap(mat_a_row, mat_a_col);
	}
	int mat_b_row=K;
	int mat_b_col=N;
	if(TransB!=CblasNoTrans) {
		std::swap(mat_b_row, mat_b_col);
	}
	if(TransA==CblasConjTrans || TransB==CblasConjTrans || lda!=mat_a_col || ldb!=mat_b_col) {
		fprintf(stderr, "%s, %d\n", __FILE__, __LINE__);
		abort();
	}
	if(Order==CblasRowMajor) {
		Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_a(A, mat_a_row, mat_a_col);
		Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_b(B, mat_b_row, mat_b_col);
		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_c(C, M, N);
		mat_c*=beta;
		if(TransA==CblasNoTrans && TransB==CblasNoTrans) {
			mat_c.noalias() += alpha*mat_a*mat_b;
		}
		else if(TransA==CblasTrans && TransB==CblasNoTrans) {
			mat_c.noalias() += alpha*mat_a.transpose()*mat_b;
		}
		else if(TransA==CblasNoTrans && TransB==CblasTrans) {
			mat_c.noalias() += alpha*mat_a*mat_b.transpose();
		}
		else {
			mat_c.noalias() += alpha*mat_a.transpose()*mat_b.transpose();
		}
	}
	else {
		Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat_a(A, mat_a_row, mat_a_col);
		Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat_b(B, mat_b_row, mat_b_col);
		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat_c(C, M, N);
		mat_c*=beta;
		if(TransA==CblasNoTrans && TransB==CblasNoTrans) {
			mat_c.noalias() += alpha*mat_a*mat_b;
		}
		else if(TransA==CblasTrans && TransB==CblasNoTrans) {
			mat_c.noalias() += alpha*mat_a.transpose()*mat_b;
		}
		else if(TransA==CblasNoTrans && TransB==CblasTrans) {
			mat_c.noalias() += alpha*mat_a*mat_b.transpose();
		}
		else {
			mat_c.noalias() += alpha*mat_a.transpose()*mat_b.transpose();
		}
	}
}

inline void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
		const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha,
		const double  *A, const int lda, const double  *B, const int ldb, const double beta, double  *C, const int ldc) {
	int mat_a_row=M;
	int mat_a_col=K;
	if(TransA!=CblasNoTrans) {
		std::swap(mat_a_row, mat_a_col);
	}
	int mat_b_row=K;
	int mat_b_col=N;
	if(TransB!=CblasNoTrans) {
		std::swap(mat_b_row, mat_b_col);
	}
	if(TransA==CblasConjTrans || TransB==CblasConjTrans || lda!=mat_a_col || ldb!=mat_b_col) {
		fprintf(stderr, "%s, %d\n", __FILE__, __LINE__);
		abort();
	}
	if(Order==CblasRowMajor) {
		Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_a(A, mat_a_row, mat_a_col);
		Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_b(B, mat_b_row, mat_b_col);
		Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_c(C, M, N);
		mat_c*=beta;
		if(TransA==CblasNoTrans && TransB==CblasNoTrans) {
			mat_c.noalias() += alpha*mat_a*mat_b;
		}
		else if(TransA==CblasTrans && TransB==CblasNoTrans) {
			mat_c.noalias() += alpha*mat_a.transpose()*mat_b;
		}
		else if(TransA==CblasNoTrans && TransB==CblasTrans) {
			mat_c.noalias() += alpha*mat_a*mat_b.transpose();
		}
		else {
			mat_c.noalias() += alpha*mat_a.transpose()*mat_b.transpose();
		}
	}
	else {
		Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat_a(A, mat_a_row, mat_a_col);
		Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat_b(B, mat_b_row, mat_b_col);
		Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat_c(C, M, N);
		mat_c*=beta;
		if(TransA==CblasNoTrans && TransB==CblasNoTrans) {
			mat_c.noalias() += alpha*mat_a*mat_b;
		}
		else if(TransA==CblasTrans && TransB==CblasNoTrans) {
			mat_c.noalias() += alpha*mat_a.transpose()*mat_b;
		}
		else if(TransA==CblasNoTrans && TransB==CblasTrans) {
			mat_c.noalias() += alpha*mat_a*mat_b.transpose();
		}
		else {
			mat_c.noalias() += alpha*mat_a.transpose()*mat_b.transpose();
		}
	}
}

inline void cblas_sgemv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
		const int M, const int N, const float alpha, const float  *A, const int lda,
		const float  *X, const int incX, const float beta, float  *Y, const int incY) {
	if(TransA==CblasConjTrans) {
		fprintf(stderr, "%s, %d\n", __FILE__, __LINE__);
		abort();
	}
	int mat_a_row=M;
	int mat_a_col=N;
	if(TransA!=CblasNoTrans) {
		std::swap(mat_a_row, mat_a_col);
	}
	if(Order==CblasRowMajor) {
		Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_a(A, M, N);
		Eigen::Map<const Eigen::VectorXf, Eigen::Aligned, DynamicStride> mat_x(X, mat_a_col, DynamicStride(N*incX, incX));
		Eigen::Map<Eigen::VectorXf, Eigen::Aligned, DynamicStride> mat_y(Y, mat_a_row, DynamicStride(N*incY, incY));
		mat_y*=beta;
		if(TransA==CblasNoTrans) {
			mat_y.noalias() += alpha*mat_a*mat_x;
		}
		else {
			mat_y.noalias() += alpha*mat_a.transpose()*mat_x;
		}
	}
	else {
		Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat_a(A, M, N);
		Eigen::Map<const Eigen::VectorXf, Eigen::Aligned, DynamicStride> mat_x(X, mat_a_col, DynamicStride(N*incX, incX));
		Eigen::Map<Eigen::VectorXf, Eigen::Aligned, DynamicStride> mat_y(Y, mat_a_row, DynamicStride(N*incY, incY));
		mat_y*=beta;
		if(TransA==CblasNoTrans) {
			mat_y.noalias() += alpha*mat_a*mat_x;
		}
		else {
			mat_y.noalias() += alpha*mat_a.transpose()*mat_x;
		}
	}
}

inline void cblas_dgemv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
		const int M, const int N, const double alpha, const double  *A, const int lda,
		const double  *X, const int incX, const double beta, double  *Y, const int incY) {
	if(TransA==CblasConjTrans) {
		fprintf(stderr, "%s, %d\n", __FILE__, __LINE__);
		abort();
	}
	int mat_a_row=M;
	int mat_a_col=N;
	if(TransA!=CblasNoTrans) {
		std::swap(mat_a_row, mat_a_col);
	}
	if(Order==CblasRowMajor) {
		Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_a(A, M, N);
		Eigen::Map<const Eigen::VectorXd, Eigen::Aligned, DynamicStride> mat_x(X, mat_a_col, DynamicStride(N*incX, incX));
		Eigen::Map<Eigen::VectorXd, Eigen::Aligned, DynamicStride> mat_y(Y, mat_a_row, DynamicStride(N*incY, incY));
		mat_y*=beta;
		if(TransA==CblasNoTrans) {
			mat_y.noalias() += alpha*mat_a*mat_x;
		}
		else {
			mat_y.noalias() += alpha*mat_a.transpose()*mat_x;
		}
	}
	else {
		Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat_a(A, M, N);
		Eigen::Map<const Eigen::VectorXd, Eigen::Aligned, DynamicStride> mat_x(X, mat_a_col, DynamicStride(N*incX, incX));
		Eigen::Map<Eigen::VectorXd, Eigen::Aligned, DynamicStride> mat_y(Y, mat_a_row, DynamicStride(N*incY, incY));
		mat_y*=beta;
		if(TransA==CblasNoTrans) {
			mat_y.noalias() += alpha*mat_a*mat_x;
		}
		else {
			mat_y.noalias() += alpha*mat_a.transpose()*mat_x;
		}
	}
}

inline void cblas_saxpy(const int N, const float alpha, const float *X,
		const int incX, float *Y, const int incY) {
	Eigen::Map<const Eigen::VectorXf, Eigen::Aligned, DynamicStride> mat_x(X, N, DynamicStride(N*incX, incX));
	Eigen::Map<Eigen::VectorXf, Eigen::Aligned, DynamicStride> mat_y(Y, N, DynamicStride(N*incY, incY));
	mat_y+=alpha*mat_x;
}

inline void cblas_daxpy(const int N, const double alpha, const double *X,
		const int incX, double *Y, const int incY) {
	Eigen::Map<const Eigen::VectorXd, Eigen::Aligned, DynamicStride> mat_x(X, N, DynamicStride(N*incX, incX));
	Eigen::Map<Eigen::VectorXd, Eigen::Aligned, DynamicStride> mat_y(Y, N, DynamicStride(N*incY, incY));
	mat_y+=alpha*mat_x;
}

inline void cblas_sscal(const int N, const float alpha, float *X, const int incX) {
	Eigen::Map<Eigen::VectorXf, Eigen::Aligned, DynamicStride> mat_x(X, N, DynamicStride(N*incX, incX));
	mat_x*=alpha;
}

inline void cblas_dscal(const int N, const double alpha, double *X, const int incX) {
	Eigen::Map<Eigen::VectorXd, Eigen::Aligned, DynamicStride> mat_x(X, N, DynamicStride(N*incX, incX));
	mat_x*=alpha;
}

inline void cblas_saxpby(const int N, const float alpha, const float *X,
		const int incX, const float beta, float *Y, const int incY) {
	Eigen::Map<const Eigen::VectorXf, Eigen::Aligned, DynamicStride> mat_x(X, N, DynamicStride(N*incX, incX));
	Eigen::Map<Eigen::VectorXf, Eigen::Aligned, DynamicStride> mat_y(Y, N, DynamicStride(N*incY, incY));
	mat_y*=beta;
	mat_y+=alpha*mat_x;
}

inline void cblas_daxpby(const int N, const double alpha, const double *X,
		const int incX, const double beta, double *Y, const int incY) {
	Eigen::Map<const Eigen::VectorXd, Eigen::Aligned, DynamicStride> mat_x(X, N, DynamicStride(N*incX, incX));
	Eigen::Map<Eigen::VectorXd, Eigen::Aligned, DynamicStride> mat_y(Y, N, DynamicStride(N*incY, incY));
	mat_y*=beta;
	mat_y+=alpha*mat_x;
}

inline float cblas_sdot(const int N, const float *X, const int incX, const float *Y, const int incY) {
	Eigen::Map<const Eigen::VectorXf, Eigen::Aligned, DynamicStride> mat_x(X, N, DynamicStride(N*incX, incX));
	Eigen::Map<const Eigen::VectorXf, Eigen::Aligned, DynamicStride> mat_y(Y, N, DynamicStride(N*incY, incY));
	return mat_x.dot(mat_y);
}

inline double cblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY) {
	Eigen::Map<const Eigen::VectorXd, Eigen::Aligned, DynamicStride> mat_x(X, N, DynamicStride(N*incX, incX));
	Eigen::Map<const Eigen::VectorXd, Eigen::Aligned, DynamicStride> mat_y(Y, N, DynamicStride(N*incY, incY));
	return mat_x.dot(mat_y);
}

inline float cblas_sasum(const int N, const float *X, const int incX) {
	Eigen::Map<const Eigen::VectorXf, Eigen::Aligned, DynamicStride> mat_x(X, N, DynamicStride(N*incX, incX));
	return mat_x.cwiseAbs().sum();
}

inline double cblas_dasum(const int N, const double *X, const int incX) {
	Eigen::Map<const Eigen::VectorXd, Eigen::Aligned, DynamicStride> mat_x(X, N, DynamicStride(N*incX, incX));
	return mat_x.cwiseAbs().sum();
}

inline void cblas_scopy(const int N, const float *X, const int incX, float *Y, const int incY) {
	Eigen::Map<const Eigen::VectorXf, Eigen::Aligned, DynamicStride> mat_x(X, N, DynamicStride(N*incX, incX));
	Eigen::Map<Eigen::VectorXf, Eigen::Aligned, DynamicStride> mat_y(Y, N, DynamicStride(N*incY, incY));
	mat_y=mat_x;
}

inline void cblas_dcopy(const int N, const double *X, const int incX, double *Y, const int incY) {
	Eigen::Map<const Eigen::VectorXd, Eigen::Aligned, DynamicStride> mat_x(X, N, DynamicStride(N*incX, incX));
	Eigen::Map<Eigen::VectorXd, Eigen::Aligned, DynamicStride> mat_y(Y, N, DynamicStride(N*incY, incY));
	mat_y=mat_x;
}

#else

extern "C" {
#include <cblas.h>
}

// In addition, MKL comes with an additional function axpby that is not present
// in standard blas. We will simply use a two-step (inefficient, of course) way
// to mimic that.
inline void cblas_saxpby(const int N, const float alpha, const float* X,
		const int incX, const float beta, float* Y,
		const int incY) {
	cblas_sscal(N, beta, Y, incY);
	cblas_saxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_daxpby(const int N, const double alpha, const double* X,
		const int incX, const double beta, double* Y,
		const int incY) {
	cblas_dscal(N, beta, Y, incY);
	cblas_daxpy(N, alpha, X, incX, Y, incY);
}

#endif  // USE_EIGEN

#include <math.h>

// Functions that caffe uses but are not present if MKL is not linked.

// A simple way to define the vsl unary functions. The operation should
// be in the form e.g. y[i] = sqrt(a[i])
#define DEFINE_VSL_UNARY_FUNC(name, operation) \
	template<typename Dtype> \
	void v##name(const int n, const Dtype* a, Dtype* y) { \
		CHECK_GT(n, 0); CHECK(a); CHECK(y); \
		for (int i = 0; i < n; ++i) { operation; } \
	} \
	inline void vs##name( \
		const int n, const float* a, float* y) { \
		v##name<float>(n, a, y); \
	} \
	inline void vd##name( \
			const int n, const double* a, double* y) { \
		v##name<double>(n, a, y); \
	}

DEFINE_VSL_UNARY_FUNC(Sqr, y[i] = a[i] * a[i]);
DEFINE_VSL_UNARY_FUNC(Exp, y[i] = exp(a[i]));
DEFINE_VSL_UNARY_FUNC(Ln, y[i] = log(a[i]));
DEFINE_VSL_UNARY_FUNC(Abs, y[i] = fabs(a[i]));

// A simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. y[i] = pow(a[i], b)
#define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation) \
	template<typename Dtype> \
	void v##name(const int n, const Dtype* a, const Dtype b, Dtype* y) { \
		CHECK_GT(n, 0); CHECK(a); CHECK(y); \
		for (int i = 0; i < n; ++i) { operation; } \
	} \
	inline void vs##name( \
		const int n, const float* a, const float b, float* y) { \
		v##name<float>(n, a, b, y); \
	} \
	inline void vd##name( \
			const int n, const double* a, const float b, double* y) { \
		v##name<double>(n, a, b, y); \
	}

DEFINE_VSL_UNARY_FUNC_WITH_PARAM(Powx, y[i] = pow(a[i], b));

// A simple way to define the vsl binary functions. The operation should
// be in the form e.g. y[i] = a[i] + b[i]
#define DEFINE_VSL_BINARY_FUNC(name, operation) \
	template<typename Dtype> \
	void v##name(const int n, const Dtype* a, const Dtype* b, Dtype* y) { \
		CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(y); \
		for (int i = 0; i < n; ++i) { operation; } \
	} \
	inline void vs##name( \
		const int n, const float* a, const float* b, float* y) { \
		v##name<float>(n, a, b, y); \
	} \
	inline void vd##name( \
			const int n, const double* a, const double* b, double* y) { \
		v##name<double>(n, a, b, y); \
	}

DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i]);
DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i]);

#endif  // USE_MKL
// MKL_ALTERNATE

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return std::nextafter(b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  std::uniform_real_distribution<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  for (int i = 0; i < n; ++i) {
    r[i] = random_distribution(*caffe_rng());
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  std::normal_distribution<Dtype> random_distribution(a, sigma);
  for (int i = 0; i < n; ++i) {
    r[i] = random_distribution(*caffe_rng());
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  std::bernoulli_distribution random_distribution(p);
  for (int i = 0; i < n; ++i) {
    r[i] = random_distribution(*caffe_rng())?1:0;
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  std::bernoulli_distribution random_distribution(p);
  for (int i = 0; i < n; ++i) {
    r[i] = random_distribution(*caffe_rng())?1U:0U;
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

}  // namespace caffe
