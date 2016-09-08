#ifndef CAFFE_UTIL_MKL_ALTERNATE_H_
#define CAFFE_UTIL_MKL_ALTERNATE_H_

#ifdef USE_MKL

#include <mkl.h>

#else  // If use MKL, simply include the MKL header

#ifdef USE_EIGEN

#include <Eigen/Dense>

enum CBLAS_ORDER {
	CblasRowMajor = 101,
	CblasColMajor = 102
};
enum CBLAS_TRANSPOSE {
	CblasNoTrans = 111,
	CblasTrans = 112,
	CblasConjTrans = 113
};

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
#endif  // CAFFE_UTIL_MKL_ALTERNATE_H_
