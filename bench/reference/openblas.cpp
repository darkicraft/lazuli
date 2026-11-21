#include <cblas.h>
#include <reference/openblas.hpp>

namespace lazuli::bench::reference {

void set_threads(int threads) {
    openblas_set_num_threads(threads);
}

int thread_count() noexcept {
    return openblas_get_num_threads();
}

int core_count() noexcept {
    return openblas_get_num_procs();
}

void matmul(std::size_t rows,
            std::size_t inner,
            std::size_t cols,
            const double* a,
            const double* b,
            double* c) {
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                static_cast<blasint>(rows),
                static_cast<blasint>(cols),
                static_cast<blasint>(inner),
                1.0,
                a,
                static_cast<blasint>(inner),
                b,
                static_cast<blasint>(cols),
                0.0,
                c,
                static_cast<blasint>(cols));
}

void matmul(std::size_t rows,
            std::size_t inner,
            std::size_t cols,
            const float* a,
            const float* b,
            float* c) {
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                static_cast<blasint>(rows),
                static_cast<blasint>(cols),
                static_cast<blasint>(inner),
                1.0F,
                a,
                static_cast<blasint>(inner),
                b,
                static_cast<blasint>(cols),
                0.0F,
                c,
                static_cast<blasint>(cols));
}

}  // namespace lazuli::bench::reference
