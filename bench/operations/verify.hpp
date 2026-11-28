#pragma once

namespace lazuli::bench {

/// Correctness residuals and timings for the algorithms that have no staged
/// performance path: Householder QR and Jacobi eigenvalues.
void run_verify(int repetitions);

}  // namespace lazuli::bench
