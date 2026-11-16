#include <complex>

#include "lazuli/matrix/matrix.hpp"

namespace lazuli::detail {

template class matrix<float>;
template class matrix<double>;
template class matrix<std::complex<float>>;
template class matrix<std::complex<double>>;

}  // namespace lazuli::detail
