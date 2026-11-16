#include <complex>

#include "lazuli/matrix/vector.hpp"

namespace lazuli::detail {

template class vector<float>;
template class vector<double>;
template class vector<std::complex<float>>;
template class vector<std::complex<double>>;

}  // namespace lazuli::detail
