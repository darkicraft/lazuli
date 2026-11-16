#pragma once

/// @file
/// @brief Non-owning windows over strided, dense two-dimensional blocks.

#include <concepts>
#include <cstddef>
#include <type_traits>

#include "lazuli/core/error.hpp"

namespace lazuli::detail {

/// Window over a dense block of elements, described by a pointer and two
/// strides.
///
/// Element `(row, col)` lives at `data[row * row_stride + col * col_stride]`.
/// Carrying both strides makes transposes, row slices and column slices windows
/// rather than copies, and lets a caller wrap column-major storage without
/// copying it.
///
/// Constness of the view object does not propagate to the elements:
/// `matrix_view<const T>` is the read-only spelling, exactly like `std::span`.
template <typename T>
class matrix_view {
  public:
    using element_type = T;
    using value_type = std::remove_const_t<T>;

    constexpr matrix_view() noexcept = default;

    constexpr matrix_view(T* data,
                          std::size_t rows,
                          std::size_t cols,
                          std::size_t row_stride,
                          std::size_t col_stride) noexcept
        : m_data(data),
          m_rows(rows),
          m_cols(cols),
          m_row_stride(row_stride),
          m_col_stride(col_stride) {}

    /// Widening conversion: a window of mutable elements is a window of const elements.
    template <typename U>
        requires(std::is_convertible_v<U*, T*> && !std::is_same_v<U, T>)
    constexpr matrix_view(const matrix_view<U>& other) noexcept
        : m_data(other.data()),
          m_rows(other.rows()),
          m_cols(other.cols()),
          m_row_stride(other.row_stride()),
          m_col_stride(other.col_stride()) {}

    [[nodiscard]] constexpr std::size_t rows() const noexcept { return m_rows; }
    [[nodiscard]] constexpr std::size_t cols() const noexcept { return m_cols; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return m_rows * m_cols; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] constexpr std::size_t row_stride() const noexcept { return m_row_stride; }
    [[nodiscard]] constexpr std::size_t col_stride() const noexcept { return m_col_stride; }
    [[nodiscard]] constexpr T* data() const noexcept { return m_data; }

    /// Unchecked access. The caller guarantees `row < rows()` and `col <
    /// cols()`.
    [[nodiscard]] constexpr T& operator()(std::size_t row, std::size_t col) const noexcept {
        return m_data[row * m_row_stride + col * m_col_stride];
    }

    /// Checked access.
    [[nodiscard]] T& at(std::size_t row, std::size_t col) const {
        if (row >= m_rows) {
            fail_out_of_bounds(row, m_rows);
        }
        if (col >= m_cols) {
            fail_out_of_bounds(col, m_cols);
        }
        return (*this)(row, col);
    }

    /// Window of `rows` by `cols` elements starting at `row`, `col`.
    [[nodiscard]] matrix_view block(std::size_t row,
                                    std::size_t col,
                                    std::size_t rows,
                                    std::size_t cols) const {
        if (row + rows > m_rows) {
            fail_out_of_bounds(row + rows, m_rows);
        }
        if (col + cols > m_cols) {
            fail_out_of_bounds(col + cols, m_cols);
        }
        return matrix_view{m_data + row * m_row_stride + col * m_col_stride,
                           rows,
                           cols,
                           m_row_stride,
                           m_col_stride};
    }

    [[nodiscard]] matrix_view row(std::size_t index) const {
        if (index >= m_rows) {
            fail_out_of_bounds(index, m_rows);
        }
        return block(index, 0, 1, m_cols);
    }

    [[nodiscard]] matrix_view col(std::size_t index) const {
        if (index >= m_cols) {
            fail_out_of_bounds(index, m_cols);
        }
        return block(0, index, m_rows, 1);
    }

    [[nodiscard]] matrix_view transpose() const noexcept {
        return matrix_view{m_data, m_cols, m_rows, m_col_stride, m_row_stride};
    }

    /// Writes `value` to every element of the window.
    void fill(const T& value)
        requires(!std::is_const_v<T>)
    {
        for (std::size_t row = 0; row < m_rows; ++row) {
            for (std::size_t column = 0; column < m_cols; ++column) {
                (*this)(row, column) = value;
            }
        }
    }

    /// Fills every element of the window from `generator(row, column)`.
    template <typename Generator>
        requires(!std::is_const_v<T> && std::invocable<Generator&, std::size_t, std::size_t>)
    void fill(Generator&& generator) {
        for (std::size_t row = 0; row < m_rows; ++row) {
            for (std::size_t column = 0; column < m_cols; ++column) {
                (*this)(row, column) = generator(row, column);
            }
        }
    }

  private:
    T* m_data = nullptr;
    std::size_t m_rows = 0;
    std::size_t m_cols = 0;
    std::size_t m_row_stride = 0;
    std::size_t m_col_stride = 0;
};

}  // namespace lazuli::detail
