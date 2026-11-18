#pragma once

/// @file
/// @brief Row-major, contiguous, owning dense matrix.

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>

#include "lazuli/core/aligned.hpp"
#include "lazuli/core/scalar.hpp"
#include "lazuli/matrix/like.hpp"
#include "lazuli/matrix/random.hpp"
#include "lazuli/matrix/view.hpp"

namespace lazuli::detail {

template <scalar T>
class matrix {
  public:
    using value_type = T;
    using element_type = T;

    constexpr matrix() noexcept = default;

    /// Zero-filled `rows` by `cols` matrix.
    matrix(std::size_t rows, std::size_t cols)
        : m_rows(rows),
          m_cols(cols),
          m_buffer(rows * cols) {
        if (!m_buffer.empty()) {
            std::fill_n(m_buffer.data(), m_buffer.size(), T{});
        }
    }

    /// Same shape, elements deliberately left untouched.
    [[nodiscard]] static matrix uninitialized(std::size_t rows, std::size_t cols) {
        matrix result;
        result.m_rows = rows;
        result.m_cols = cols;
        result.m_buffer = aligned_buffer<T>{rows * cols};
        return result;
    }

    /// Uniformly distributed values in `bounds`, reproducible from `seed`.
    [[nodiscard]] static matrix random(std::size_t rows,
                                       std::size_t cols,
                                       std::uint64_t seed,
                                       interval<T> bounds) {
        matrix result = uninitialized(rows, cols);
        splitmix64 generator{seed};
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < cols; ++column) {
                result(row, column) = uniform_between<T>(generator, bounds);
            }
        }
        return result;
    }

    matrix(std::initializer_list<std::initializer_list<T>> rows) {
        m_rows = rows.size();
        m_cols = m_rows == 0 ? 0 : rows.begin()->size();
        for (const auto& values : rows) {
            if (values.size() != m_cols) {
                fail_dimension_mismatch("ragged initialiser list");
            }
        }
        m_buffer = aligned_buffer<T>{m_rows * m_cols};
        std::size_t index = 0;
        for (const auto& values : rows) {
            for (const auto& value : values) {
                m_buffer.data()[index] = value;
                ++index;
            }
        }
    }

    matrix(const matrix& other)
        : m_rows(other.m_rows),
          m_cols(other.m_cols),
          m_buffer(other.m_buffer.size()) {
        std::copy_n(other.m_buffer.data(), m_buffer.size(), m_buffer.data());
    }

    matrix(matrix&& other) noexcept
        : m_rows(other.m_rows),
          m_cols(other.m_cols),
          m_buffer(std::move(other.m_buffer)) {
        other.m_rows = 0;
        other.m_cols = 0;
    }

    matrix& operator=(const matrix& other) {
        matrix copy{other};
        swap(copy);
        return *this;
    }

    matrix& operator=(matrix&& other) noexcept {
        matrix moved{std::move(other)};
        swap(moved);
        return *this;
    }

    ~matrix() = default;

    void swap(matrix& other) noexcept {
        std::swap(m_rows, other.m_rows);
        std::swap(m_cols, other.m_cols);
        m_buffer.swap(other.m_buffer);
    }

    [[nodiscard]] constexpr std::size_t rows() const noexcept { return m_rows; }
    [[nodiscard]] constexpr std::size_t cols() const noexcept { return m_cols; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return m_rows * m_cols; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] constexpr std::size_t row_stride() const noexcept { return m_cols; }
    [[nodiscard]] constexpr std::size_t col_stride() const noexcept { return 1; }

    /// Throws when the indices are out of range.
    [[nodiscard]] decltype(auto) at(this auto&& self, std::size_t row, std::size_t col) {
        if (row >= self.m_rows) {
            fail_out_of_bounds(row, self.m_rows);
        }
        if (col >= self.m_cols) {
            fail_out_of_bounds(col, self.m_cols);
        }
        return self(row, col);
    }

    /// Unchecked access. The caller guarantees the indices are in range.
    [[nodiscard]] decltype(auto) operator()(this auto&& self,
                                            std::size_t row,
                                            std::size_t col) noexcept {
        return self.m_buffer.data()[row * self.m_cols + col];
    }

    [[nodiscard]] decltype(auto) data(this auto&& self) noexcept { return self.m_buffer.data(); }

    [[nodiscard]] auto view(this auto&& self) noexcept {
        using element = std::
            conditional_t<std::is_const_v<std::remove_reference_t<decltype(self)>>, const T, T>;
        return matrix_view<element>{self.data(), self.m_rows, self.m_cols, self.m_cols, 1};
    }

    template <typename Self>
    operator matrix_view<
        std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, const T, T>>(
        this Self&& self) noexcept {
        return self.view();
    }

    [[nodiscard]] auto block(
        this auto&& self, std::size_t row, std::size_t col, std::size_t rows, std::size_t cols) {
        return self.view().block(row, col, rows, cols);
    }

    [[nodiscard]] auto row(this auto&& self, std::size_t index) { return self.view().row(index); }

    [[nodiscard]] auto col(this auto&& self, std::size_t index) { return self.view().col(index); }

    [[nodiscard]] auto transpose(this auto&& self) noexcept { return self.view().transpose(); }

    /// Whether this matrix shares storage with `grid`.
    template <typename E>
        requires(matrix_like<std::decay_t<E>> && !std::is_same_v<std::decay_t<E>, matrix>)
    matrix(E&& expression) {
        assign_expression(std::forward<E>(expression), true);
    }

    /// Evaluates an expression into this matrix, resizing when it is empty.
    template <typename E>
        requires(matrix_like<std::decay_t<E>> && !std::is_same_v<std::decay_t<E>, matrix>)
    matrix& operator=(E&& expression) {
        assign_expression(std::forward<E>(expression), true);
        return *this;
    }

    /// Assignment that skips the overlap check. Only use it when the destination cannot alias
    /// any operand.
    struct noalias_proxy {
        matrix& target;

        template <typename E>
            requires matrix_like<std::decay_t<E>>
        matrix& operator=(E&& expression) {
            target.assign_expression(std::forward<E>(expression), false);
            return target;
        }
    };

    [[nodiscard]] noalias_proxy noalias() noexcept { return noalias_proxy{*this}; }

    /// Visits this matrix as a leaf of an expression.
    template <typename F>
    void visit_leaves(F&& visit) const {
        visit(view());
    }

    /// Writes `value` to every element.
    void fill(const T& value) { view().fill(value); }

    /// Fills every element from `generator(row, column)`.
    template <typename Generator>
        requires std::invocable<Generator&, std::size_t, std::size_t>
    void fill(Generator&& generator) {
        view().fill(std::forward<Generator>(generator));
    }

  private:
    template <typename E>
    void assign_expression(E&& expression, bool check_alias) {
        auto node = to_node(std::forward<E>(expression));
        using node_type = decltype(node);
        static_assert(std::is_same_v<typename node_type::value_type, T>);

        if (m_rows == 0 && m_cols == 0) {
            m_rows = node.rows();
            m_cols = node.cols();
            m_buffer = aligned_buffer<T>{m_rows * m_cols};
        } else if (m_rows != node.rows() || m_cols != node.cols()) {
            fail_dimension_mismatch("assignment shape disagrees");
        }

        if (check_alias && has_alias(node)) {
            matrix temporary(node.rows(), node.cols());
            assign_into(node, temporary.view());
            swap(temporary);
            return;
        }
        assign_into(node, view());
    }

    template <typename Node>
    [[nodiscard]] bool has_alias(const Node& node) const {
        bool found = false;
        node.visit_leaves([&](const auto& leaf) {
            if (!found && overlaps(view(), leaf)) {
                found = true;
            }
        });
        return found;
    }

    std::size_t m_rows = 0;
    std::size_t m_cols = 0;
    aligned_buffer<T> m_buffer;
};

}  // namespace lazuli::detail
