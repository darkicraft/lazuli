#pragma once

/// @file
/// @brief Row-major, contiguous, owning one-dimensional array.

#include <lazuli/core/aligned.hpp>
#include <lazuli/core/scalar.hpp>
#include <lazuli/matrix/like.hpp>
#include <lazuli/matrix/view.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <utility>

namespace lazuli::detail {

template <scalar T>
class vector {
  public:
    using value_type = T;
    using element_type = T;
    using column_tag = void;

    constexpr vector() noexcept = default;

    explicit vector(std::size_t size)
        : m_buffer(size) {
        if (!m_buffer.empty()) {
            std::fill_n(m_buffer.data(), m_buffer.size(), T{});
        }
    }

    /// Uniformly distributed values in `bounds`, reproducible from `seed`.
    [[nodiscard]] static vector random(std::size_t size, std::uint64_t seed, interval<T> bounds) {
        vector result = uninitialized(size);
        splitmix64 generator{seed};
        for (std::size_t index = 0; index < size; ++index) {
            result[index] = uniform_between<T>(generator, bounds);
        }
        return result;
    }

    [[nodiscard]] static vector uninitialized(std::size_t size) {
        vector result;
        result.m_buffer = aligned_buffer<T>{size};
        return result;
    }

    vector(std::initializer_list<T> values)
        : m_buffer(values.size()) {
        std::copy(values.begin(), values.end(), m_buffer.data());
    }

    vector(const vector& other)
        : m_buffer(other.m_buffer.size()) {
        std::copy_n(other.m_buffer.data(), m_buffer.size(), m_buffer.data());
    }

    vector(vector&& other) noexcept
        : m_buffer(std::move(other.m_buffer)) {}

    vector& operator=(const vector& other) {
        vector copy{other};
        swap(copy);
        return *this;
    }

    vector& operator=(vector&& other) noexcept {
        vector moved{std::move(other)};
        swap(moved);
        return *this;
    }

    ~vector() = default;

    void swap(vector& other) noexcept { m_buffer.swap(other.m_buffer); }

    [[nodiscard]] std::size_t size() const noexcept { return m_buffer.size(); }
    [[nodiscard]] bool empty() const noexcept { return m_buffer.size() == 0; }

    [[nodiscard]] decltype(auto) data(this auto&& self) noexcept { return self.m_buffer.data(); }

    [[nodiscard]] decltype(auto) operator[](this auto&& self, std::size_t index) noexcept {
        return self.m_buffer.data()[index];
    }

    /// Throws when the index is out of range.
    [[nodiscard]] decltype(auto) at(this auto&& self, std::size_t index) {
        if (index >= self.size()) {
            fail_out_of_bounds(index, self.size());
        }
        return self[index];
    }

    [[nodiscard]] auto begin(this auto&& self) noexcept { return self.data(); }
    [[nodiscard]] auto end(this auto&& self) noexcept { return self.data() + self.size(); }

    /// View of the vector as a single column, so it can be passed where a
    /// matrix view is expected.
    [[nodiscard]] auto as_column(this auto&& self) noexcept {
        using element = std::
            conditional_t<std::is_const_v<std::remove_reference_t<decltype(self)>>, const T, T>;
        return matrix_view<element>{
            self.data(), self.size(), std::size_t{1}, std::size_t{1}, self.size()};
    }

    /// A vector reads as an n-by-1 grid inside expressions.
    [[nodiscard]] std::size_t rows() const noexcept { return size(); }
    [[nodiscard]] std::size_t cols() const noexcept { return 1; }

    [[nodiscard]] T& operator()(std::size_t row, std::size_t /*column*/) noexcept {
        return m_buffer.data()[row];
    }

    [[nodiscard]] const T& operator()(std::size_t row, std::size_t /*column*/) const noexcept {
        return m_buffer.data()[row];
    }

    /// Evaluates a single-column expression into this vector, resizing when it is empty.
    template <typename E>
        requires(matrix_like<std::decay_t<E>> && !std::is_same_v<std::decay_t<E>, vector>)
    vector(E&& expression) {
        assign_expression(std::forward<E>(expression));
    }

    template <typename E>
        requires(matrix_like<std::decay_t<E>> && !std::is_same_v<std::decay_t<E>, vector>)
    vector& operator=(E&& expression) {
        assign_expression(std::forward<E>(expression));
        return *this;
    }

    /// Visits this vector as a leaf of an expression, seen as a column.
    template <typename F>
    void visit_leaves(F&& visit) const {
        visit(as_column());
    }

    /// Writes `value` to every element.
    void fill(const T& value) {
        for (std::size_t index = 0; index < size(); ++index) {
            m_buffer.data()[index] = value;
        }
    }

    /// Fills every element from `generator(index)`.
    template <typename Generator>
        requires std::invocable<Generator&, std::size_t>
    void fill(Generator&& generator) {
        for (std::size_t index = 0; index < size(); ++index) {
            m_buffer.data()[index] = generator(index);
        }
    }

  private:
    template <typename E>
    void assign_expression(E&& expression) {
        auto node = to_node(std::forward<E>(expression));
        using node_type = decltype(node);
        static_assert(std::is_same_v<typename node_type::value_type, T>);
        if (node.cols() != 1) {
            fail_dimension_mismatch("a vector needs a single-column result");
        }

        if (empty()) {
            m_buffer = aligned_buffer<T>{node.rows()};
        } else if (size() != node.rows()) {
            fail_dimension_mismatch("assignment shape disagrees");
        }

        bool aliased = false;
        node.visit_leaves([&](const auto& leaf) {
            if (!aliased && overlaps(as_column(), leaf)) {
                aliased = true;
            }
        });
        if (aliased) {
            vector temporary(node.rows());
            assign_into(node, temporary.as_column());
            swap(temporary);
            return;
        }
        assign_into(node, as_column());
    }

    aligned_buffer<T> m_buffer;
};

}  // namespace lazuli::detail
