#pragma once

/// @file
/// @brief Lazy expression nodes, and the operators that build them.
///
/// Elementwise nodes fuse: assigning a chain walks the destination once and inlines the whole
/// tree. A matrix product cannot fuse, because the kernel writes a whole tile rather than one
/// element, so a product either writes straight into the destination (when it is the root of the
/// expression) or materialises itself once into a cache the first time an element is read.

#include <lazuli/core/options.hpp>
#include <lazuli/core/scalar.hpp>
#include <lazuli/kernel/matmul.hpp>
#include <lazuli/matrix/like.hpp>
#include <lazuli/matrix/matrix.hpp>

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

namespace lazuli::detail {

/// Evaluates a node into a fresh matrix.
template <typename Node>
[[nodiscard]] matrix<typename Node::value_type> evaluate(const Node& node, options settings = {}) {
    matrix<typename Node::value_type> result(node.rows(), node.cols());
    node.assign_to(result.view(), settings);
    return result;
}

template <typename L, typename R>
void require_same_shape(const L& left, const R& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        fail_dimension_mismatch("elementwise operand shapes disagree");
    }
}

/// Binary elementwise nodes share everything but the operation.
template <typename L, typename R>
class add {
  public:
    using value_type = typename L::value_type;
    using node_tag = void;
    static_assert(std::is_same_v<value_type, typename R::value_type>);

    add(L left, R right)
        : m_left(std::move(left)),
          m_right(std::move(right)) {
        require_same_shape(m_left, m_right);
    }

    [[nodiscard]] std::size_t rows() const { return m_left.rows(); }
    [[nodiscard]] std::size_t cols() const { return m_left.cols(); }

    [[nodiscard]] value_type at(std::size_t row, std::size_t column) const {
        return static_cast<value_type>(m_left(row, column)) +
               static_cast<value_type>(m_right(row, column));
    }

    [[nodiscard]] value_type operator()(std::size_t row, std::size_t column) const {
        return at(row, column);
    }

    void assign_to(matrix_view<value_type> out, options settings = {}) const {
        configure(settings);
        for (std::size_t row = 0; row < out.rows(); ++row) {
            for (std::size_t column = 0; column < out.cols(); ++column) {
                out(row, column) = at(row, column);
            }
        }
    }

    template <typename F>
    void visit_leaves(F&& visit) const {
        m_left.visit_leaves(visit);
        m_right.visit_leaves(visit);
    }

    [[nodiscard]] auto eval(options settings = {}) const { return evaluate(*this, settings); }

    void configure(options settings) const {
        m_left.configure(settings);
        m_right.configure(settings);
    }

  private:
    L m_left;
    R m_right;
};

template <typename L, typename R>
class sub {
  public:
    using value_type = typename L::value_type;
    using node_tag = void;
    static_assert(std::is_same_v<value_type, typename R::value_type>);

    sub(L left, R right)
        : m_left(std::move(left)),
          m_right(std::move(right)) {
        require_same_shape(m_left, m_right);
    }

    [[nodiscard]] std::size_t rows() const { return m_left.rows(); }
    [[nodiscard]] std::size_t cols() const { return m_left.cols(); }

    [[nodiscard]] value_type at(std::size_t row, std::size_t column) const {
        return static_cast<value_type>(m_left(row, column)) -
               static_cast<value_type>(m_right(row, column));
    }

    [[nodiscard]] value_type operator()(std::size_t row, std::size_t column) const {
        return at(row, column);
    }

    void assign_to(matrix_view<value_type> out, options settings = {}) const {
        configure(settings);
        for (std::size_t row = 0; row < out.rows(); ++row) {
            for (std::size_t column = 0; column < out.cols(); ++column) {
                out(row, column) = at(row, column);
            }
        }
    }

    template <typename F>
    void visit_leaves(F&& visit) const {
        m_left.visit_leaves(visit);
        m_right.visit_leaves(visit);
    }

    [[nodiscard]] auto eval(options settings = {}) const { return evaluate(*this, settings); }

    void configure(options settings) const {
        m_left.configure(settings);
        m_right.configure(settings);
    }

  private:
    L m_left;
    R m_right;
};

template <typename L>
class negate {
  public:
    using value_type = typename L::value_type;
    using node_tag = void;

    explicit negate(L operand)
        : m_operand(std::move(operand)) {}

    [[nodiscard]] std::size_t rows() const { return m_operand.rows(); }
    [[nodiscard]] std::size_t cols() const { return m_operand.cols(); }

    [[nodiscard]] value_type at(std::size_t row, std::size_t column) const {
        return -static_cast<value_type>(m_operand(row, column));
    }

    [[nodiscard]] value_type operator()(std::size_t row, std::size_t column) const {
        return at(row, column);
    }

    void assign_to(matrix_view<value_type> out, options settings = {}) const {
        configure(settings);
        for (std::size_t row = 0; row < out.rows(); ++row) {
            for (std::size_t column = 0; column < out.cols(); ++column) {
                out(row, column) = at(row, column);
            }
        }
    }

    template <typename F>
    void visit_leaves(F&& visit) const {
        m_operand.visit_leaves(visit);
    }

    [[nodiscard]] auto eval(options settings = {}) const { return evaluate(*this, settings); }

    void configure(options settings) const { m_operand.configure(settings); }

  private:
    L m_operand;
};

template <typename L>
class scale {
  public:
    using value_type = typename L::value_type;
    using node_tag = void;

    scale(value_type factor, L operand)
        : m_factor(factor),
          m_operand(std::move(operand)) {}

    [[nodiscard]] std::size_t rows() const { return m_operand.rows(); }
    [[nodiscard]] std::size_t cols() const { return m_operand.cols(); }

    [[nodiscard]] value_type at(std::size_t row, std::size_t column) const {
        return m_factor * static_cast<value_type>(m_operand(row, column));
    }

    [[nodiscard]] value_type operator()(std::size_t row, std::size_t column) const {
        return at(row, column);
    }

    void assign_to(matrix_view<value_type> out, options settings = {}) const {
        configure(settings);
        for (std::size_t row = 0; row < out.rows(); ++row) {
            for (std::size_t column = 0; column < out.cols(); ++column) {
                out(row, column) = at(row, column);
            }
        }
    }

    template <typename F>
    void visit_leaves(F&& visit) const {
        m_operand.visit_leaves(visit);
    }

    [[nodiscard]] auto eval(options settings = {}) const { return evaluate(*this, settings); }

    void configure(options settings) const { m_operand.configure(settings); }

  private:
    value_type m_factor;
    L m_operand;
};

template <typename L, typename R>
class product {
  public:
    using value_type = typename L::value_type;
    using node_tag = void;
    static_assert(std::is_same_v<value_type, typename R::value_type>);

    product(L left, R right)
        : m_left(std::move(left)),
          m_right(std::move(right)) {
        if (m_left.cols() != m_right.rows()) {
            fail_dimension_mismatch("matrix product operand shapes disagree");
        }
    }

    [[nodiscard]] std::size_t rows() const { return m_left.rows(); }
    [[nodiscard]] std::size_t cols() const { return m_right.cols(); }

    [[nodiscard]] value_type at(std::size_t row, std::size_t column) const {
        materialise();
        return (*m_cache)(row, column);
    }

    [[nodiscard]] value_type operator()(std::size_t row, std::size_t column) const {
        return at(row, column);
    }

    void assign_to(matrix_view<value_type> out, options settings = {}) const {
        configure(settings);
        kernel::matmul(left_view(), right_view(), out, settings);
    }

    void configure(options settings) const {
        m_settings = settings;
        m_left.configure(settings);
        m_right.configure(settings);
    }

    template <typename F>
    void visit_leaves(F&& visit) const {
        m_left.visit_leaves(visit);
        m_right.visit_leaves(visit);
    }

    [[nodiscard]] auto eval(options settings = {}) const { return evaluate(*this, settings); }

  private:
    void materialise() const {
        if (!m_cache) {
            m_cache.emplace(evaluate(*this, m_settings));
        }
    }

    /// The kernel wants views, so an operand that is itself an expression is materialised once.
    template <typename Side>
    [[nodiscard]] matrix_view<const value_type> materialise_operand(
        const Side& side, std::optional<matrix<value_type>>& cache) const {
        if constexpr (requires { typename Side::view_tag; }) {
            return side;
        } else {
            if (!cache) {
                cache.emplace(evaluate(side));
            }
            return cache->view();
        }
    }

    [[nodiscard]] matrix_view<const value_type> left_view() const {
        return materialise_operand(m_left, m_left_cache);
    }

    [[nodiscard]] matrix_view<const value_type> right_view() const {
        return materialise_operand(m_right, m_right_cache);
    }

    L m_left;
    R m_right;
    mutable std::optional<matrix<value_type>> m_cache;
    mutable std::optional<matrix<value_type>> m_left_cache;
    mutable std::optional<matrix<value_type>> m_right_cache;
    mutable options m_settings{};
};

template <typename L, typename R>
    requires(matrix_like<std::decay_t<L>> && matrix_like<std::decay_t<R>>)
[[nodiscard]] auto operator+(L&& left, R&& right) {
    return add{to_node(std::forward<L>(left)), to_node(std::forward<R>(right))};
}

template <typename L, typename R>
    requires(matrix_like<std::decay_t<L>> && matrix_like<std::decay_t<R>>)
[[nodiscard]] auto operator-(L&& left, R&& right) {
    return sub{to_node(std::forward<L>(left)), to_node(std::forward<R>(right))};
}

template <typename L, typename R>
    requires(matrix_like<std::decay_t<L>> && matrix_like<std::decay_t<R>>)
[[nodiscard]] auto operator*(L&& left, R&& right) {
    return product{to_node(std::forward<L>(left)), to_node(std::forward<R>(right))};
}

template <typename L>
    requires matrix_like<std::decay_t<L>>
[[nodiscard]] auto operator-(L&& operand) {
    return negate{to_node(std::forward<L>(operand))};
}

template <typename S, typename R>
    requires(std::is_arithmetic_v<std::decay_t<S>> && matrix_like<std::decay_t<R>>)
[[nodiscard]] auto operator*(S factor, R&& operand) {
    using element = typename std::decay_t<R>::value_type;
    return scale{static_cast<element>(factor), to_node(std::forward<R>(operand))};
}

template <typename L, typename S>
    requires(matrix_like<std::decay_t<L>> && std::is_arithmetic_v<std::decay_t<S>>)
[[nodiscard]] auto operator*(L&& operand, S factor) {
    using element = typename std::decay_t<L>::value_type;
    return scale{static_cast<element>(factor), to_node(std::forward<L>(operand))};
}

template <typename L, typename S>
    requires(matrix_like<std::decay_t<L>> && std::is_arithmetic_v<std::decay_t<S>>)
[[nodiscard]] auto operator/(L&& operand, S divisor) {
    using element = typename std::decay_t<L>::value_type;
    return scale{element{1} / static_cast<element>(divisor), to_node(std::forward<L>(operand))};
}

}  // namespace lazuli::detail
