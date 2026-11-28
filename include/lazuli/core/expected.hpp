#pragma once

/// @file
/// @brief The result type used by the no-throw facade.
///
/// The library does not depend on `<expected>`: the type is always this one, so a library built
/// as C++20 has the same return types as a consumer compiled with any later standard. Phase one
/// covers construction, inspection and value access; the monadic operations and comparisons are
/// additive.

#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace lazuli::detail {

/// The error half of an `expected`, used to construct an error result unambiguously.
template <typename E>
class unexpected {
  public:
    constexpr explicit unexpected(const E& error)
        : m_error(error) {}
    constexpr explicit unexpected(E&& error)
        : m_error(std::move(error)) {}

    [[nodiscard]] constexpr const E& error() const& noexcept { return m_error; }
    [[nodiscard]] constexpr E& error() & noexcept { return m_error; }
    [[nodiscard]] constexpr E&& error() && noexcept { return std::move(m_error); }

    void swap(unexpected& other) noexcept(std::is_nothrow_swappable_v<E>) {
        using std::swap;
        swap(m_error, other.m_error);
    }

  private:
    E m_error;
};

template <typename E>
unexpected(E) -> unexpected<E>;

template <typename E>
void swap(unexpected<E>& left, unexpected<E>& right) noexcept(noexcept(left.swap(right))) {
    left.swap(right);
}

template <typename E>
[[nodiscard]] bool operator==(const unexpected<E>& left, const unexpected<E>& right) {
    return left.error() == right.error();
}

/// Tag for constructing an error result in place.
struct unexpect_t {
    explicit unexpect_t() = default;
};

inline constexpr unexpect_t unexpect{};

/// Thrown by `expected::value()` when the object holds an error.
template <typename E>
class bad_expected_access : public std::exception {
  public:
    explicit bad_expected_access(E error)
        : m_error(std::move(error)) {}

    [[nodiscard]] const char* what() const noexcept override { return "bad expected access"; }

    [[nodiscard]] E& error() & noexcept { return m_error; }
    [[nodiscard]] const E& error() const& noexcept { return m_error; }
    [[nodiscard]] E&& error() && noexcept { return std::move(m_error); }

  private:
    E m_error;
};

/// A value or an error. The value state is the common path.
template <typename T, typename E>
class expected {
  public:
    using value_type = T;
    using error_type = E;
    using unexpected_type = unexpected<E>;

    expected()
        requires std::is_default_constructible_v<T>
        : m_has_value(true) {
        std::construct_at(&m_storage.value);
    }

    expected(const T& value)
        : m_has_value(true) {
        std::construct_at(&m_storage.value, value);
    }

    expected(T&& value)
        : m_has_value(true) {
        std::construct_at(&m_storage.value, std::move(value));
    }

    expected(const unexpected<E>& error)
        : m_has_value(false) {
        std::construct_at(&m_storage.error, error.error());
    }

    expected(unexpected<E>&& error)
        : m_has_value(false) {
        std::construct_at(&m_storage.error, std::move(error).error());
    }

    template <typename... Args>
        requires std::is_constructible_v<T, Args...>
    explicit expected(std::in_place_t, Args&&... args)
        : m_has_value(true) {
        std::construct_at(&m_storage.value, std::forward<Args>(args)...);
    }

    template <typename... Args>
        requires std::is_constructible_v<E, Args...>
    explicit expected(unexpect_t, Args&&... args)
        : m_has_value(false) {
        std::construct_at(&m_storage.error, std::forward<Args>(args)...);
    }

    expected(const expected& other)
        : m_has_value(other.m_has_value) {
        if (m_has_value) {
            std::construct_at(&m_storage.value, other.m_storage.value);
        } else {
            std::construct_at(&m_storage.error, other.m_storage.error);
        }
    }

    expected(expected&& other) noexcept(std::is_nothrow_move_constructible_v<T> &&
                                        std::is_nothrow_move_constructible_v<E>)
        : m_has_value(other.m_has_value) {
        if (m_has_value) {
            std::construct_at(&m_storage.value, std::move(other.m_storage.value));
        } else {
            std::construct_at(&m_storage.error, std::move(other.m_storage.error));
        }
    }

    expected& operator=(const expected& other) {
        if (this == &other) {
            return *this;
        }
        if (m_has_value == other.m_has_value) {
            if (m_has_value) {
                m_storage.value = other.m_storage.value;
            } else {
                m_storage.error = other.m_storage.error;
            }
            return *this;
        }
        if (other.m_has_value) {
            T value{other.m_storage.value};
            destroy_active();
            std::construct_at(&m_storage.value, std::move(value));
            m_has_value = true;
        } else {
            E error{other.m_storage.error};
            destroy_active();
            std::construct_at(&m_storage.error, std::move(error));
            m_has_value = false;
        }
        return *this;
    }

    expected& operator=(expected&& other) {
        if (this == &other) {
            return *this;
        }
        if (m_has_value == other.m_has_value) {
            if (m_has_value) {
                m_storage.value = std::move(other.m_storage.value);
            } else {
                m_storage.error = std::move(other.m_storage.error);
            }
            return *this;
        }
        if (other.m_has_value) {
            T value{std::move(other.m_storage.value)};
            destroy_active();
            std::construct_at(&m_storage.value, std::move(value));
            m_has_value = true;
        } else {
            E error{std::move(other.m_storage.error)};
            destroy_active();
            std::construct_at(&m_storage.error, std::move(error));
            m_has_value = false;
        }
        return *this;
    }

    expected& operator=(const T& value) {
        *this = expected{value};
        return *this;
    }

    expected& operator=(T&& value) {
        *this = expected{std::move(value)};
        return *this;
    }

    expected& operator=(const unexpected<E>& error) {
        *this = expected{error};
        return *this;
    }

    expected& operator=(unexpected<E>&& error) {
        *this = expected{std::move(error)};
        return *this;
    }

    ~expected() { destroy_active(); }

    [[nodiscard]] constexpr bool has_value() const noexcept { return m_has_value; }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return m_has_value; }

    [[nodiscard]] T& value() & {
        if (!m_has_value) {
            bad_access();
        }
        return m_storage.value;
    }

    [[nodiscard]] const T& value() const& {
        if (!m_has_value) {
            bad_access();
        }
        return m_storage.value;
    }

    [[nodiscard]] T&& value() && {
        if (!m_has_value) {
            bad_access();
        }
        return std::move(m_storage.value);
    }

    [[nodiscard]] const T&& value() const&& {
        if (!m_has_value) {
            bad_access();
        }
        return std::move(m_storage.value);
    }

    [[nodiscard]] T& operator*() & noexcept { return m_storage.value; }
    [[nodiscard]] const T& operator*() const& noexcept { return m_storage.value; }
    [[nodiscard]] T&& operator*() && noexcept { return std::move(m_storage.value); }
    [[nodiscard]] const T&& operator*() const&& noexcept { return std::move(m_storage.value); }

    [[nodiscard]] T* operator->() noexcept { return std::addressof(m_storage.value); }
    [[nodiscard]] const T* operator->() const noexcept { return std::addressof(m_storage.value); }

    [[nodiscard]] E& error() & noexcept { return m_storage.error; }
    [[nodiscard]] const E& error() const& noexcept { return m_storage.error; }
    [[nodiscard]] E&& error() && noexcept { return std::move(m_storage.error); }

    template <typename U>
    [[nodiscard]] T value_or(U&& fallback) const& {
        return m_has_value ? m_storage.value : static_cast<T>(std::forward<U>(fallback));
    }

    template <typename U>
    [[nodiscard]] T value_or(U&& fallback) && {
        return m_has_value ? std::move(m_storage.value) : static_cast<T>(std::forward<U>(fallback));
    }

    template <typename... Args>
        requires std::is_constructible_v<T, Args...>
    T& emplace(Args&&... args) {
        T value(std::forward<Args>(args)...);
        destroy_active();
        std::construct_at(&m_storage.value, std::move(value));
        m_has_value = true;
        return m_storage.value;
    }

    void swap(expected& other) {
        if (m_has_value == other.m_has_value) {
            using std::swap;
            if (m_has_value) {
                swap(m_storage.value, other.m_storage.value);
            } else {
                swap(m_storage.error, other.m_storage.error);
            }
            return;
        }
        expected temporary{std::move(other)};
        other = std::move(*this);
        *this = std::move(temporary);
    }

    template <typename F>
    auto and_then(F&& function) & {
        using result_type = std::invoke_result_t<F, T&>;
        if (!m_has_value) {
            return result_type{unexpected<E>{m_storage.error}};
        }
        return std::invoke(std::forward<F>(function), m_storage.value);
    }

    template <typename F>
    auto and_then(F&& function) const& {
        using result_type = std::invoke_result_t<F, const T&>;
        if (!m_has_value) {
            return result_type{unexpected<E>{m_storage.error}};
        }
        return std::invoke(std::forward<F>(function), m_storage.value);
    }

    template <typename F>
    auto and_then(F&& function) && {
        using result_type = std::invoke_result_t<F, T&&>;
        if (!m_has_value) {
            return result_type{unexpected<E>{std::move(m_storage.error)}};
        }
        return std::invoke(std::forward<F>(function), std::move(m_storage.value));
    }

    template <typename F>
    auto transform(F&& function) & {
        using result_value = std::invoke_result_t<F, T&>;
        if (!m_has_value) {
            return expected<result_value, E>{unexpected<E>{m_storage.error}};
        }
        if constexpr (std::is_void_v<result_value>) {
            std::invoke(std::forward<F>(function), m_storage.value);
            return expected<result_value, E>{};
        } else {
            return expected<result_value, E>{
                std::invoke(std::forward<F>(function), m_storage.value)};
        }
    }

    template <typename F>
    auto transform(F&& function) const& {
        using result_value = std::invoke_result_t<F, const T&>;
        if (!m_has_value) {
            return expected<result_value, E>{unexpected<E>{m_storage.error}};
        }
        if constexpr (std::is_void_v<result_value>) {
            std::invoke(std::forward<F>(function), m_storage.value);
            return expected<result_value, E>{};
        } else {
            return expected<result_value, E>{
                std::invoke(std::forward<F>(function), m_storage.value)};
        }
    }

    template <typename F>
    auto transform(F&& function) && {
        using result_value = std::invoke_result_t<F, T&&>;
        if (!m_has_value) {
            return expected<result_value, E>{unexpected<E>{std::move(m_storage.error)}};
        }
        if constexpr (std::is_void_v<result_value>) {
            std::invoke(std::forward<F>(function), std::move(m_storage.value));
            return expected<result_value, E>{};
        } else {
            return expected<result_value, E>{
                std::invoke(std::forward<F>(function), std::move(m_storage.value))};
        }
    }

    template <typename F>
    expected or_else(F&& function) & {
        if (m_has_value) {
            return *this;
        }
        return std::invoke(std::forward<F>(function), m_storage.error);
    }

    template <typename F>
    expected or_else(F&& function) const& {
        if (m_has_value) {
            return *this;
        }
        return std::invoke(std::forward<F>(function), m_storage.error);
    }

    template <typename F>
    expected or_else(F&& function) && {
        if (m_has_value) {
            return std::move(*this);
        }
        return std::invoke(std::forward<F>(function), std::move(m_storage.error));
    }

    template <typename F>
    auto transform_error(F&& function) & {
        using new_error = std::invoke_result_t<F, E&>;
        if (m_has_value) {
            return expected<T, new_error>{m_storage.value};
        }
        return expected<T, new_error>{
            unexpected<new_error>{std::invoke(std::forward<F>(function), m_storage.error)}};
    }

    template <typename F>
    auto transform_error(F&& function) const& {
        using new_error = std::invoke_result_t<F, const E&>;
        if (m_has_value) {
            return expected<T, new_error>{m_storage.value};
        }
        return expected<T, new_error>{
            unexpected<new_error>{std::invoke(std::forward<F>(function), m_storage.error)}};
    }

    template <typename F>
    auto transform_error(F&& function) && {
        using new_error = std::invoke_result_t<F, E&&>;
        if (m_has_value) {
            return expected<T, new_error>{std::move(m_storage.value)};
        }
        return expected<T, new_error>{unexpected<new_error>{
            std::invoke(std::forward<F>(function), std::move(m_storage.error))}};
    }

  private:
    union storage_type {
        storage_type() {}
        ~storage_type() {}

        T value;
        E error;
    };

    void destroy_active() noexcept {
        if (m_has_value) {
            std::destroy_at(&m_storage.value);
        } else {
            std::destroy_at(&m_storage.error);
        }
    }

    [[noreturn]] void bad_access() const {
#if defined(__cpp_exceptions)
        throw bad_expected_access<E>{m_storage.error};
#else
        std::abort();
#endif
    }

    storage_type m_storage;
    bool m_has_value;
};

template <typename T, typename E>
    requires(!std::is_void_v<T>)
void swap(expected<T, E>& left, expected<T, E>& right) noexcept(noexcept(left.swap(right))) {
    left.swap(right);
}

template <typename T, typename E>
    requires(!std::is_void_v<T>)
[[nodiscard]] bool operator==(const expected<T, E>& left, const expected<T, E>& right) {
    if (left.has_value() != right.has_value()) {
        return false;
    }
    return left.has_value() ? *left == *right : left.error() == right.error();
}

template <typename T, typename E>
    requires(!std::is_void_v<T>)
[[nodiscard]] bool operator==(const expected<T, E>& left, const unexpected<E>& right) {
    return !left.has_value() && left.error() == right.error();
}

template <typename T, typename E>
    requires(!std::is_void_v<T>)
[[nodiscard]] bool operator==(const unexpected<E>& left, const expected<T, E>& right) {
    return right == left;
}

/// The `void` specialisation: success carries no value, failure still carries the error.
template <typename E>
class expected<void, E> {
  public:
    using value_type = void;
    using error_type = E;
    using unexpected_type = unexpected<E>;

    expected() noexcept = default;

    explicit expected(std::in_place_t) noexcept
        : m_has_value(true) {}

    expected(const unexpected<E>& error)
        : m_has_value(false) {
        std::construct_at(&m_storage.error, error.error());
    }

    expected(unexpected<E>&& error)
        : m_has_value(false) {
        std::construct_at(&m_storage.error, std::move(error).error());
    }

    template <typename... Args>
        requires std::is_constructible_v<E, Args...>
    explicit expected(unexpect_t, Args&&... args)
        : m_has_value(false) {
        std::construct_at(&m_storage.error, std::forward<Args>(args)...);
    }

    expected(const expected& other)
        : m_has_value(other.m_has_value) {
        if (!m_has_value) {
            std::construct_at(&m_storage.error, other.m_storage.error);
        }
    }

    expected(expected&& other) noexcept(std::is_nothrow_move_constructible_v<E>)
        : m_has_value(other.m_has_value) {
        if (!m_has_value) {
            std::construct_at(&m_storage.error, std::move(other.m_storage.error));
        }
    }

    expected& operator=(const expected& other) {
        if (this == &other) {
            return *this;
        }
        if (!m_has_value && !other.m_has_value) {
            m_storage.error = other.m_storage.error;
            return *this;
        }
        if (m_has_value && other.m_has_value) {
            return *this;
        }
        if (other.m_has_value) {
            std::destroy_at(&m_storage.error);
            m_has_value = true;
        } else {
            E error{other.m_storage.error};
            if (!m_has_value) {
                std::destroy_at(&m_storage.error);
            }
            std::construct_at(&m_storage.error, std::move(error));
            m_has_value = false;
        }
        return *this;
    }

    expected& operator=(expected&& other) {
        if (this == &other) {
            return *this;
        }
        if (!m_has_value && !other.m_has_value) {
            m_storage.error = std::move(other.m_storage.error);
            return *this;
        }
        if (m_has_value && other.m_has_value) {
            return *this;
        }
        if (other.m_has_value) {
            std::destroy_at(&m_storage.error);
            m_has_value = true;
        } else {
            E error{std::move(other.m_storage.error)};
            if (!m_has_value) {
                std::destroy_at(&m_storage.error);
            }
            std::construct_at(&m_storage.error, std::move(error));
            m_has_value = false;
        }
        return *this;
    }

    expected& operator=(const unexpected<E>& error) {
        *this = expected{error};
        return *this;
    }

    expected& operator=(unexpected<E>&& error) {
        *this = expected{std::move(error)};
        return *this;
    }

    ~expected() {
        if (!m_has_value) {
            std::destroy_at(&m_storage.error);
        }
    }

    [[nodiscard]] constexpr bool has_value() const noexcept { return m_has_value; }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return m_has_value; }

    void value() const {
        if (!m_has_value) {
            bad_access();
        }
    }

    void operator*() const noexcept {}

    [[nodiscard]] E& error() & noexcept { return m_storage.error; }
    [[nodiscard]] const E& error() const& noexcept { return m_storage.error; }
    [[nodiscard]] E&& error() && noexcept { return std::move(m_storage.error); }

    void emplace() noexcept {
        if (!m_has_value) {
            std::destroy_at(&m_storage.error);
            m_has_value = true;
        }
    }

    void swap(expected& other) {
        if (m_has_value == other.m_has_value) {
            if (!m_has_value) {
                using std::swap;
                swap(m_storage.error, other.m_storage.error);
            }
            return;
        }
        expected temporary{std::move(other)};
        other = std::move(*this);
        *this = std::move(temporary);
    }

    template <typename F>
    auto and_then(F&& function) & {
        using result_type = std::invoke_result_t<F>;
        if (!m_has_value) {
            return result_type{unexpected<E>{m_storage.error}};
        }
        return std::invoke(std::forward<F>(function));
    }

    template <typename F>
    auto and_then(F&& function) const& {
        using result_type = std::invoke_result_t<F>;
        if (!m_has_value) {
            return result_type{unexpected<E>{m_storage.error}};
        }
        return std::invoke(std::forward<F>(function));
    }

    template <typename F>
    auto and_then(F&& function) && {
        using result_type = std::invoke_result_t<F>;
        if (!m_has_value) {
            return result_type{unexpected<E>{std::move(m_storage.error)}};
        }
        return std::invoke(std::forward<F>(function));
    }

    template <typename F>
    auto transform(F&& function) & {
        using result_value = std::invoke_result_t<F>;
        if (!m_has_value) {
            return expected<result_value, E>{unexpected<E>{m_storage.error}};
        }
        if constexpr (std::is_void_v<result_value>) {
            std::invoke(std::forward<F>(function));
            return expected<result_value, E>{};
        } else {
            return expected<result_value, E>{std::invoke(std::forward<F>(function))};
        }
    }

    template <typename F>
    auto transform(F&& function) const& {
        using result_value = std::invoke_result_t<F>;
        if (!m_has_value) {
            return expected<result_value, E>{unexpected<E>{m_storage.error}};
        }
        if constexpr (std::is_void_v<result_value>) {
            std::invoke(std::forward<F>(function));
            return expected<result_value, E>{};
        } else {
            return expected<result_value, E>{std::invoke(std::forward<F>(function))};
        }
    }

    template <typename F>
    auto transform(F&& function) && {
        using result_value = std::invoke_result_t<F>;
        if (!m_has_value) {
            return expected<result_value, E>{unexpected<E>{std::move(m_storage.error)}};
        }
        if constexpr (std::is_void_v<result_value>) {
            std::invoke(std::forward<F>(function));
            return expected<result_value, E>{};
        } else {
            return expected<result_value, E>{std::invoke(std::forward<F>(function))};
        }
    }

    template <typename F>
    expected or_else(F&& function) & {
        if (m_has_value) {
            return *this;
        }
        return std::invoke(std::forward<F>(function), m_storage.error);
    }

    template <typename F>
    expected or_else(F&& function) const& {
        if (m_has_value) {
            return *this;
        }
        return std::invoke(std::forward<F>(function), m_storage.error);
    }

    template <typename F>
    expected or_else(F&& function) && {
        if (m_has_value) {
            return std::move(*this);
        }
        return std::invoke(std::forward<F>(function), std::move(m_storage.error));
    }

    template <typename F>
    auto transform_error(F&& function) & {
        using new_error = std::invoke_result_t<F, E&>;
        if (m_has_value) {
            return expected<void, new_error>{};
        }
        return expected<void, new_error>{
            unexpected<new_error>{std::invoke(std::forward<F>(function), m_storage.error)}};
    }

    template <typename F>
    auto transform_error(F&& function) const& {
        using new_error = std::invoke_result_t<F, const E&>;
        if (m_has_value) {
            return expected<void, new_error>{};
        }
        return expected<void, new_error>{
            unexpected<new_error>{std::invoke(std::forward<F>(function), m_storage.error)}};
    }

    template <typename F>
    auto transform_error(F&& function) && {
        using new_error = std::invoke_result_t<F, E&&>;
        if (m_has_value) {
            return expected<void, new_error>{};
        }
        return expected<void, new_error>{unexpected<new_error>{
            std::invoke(std::forward<F>(function), std::move(m_storage.error))}};
    }

  private:
    union storage_type {
        storage_type() {}
        ~storage_type() {}

        E error;
    };

    [[noreturn]] void bad_access() const {
#if defined(__cpp_exceptions)
        throw bad_expected_access<E>{m_storage.error};
#else
        std::abort();
#endif
    }

    storage_type m_storage;
    bool m_has_value = true;
};

template <typename E>
void swap(expected<void, E>& left, expected<void, E>& right) noexcept(noexcept(left.swap(right))) {
    left.swap(right);
}

template <typename E>
[[nodiscard]] bool operator==(const expected<void, E>& left, const expected<void, E>& right) {
    if (left.has_value() != right.has_value()) {
        return false;
    }
    return left.has_value() || left.error() == right.error();
}

template <typename E>
[[nodiscard]] bool operator==(const expected<void, E>& left, const unexpected<E>& right) {
    return !left.has_value() && left.error() == right.error();
}

template <typename E>
[[nodiscard]] bool operator==(const unexpected<E>& left, const expected<void, E>& right) {
    return right == left;
}

}  // namespace lazuli::detail
