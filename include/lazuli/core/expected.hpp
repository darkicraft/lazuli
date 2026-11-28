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

  private:
    E m_error;
};

template <typename E>
unexpected(E) -> unexpected<E>;

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

/// The `void` specialisation: success carries no value, failure still carries the error.
template <typename E>
class expected<void, E> {
  public:
    using value_type = void;
    using error_type = E;
    using unexpected_type = unexpected<E>;

    expected() noexcept = default;

    expected(const unexpected<E>& error)
        : m_has_value(false) {
        std::construct_at(&m_storage.error, error.error());
    }

    expected(unexpected<E>&& error)
        : m_has_value(false) {
        std::construct_at(&m_storage.error, std::move(error).error());
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

}  // namespace lazuli::detail
