#ifndef PROPEL_CHECK_H_
#define PROPEL_CHECK_H_

#ifdef _MSC_VER  // msvc
#define no_return __declspec(noreturn)
#else  // gcc and clang
#define no_return __attribute__((noreturn))
#endif

#ifdef _MSC_VER
#define no_inline __declspec(noinline)
#else  // gcc and clang
#define no_inline __attribute__((noinline))
#endif

#define __to_string_helper(v) #v
#define __to_string(v) __to_string_helper(v)
#define __expand(v) v

#define __check_helper(expression, message, ...)                 \
  (void) ((!!(expression)) ||                                    \
          (abort_with_message("" message "\n"                    \
                              "  test: " #expression "\n"        \
                              "  file: " __FILE__ "\n"           \
                              "  line: " __to_string(__LINE__)), \
           0))
#define check(...) __expand(__check_helper(__VA_ARGS__, "Check failed", _))

#define fatal(message)                         \
  (abort_with_message("" message "\n"          \
                      "  file: " __FILE__ "\n" \
                      "  line: " __to_string(__LINE__)))

static void no_inline no_return abort_with_message(const char* message);

// TODO(piscisaureus) Move to separate compilation unit.
void abort_with_message(const char* message) {
  puts(message);
  abort();
}

#endif  // PROPEL_CHECK_H_
