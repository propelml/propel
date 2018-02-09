/*
   Copyright 2018 Propel http://propel.site/.  All rights reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */
#ifndef SRC_CHECK_H_
#define SRC_CHECK_H_

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

#endif  // SRC_CHECK_H_
