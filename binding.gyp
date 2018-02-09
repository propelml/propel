# Copyright 2018 Propel http://propel.site/.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

{
  'variables': {
    'tensorflow_include_dir': '<(module_root_dir)/deps/libtensorflow/include',
    'tensorflow_headers': [
      '<@(tensorflow_include_dir)/tensorflow/c/c_api.h',
      '<@(tensorflow_include_dir)/tensorflow/c/eager/c_api.h',
    ]
  },
  'targets': [
    {
      'target_name': 'tensorflow-binding',
      'sources': [ 'src/binding.cc' ],
      'include_dirs': [
        '<(tensorflow_include_dir)',
        '<(module_root_dir)',
      ],
      'conditions': [
        ['OS=="win"', {
          'defines': [ 'COMPILER_MSVC' ],
          'libraries': [ 'tensorflow' ],
          'library_dirs': [ '<(INTERMEDIATE_DIR)' ],
          'msvs_disabled_warnings': [
            # Warning	C4190: 'TF_NewWhile' has C-linkage specified, but returns
            # UDT 'TF_WhileParams' which is incompatible with C.
            # (in include/tensorflow/c/c_api.h)
            # This is a tensorflow bug but it doesn't affect propel.
            4190
          ],
          'actions': [
            {
              'action_name': 'generate_def',
              'inputs': [
                '<(module_root_dir)/tools/generate_def.js',
                '<@(tensorflow_headers)'
              ],
              'outputs': [
                '<(INTERMEDIATE_DIR)/tensorflow.def'
              ],
              'action': [
                'cmd',
                '/c node <@(_inputs) > <@(_outputs)'
              ]
            },
            {
              'action_name': 'build-tensorflow-lib',
              'inputs': [
                '<(INTERMEDIATE_DIR)/tensorflow.def'
              ],
              'outputs': [
                '<(INTERMEDIATE_DIR)/tensorflow.lib'
              ],
              'action': [
                'lib',
                '/def:<@(_inputs)',
                '/out:<@(_outputs)',
                '/machine:<@(target_arch)'
              ]
            },
            {
              'action_name': 'extract_dll',
              'inputs': [
                '<(module_root_dir)/tools/extract_dll.js'
              ],
              'outputs': [
                '<(PRODUCT_DIR)/tensorflow.dll'
              ],
              'action': [
                'node',
                '<@(_inputs)',
                '<(PRODUCT_DIR)'
              ]
            }
          ],
        }, { # Linux or Mac
          'actions': [
            {
              'action_name': 'extract_so',
              'inputs': [
                '<(module_root_dir)/tools/extract_so.js'
              ],
              'outputs': [
                '<(PRODUCT_DIR)/libtensorflow.so',
                # unlisted to avoid spurious rebuilds:
                # '<(PRODUCT_DIR)/libtensorflow_framework.so'
              ],
              'action': [
                'node',
                '<@(_inputs)',
                '<(PRODUCT_DIR)'
              ]
            }
          ]
        }],
        ['OS=="linux"', {
          'libraries': [
            '-Wl,-rpath,\$$ORIGIN',
            '-ltensorflow'
          ],
          'library_dirs': [ '<(PRODUCT_DIR)' ],
        }],
        ['OS=="mac"', {
          'libraries': [
            '-Wl,-rpath,@loader_path',
            '-ltensorflow',
          ],
        }],
      ]
    }
  ]
}
