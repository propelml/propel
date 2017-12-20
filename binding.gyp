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
      'sources': [ 'binding.cc' ],
      'include_dirs': [ '<@(tensorflow_include_dir)' ],
      'conditions': [
        ['OS=="win"', {
          'defines': [ 'COMPILER_MSVC' ],
          'libraries': [ 'tensorflow' ],
          'library_dirs': [ '<(INTERMEDIATE_DIR)' ],
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
