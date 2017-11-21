Given that there are no usable libtensorflow distributions, the
required files are obtained as follows:

* `libtensorflow.so` and `libtensorflow_framework.so`, which are used
  on MacOS and Linux, are downloaded and extracted from the Tensorflow
  CI server. This is done by `download-so.js`.

* `tensorflow.dll`, on Windows, is downloaded from the CI by
  `download-dll.js`.

* `tensorflow.lib` isn't included in the CI download, so it's generated
  from the header files. This is a two-step process:
    - `tensorflow.def` is generated from the header files by
      `generate-def.js`.
    - `tensorflow.lib` is generated from the .def file by the Windows
      linker; the build rule is located in the project's `binding.gyp`.

* The header files located in `include/` are copied manually from the
  tensorflow source tree. They are checked into the git repository.
  Using the (CI) distribution isn't possible, because the headers for
  the Tensorflow Eager API aren't included.
