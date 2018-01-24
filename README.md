# Propel

A Machine Learning Framework for JavaScript.

[![Build Status](https://travis-ci.com/propelml/propel.svg?token=eWz4oGVxypBGsz78gdKp&branch=master)](https://travis-ci.com/propelml/propel)

## Contributing

Check out Propel, including its git submodules.

```bash
git clone --recursive https://github.com/propelml/propel.git
```

Propel generally bundles its dependencies in the "deps" submodule.

To run the tests:

```bash
./tools/presubmit.js
```

One unincluded dependency is a chromium binary for running tests.
If puppeteer complains that chromium hasn't been downloaded, run:

```bash
npm rebuild puppeteer
```

Most of the tests are run on every pull request, however there are three sets
of tests which currently must be run manually:

 1. Check that tests pass on a Linux machine with CUDA.
    Run this command: `PROPEL_BUILD_GPU=1 ./tools/presubmit.js`

 2. Check that the WebGL backend works. (Puppeteer headless doesn't support
    WebGL.)
    Run this command: `PP_TEST_DEBUG=1 ts-node test_browser`

 3. Check that the DeepLearn tests run. (These tests also require WebGL)
    Run this command:  `PP_TEST_DL=1 ts-node test_browser`


## Packages

npm packages are built with `./tools/package.js`. Here are links
to the various packages:

    https://www.npmjs.com/package/propel
    https://www.npmjs.com/package/propel_linux
    https://www.npmjs.com/package/propel_linux_gpu
    https://www.npmjs.com/package/propel_mac
    https://www.npmjs.com/package/propel_windows

If you're on Linux and would like to build a CUDA version of
Propel set the environmental variable `PROPEL_BUILD_GPU=1`.
