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

One unincluded dependeny is a chromium binary for running tests.
If puppeteer complains that chromium hasn't been downloaded, run:

```bash
npm rebuild puppeteer
```

If you're on Linux and would like to build a CUDA version of
Propel set the environmental variable `PROPEL_BUILD_GPU=1`.

## Packages

npm packages are built with `./tools/package.js`. Here are links
to the various packages:

    https://www.npmjs.com/package/propel
    https://www.npmjs.com/package/propel_linux
    https://www.npmjs.com/package/propel_linux_gpu
    https://www.npmjs.com/package/propel_mac
    https://www.npmjs.com/package/propel_windows
