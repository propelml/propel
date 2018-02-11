# Propel

A Machine Learning Framework for JavaScript.

| **Linux & Mac** | **Windows** |
|:---------------:|:-----------:|
| [![][Travis CI badge]][Travis CI link] | [![][AppVeyor badge]][AppVeyor link] |

## How to run TF examples

```
$ ./tools/build_binding.js
$ ts-node src/nn_example_main.ts
```

If you're on Linux and would like to build a CUDA version of
Propel set the environmental variable `PROPEL_BUILD_GPU=1`.

```
$ PROPEL_BUILD_GPU=1 ./tools/build_binding.js
$ ts-node src/nn_example_main.ts
```


## Packages

npm packages are built with `./tools/package.js`. Here are links
to the various packages:

    https://www.npmjs.com/package/propel
    https://www.npmjs.com/package/propel_linux
    https://www.npmjs.com/package/propel_linux_gpu
    https://www.npmjs.com/package/propel_mac
    https://www.npmjs.com/package/propel_windows


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


### Adding An Op

Propel is under heavy development and is missing implementaions for many common
ops. Here is a rough outline of how to add an op

  1. Add the frontend implementation to `api.ts` or `tensor.ts` (depending on
     if its a method of Tensor or a standalone function)

  2. Add the op's signature to the `BackendOps` interface in `types.ts`.

  3. Add the forward and backwards passes to `ops.ts`.

  4. Finally implement the op for DL and TF in `dl.ts` and `tf.ts`
     respectively.
 
  5. Add a test demonstrating the desired behavior in `api_test.ts`.
     The tensorflow binding can be built using `./tools/build_binding.js`
     and the test can be run by doing `ts-node api_test.ts MyTest`.
     The DL test can be run by setting an environmental variable:
     `PROPEL=dl ts-node api_test.ts MyTest`

[AppVeyor badge]:  https://ci.appveyor.com/api/projects/status/github/propelml/propel?branch=master&svg=true
[AppVeyor link]:   https://ci.appveyor.com/project/piscisaureus/propel/branch/master
[Travis CI badge]: https://travis-ci.org/propelml/propel.svg?branch=master
[Travis CI link]:  https://travis-ci.org/propelml/propel/builds
