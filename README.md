# Propel

Differential Programming in JavaScript.

| **Linux & Mac** | **Windows** |
|:---------------:|:-----------:|
| [![][Travis CI badge]][Travis CI link] | [![][AppVeyor badge]][AppVeyor link] |

## Status and Examples

Propel is under heavy development and its API, sadly, is subject to change.
We do intend to make a Promise of Comaptibilty in in the future; but
it can't be done yet.

Please have a look at our example neural network here
https://github.com/propelml/propel/blob/master/example.ts

Also the documentation has many examples:
http://propelml.org/docs


## How to run TF examples

    ./tools/build.js
    ts-node ./example.ts

(Our example program is in TypeScript because it is used in our
tests, however the example can be easily modified to work with normal
JavaScript. `ts-node` is a simple wrapper around `node` for executing
TypeScript files, it can be installed with `npm install -g ts-node`)

If you're on Linux and would like to build a CUDA version of
Propel set the environmental variable `PROPEL_BUILD_GPU=1`.

    PROPEL_BUILD_GPU=1 ./tools/build.js
    ts-node example.ts


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

    git clone --recursive https://github.com/propelml/propel.git

Propel generally bundles its dependencies in the "deps" submodule.

To run the complete test suite:

    ./tools/presubmit.js

One unincluded dependency is a chromium binary for running tests.
If puppeteer complains that chromium hasn't been downloaded, run:

    npm rebuild puppeteer

Most of the tests are run on every pull request, however there are three sets
of tests which currently must be run manually:

 1. Check that tests pass on a Linux machine with CUDA.
    Run this command: `PROPEL_BUILD_GPU=1 ./tools/presubmit.js`

 2. Check that the WebGL backend works. (Puppeteer headless doesn't support
    WebGL.)
    Run this command: `PP_TEST_DEBUG=1 ts-node test_browser`

 3. Check that the DeepLearn tests run. (These tests also require WebGL)
    Run this command:  `PP_TEST_DL=1 ts-node test_browser`


### Working on the website

Propel has two major components:

 1. the core framework, located in `src/`, and

 2. the website, located in `website/`.

Due to the notebook and documentation, the website is fairly non-trivial. It
is written in React (actually Preact), but does not use JSX.

When working on the website, it's often nice to only run the website tests,
and not worry about the framework. To do this run `http-server` on the
`build/website` directory, so you can view it locally:

     node ./node_modules/http-server/bin/http-server build/website

Then to automatically build the website tests on changes, run

    ./tools/parcel.js website/test_website.ts -d build/website/

To run the website tests, point your browser to
http://localhost:8080/static/test.html#script=/test_website.js

However to build the entire website, `./tools/build_website.js` must be run
after every change. (We don't yet have auto-build on change, sorry.)


### Adding An Op

Propel is under heavy development and is missing implementations for many common
ops. Here is a rough outline of how to add an op

  1. Add the frontend implementation to `api.ts` or `tensor.ts` (depending on
     if its a method of Tensor or a standalone function)

  2. Add the op's signature to the `BackendOps` interface in `types.ts`.

  3. Add the forward and backwards passes to `ops.ts`.

  4. Finally implement the op for DL and TF in `dl.ts` and `tf.ts`
     respectively.

  5. Add a test demonstrating the desired behavior in `src/api_test.ts`.
     The tensorflow binding can be built using `./tools/build_binding.js`
     and the test can be run by doing `ts-node src/api_test.ts MyTest`.
     The DL test can be run by setting an environmental variable:
     `PROPEL=dl ts-node src/api_test.ts MyTest`



[AppVeyor badge]:  https://ci.appveyor.com/api/projects/status/github/propelml/propel?branch=master&svg=true
[AppVeyor link]:   https://ci.appveyor.com/project/piscisaureus/propel/branch/master
[Travis CI badge]: https://travis-ci.org/propelml/propel.svg?branch=master
[Travis CI link]:  https://travis-ci.org/propelml/propel/builds
