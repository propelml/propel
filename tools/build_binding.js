#!/usr/bin/env node
// build script for binding.cc
// There are literally two compile commands to call. Just call them here.

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const run = require('./run');

const binDir = path.dirname(process.execPath);
const nodeDir = path.dirname(binDir);
const nodeInclude = nodeDir + "/include/node";
const buildDir = run.root + "/build";

if (process.argv.includes("clean")) {
  run.rmrf(buildDir);
  console.log('Deleted', buildDir);
}

run.mkdir(buildDir);
run.mkdir(buildDir + '/Release');

if (process.platform === "darwin" || process.platform === "linux") {
  run.sh(`node tools/extract_so.js ${buildDir}/Release`);
  process.chdir(buildDir);
  // Flags for both linux and mac.
  let cflags = `
    -c
    -o Release/binding.o 
    ../src/binding.cc
    -I${nodeInclude}
    -I${run.root}
    -I${run.root}/deps/libtensorflow/include
    -Wall
    -W
    -Wno-unused-parameter
    -std=gnu++0x
    -DNODE_GYP_MODULE_NAME=tensorflow-binding
    -DUSING_UV_SHARED=1
    -DUSING_V8_SHARED=1
    -DV8_DEPRECATION_WARNINGS=1
    -D_DARWIN_USE_64_BIT_INODE=1
    -D_LARGEFILE_SOURCE
    -D_FILE_OFFSET_BITS=64
    -DBUILDING_NODE_EXTENSION
  `;
  let ldflags = `
    -L./Release
    -o Release/tensorflow-binding.node
    Release/binding.o
    -ltensorflow
    -m64
  `;

  // OS specific flags
  if (process.platform === "darwin")  {
    cflags += `
      -stdlib=libc++
      -mmacosx-version-min=10.7
      -arch x86_64
    `;
    ldflags += `
      -Wl,-rpath,@loader_path
      -stdlib=libc++
      -mmacosx-version-min=10.7
      -arch x86_64
      -undefined dynamic_lookup
    `;
  } else {
    cflags += `-m64 -fPIC -pthread`
    ldflags += `
      -m64 -Wl,-rpath,\$ORIGIN -ltensorflow
      -shared -pthread -rdynamic
    `;
  }
  run.sh(`clang ${cflags}`);
  run.sh(`clang ${ldflags}`);
} else if (process.platform === "win32") {
  execSync("node-gyp rebuild", { cwd: `${__dirname}/..`, stdio: "inherit" });
}
