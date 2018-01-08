#!/usr/bin/env node
// build script for binding.cc
// There are literally two compile commands to call. Just call them here.

const fs = require('fs');
const path = require('path');
const run = require('./run');
const { spawnSync: spawn, execSync: exec } = require("child_process");

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
    ../binding.cc
    -I${nodeInclude}
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
  process.chdir(buildDir);
  exec(`node "${__dirname}/extract_dll.js" Release`);
  exec(`node "${__dirname}/generate_def.js" ` +
       `../deps/libtensorflow/include/tensorflow/c/c_api.h ` +
       `../deps/libtensorflow/include/tensorflow/c/eager/c_api.h ` +
       `>tensorflow.def`);
  const cc = exec("where cl.exe clang-cl.exe", { encoding: "utf8" })
    .replace(/\r?\n.*/, "");
  const cmd = `
    "${cc}"
    "..\\binding.cc"
    "tensorflow.def"
    /I "C:\\Users\\BertBelder\\.node-gyp\\8.9.0\\include\\node"
    /I "..\\deps\\libtensorflow\\include"
    /D COMPILER_MSVC
    /D BUILDING_NODE_EXTENSION
    /EHsc
    /Ox
    /link
      "C:\\Users\\BertBelder\\.node-gyp\\8.9.0\\x64\\node.lib"
      /DLL
      /OUT:"Release\\tensorflow-binding.node"
    `.replace(/\s*\n\s*/g, " ");
  console.log(cmd);
  const { status: r } = spawn(cmd, { shell: true, stdio: "inherit" });
  if (r !== 0 && r !== 2) process.exit(r);
}

require("../build/Release/tensorflow-binding.node");
