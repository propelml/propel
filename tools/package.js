#!/usr/bin/env node
// Builds and tests an npm package.
// Currently this only builds mac cpu TF releases.
// Soon: other OSes.
const run = require("./run");
const fs = require("fs");
const path = require("path");
const config = require("./config");
const {execSync} = require("child_process");

const clean = process.argv.includes("clean");

if (clean) {
  run.rmrf("build")
}

const v = run.version();
console.log("version", v);

// Build the binding.
run.sh("node tools/build_binding.js");


function createPackageJson(src, dst, packageJson = {}) {
  let p = JSON.parse(fs.readFileSync(src, "utf8"));
  p = Object.assign(p, packageJson);
  delete p["dependencies"];
  delete p["devDependencies"];
  delete p["private"];
  let s = JSON.stringify(p, null, 2);
  fs.writeFileSync(dst, s);
  console.log("Wrote " + dst);
}

function npmPack(name, cb) {
  const distDir = run.root + "/build/" + name;
  if (clean) {
    run.rmrf(distDir)
  }
  run.mkdir(distDir)
  fs.writeFileSync(distDir + "/README.md", "See http://p_____.org\n");
  if (cb) cb(distDir);
  process.chdir(distDir);
  console.log("npm pack");
  const pkgFn = path.resolve(execSync("npm pack", {encoding: "utf8"}));
  process.chdir(run.root);
  console.log("pkgFn", pkgFn);
  return pkgFn;
}

function webpackIfDNE(configName, fn) {
  if (clean  || !fs.existsSync(fn)) {
    console.log("Webpack %s", fn, configName);
    run.sh("./tools/webpack.js --config-name=" + configName)
  } else {
    console.log("Skipping webpack %s", fn, configName);
  }
}

function buildAndTest() {
  const propelPkgFn = npmPack("propel", (distDir) => {
    webpackIfDNE("propel_node", distDir + "/propel_node.js");
    webpackIfDNE("propel_web", distDir + "/propel_web.js");
    // webpackIfDNE("tests_node", distDir + "/tests_node.js");
    // webpackIfDNE("tests_web", distDir + "/tests_web.js");
    createPackageJson("package.json", distDir + "/package.json", {
      name: "propel",
      main: "propel_node.js",
      unpkg: "propel_web.js",
    });
  });

  const tfPkgFn = npmPack(config.tfPkg, (distDir) => {
    fs.copyFileSync("load_binding.js", distDir + "/load_binding.js");
    // Copy over the TF binding.
    // TODO Mac only at the moment. Make this work on windows.
    fs.copyFileSync("build/Release/tensorflow-binding.node",
                    distDir + "/tensorflow-binding.node");
    fs.copyFileSync("build/Release/libtensorflow.so",
                    distDir + "/libtensorflow.so");
    fs.copyFileSync("build/Release/libtensorflow_framework.so",
                    distDir + "/libtensorflow_framework.so");
    createPackageJson("package.json", distDir + "/package.json", {
      name: config.tfPkg,
      main: "load_binding.js",
    });
  });

  // Now test the package
  const testDir = "/tmp/propel_npm_test";
  run.rmrf(testDir);
  run.mkdir(testDir);

  // Pretend we're the tar module. Copy package.json into the npm directory so it
  // doesn't warn about not having description or repository fields.
  createPackageJson(run.root + "/node_modules/tar/package.json",
                    path.join(testDir, "package.json"));

  process.chdir(testDir);
  run.sh("npm install " + propelPkgFn);
  run.sh("npm install " + tfPkgFn);

  // Quick test that it works.
  fs.writeFileSync("test.js", `
    let propel = require('propel');
    let $ = require('propel').$;
    console.log($([1, 2, 3]).mul(42));
    console.log("Using backend: %s", propel.backend);
    if (propel.backend !== "tf") throw Error("Bad backend");
  `);
  run.sh("node test.js");

  console.log("npm publish %s", propelPkgFn);
  console.log("npm publish %s", tfPkgFn);
  return [propelPkgFn, tfPkgFn];
}

function symlink(a, b) {
  console.log("symlink", a, b);
  try {
    fs.symlinkSync(a, b)
  } catch (e) {
    if (e.code === "EEXIST") {
      console.log("EEXIST");
    } else {
      throw e;
    }
  }
}


if (true) {
  buildAndTest();
}

// chdir for symlink.
process.chdir(run.root + "/build");

console.log("\n\nPackage tested and ready.");
for (const name of ["propel", config.tfPkg]) {
  let vname = `${name}-${v}`;
  symlink(name, vname)
  console.log("./tools/ar.js %s", "build/" + vname);
}

