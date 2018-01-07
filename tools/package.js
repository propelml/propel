#!/usr/bin/env node
// Builds and tests an npm package.
// Currently this only builds mac cpu TF releases.
// Soon: other OSes.
const run = require("./run");
const fs = require("fs");
const path = require("path");
const {execSync} = require("child_process");

let distDir = path.join(run.root + "/dist");
run.rmrf(distDir)
run.mkdir(distDir)
run.sh("./tools/tsc.js");

function createPackageJson(src, dst) {
  let p = JSON.parse(fs.readFileSync(src, "utf8"));
  delete p["dependencies"];
  delete p["devDependendies"];
  delete p["peerDependendies"];
  delete p["private"];
  let s = JSON.stringify(p, null, 2);
  fs.writeFileSync(dst, s);
  console.log("Wrote " + dst);
}

// Create the package.json.
createPackageJson("package.json", distDir + "/package.json");

// Copy other files.
fs.writeFileSync(distDir + "/README.md", "See http://p_____.org\n");

// Copy over the TF binding.
// mac only at the moment.
fs.copyFileSync("build/Release/tensorflow-binding.node",
                distDir + "/tensorflow-binding.node");
fs.copyFileSync("build/Release/libtensorflow.so",
                distDir + "/libtensorflow.so");
fs.copyFileSync("build/Release/libtensorflow_framework.so",
                distDir + "/libtensorflow_framework.so");

process.chdir(distDir);
console.log("npm pack");
let pkgFn = execSync("npm pack", {encoding: "utf8"});
pkgFn = path.join(distDir, pkgFn);
console.log("pkgFn", pkgFn);

// Now test the package
const testDir = "/tmp/propel_npm_test";
run.rmrf(testDir);
run.mkdir(testDir);
// Copy package.json into the npm directory so it doesn't warn
// about not having description or repository fields. -_-
// Pretend we're the tar module.
createPackageJson(run.root + "/node_modules/tar/package.json",
                  path.join(testDir, "package.json"));

process.chdir(testDir);
run.sh("npm install " + pkgFn);

// Quick test that it works.
fs.writeFileSync("test.js", "require('propel/api_test')\n");
run.sh("node test.js");

console.log("package tested and ready", pkgFn);
console.log("To publish, run: npm publish", pkgFn);

