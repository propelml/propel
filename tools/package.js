#!/usr/bin/env node
// Builds and tests an npm package.
// Currently this only builds mac cpu TF releases.
// Soon: other OSes.
const run = require("./run");
const fs = require("fs");
const path = require("path");
const config = require("./config");
const { execSync } = require("child_process");

const clean = process.argv.includes("clean");

if (clean) {
  run.rmrf("build");
}

const v = run.version();
console.log("version", v);

// Build the binding.
run.sh("node tools/build_binding.js");

function createPackageJson(src, dst, packageJson = {}) {
  let p = JSON.parse(fs.readFileSync(src, "utf8"));
  delete p["dependencies"];
  delete p["devDependencies"];
  delete p["private"];
  p = Object.assign(p, packageJson);
  let s = JSON.stringify(p, null, 2);
  fs.writeFileSync(dst, s);
  console.log("Wrote " + dst);
}

function checkNoTars(dir) {
  for (const f of fs.readdirSync(dir)) {
    if (f.match(/\.tgz$/) != null ||
        f.match(/\.tar.gz$/) != null ||
        f.match(/\.zip$/) != null) {
      console.error("Bad filename in package dir", dir, f);
      process.exit(1);
    }
  }
}

function npmPack(name, cb) {
  const distDir = run.root + "/build/" + name;
  if (clean) {
    run.rmrf(distDir);
  }
  run.mkdir(distDir);
  fs.writeFileSync(distDir + "/README.md", "See http://propelml.org\n");

  cb(distDir);
  checkNoTars(distDir);

  process.chdir(distDir);
  console.log("npm pack");
  const pkgFn = path.resolve(execSync("npm pack", { encoding: "utf8" })).trim();
  process.chdir(run.root);

  // Move the tarball created by npm to the root of the build dir. This is so
  // it doesn't accidentally get included in future builds.
  const pkgBaseName = path.basename(pkgFn);
  const newPkgFn = path.resolve(run.root, "build", pkgBaseName);
  fs.renameSync(pkgFn, newPkgFn);

  console.log("pkgFn", newPkgFn);
  return newPkgFn;
}

function buildAndTest() {
  const propelPkgFn = npmPack("propel", distDir => {
    run.parcel("src/api.ts", distDir);
    const genFn = distDir + "/api.js";
    const mainFn = distDir + "/propel.js";
    fs.renameSync(genFn, mainFn);

    let c = fs.readFileSync(mainFn, "utf8");
    fs.writeFileSync(mainFn, c + `
      if (typeof window !== "undefined") {
        propel = require(1);
      } else {
        module.exports = require(1);
      }
    `);
    createPackageJson("package.json", distDir + "/package.json", {
      name: "propel",
      main: "propel.js",
    });
  });

  const tfPkgFn = npmPack(config.tfPkg, distDir => {
    fs.copyFileSync("src/load_binding.js", distDir + "/load_binding.js");
    // Copy over the TF binding.
    fs.copyFileSync("build/Release/tensorflow-binding.node",
                    distDir + "/tensorflow-binding.node"
    );
    if (process.platform === "win32") {
      fs.copyFileSync("build/Release/tensorflow.dll",
                      distDir + "/tensorflow.dll");
    } else {
      fs.copyFileSync("build/Release/libtensorflow.so",
                      distDir + "/libtensorflow.so");
      fs.copyFileSync("build/Release/libtensorflow_framework.so",
                      distDir + "/libtensorflow_framework.so");
    }
    createPackageJson("package.json", distDir + "/package.json", {
      name: config.tfPkg,
      main: "load_binding.js",
      dependencies: { propel: v }
    });
  });

  // Now test the package
  const tmpDir = process.env.TEMP || process.env.TMPDIR || "/tmp";
  const testDir = path.join(tmpDir, "propel_npm_test");
  run.rmrf(testDir);
  run.mkdir(testDir);

  // Pretend we're the tar module. Copy package.json into the npm directory so it
  // doesn't warn about not having description or repository fields.
  createPackageJson(run.root + "/node_modules/tar/package.json",
                    path.join(testDir, "package.json"));

  process.chdir(testDir);
  execSync("npm install " + propelPkgFn, { stdio: "inherit" });
  execSync("npm install " + tfPkgFn, { stdio: "inherit" });

  // Quick test that it works.
  fs.writeFileSync("test.js", `
    let propel = require('propel');
    console.log("propel", propel);
    let T = require('propel').T;
    console.log(T([1, 2, 3]).mul(42));
    console.log("Using backend: %s", propel.backend);
    if (propel.backend !== "tf") throw Error("Bad backend");
  `);
  run.sh("node test.js");

  console.log("npm publish %s", propelPkgFn);
  console.log("npm publish %s", tfPkgFn);
  return [propelPkgFn, tfPkgFn];
}

const skipBuild = (process.argv.indexOf("skip-build") >= 0);
if (!skipBuild) {
  buildAndTest();
}

// chdir for symlink.
process.chdir(run.root + "/build");

console.log("\n\nPackage tested and ready.");
for (const name of ["propel", config.tfPkg]) {
  let vname = `${name}-${v}`;
  run.symlink(name, vname);
  // console.log("./tools/ar.js %s", "build/" + vname);
}
