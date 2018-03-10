#!/usr/bin/env node
/*
   Pre-generates static html.

     ./tools/dev_website.js prod 

   prod flag: minimized output

     ./tools/dev_website.js gendoc 

   gendoc flag: rebuild docs.json

     ./tools/dev_website.js build prod

   build production website and exit.
*/
const run = require("./run");
const fs = require("fs");
const Bundler = require("parcel-bundler");
require("ts-node").register({ typeCheck: true });

const prodFlag = process.argv.indexOf("prod") >= 0;
exports.prodFlag = prodFlag;

let wdir = "build/dev_website/";
exports.wdir = wdir;

async function bundler(build) {
  run.mkdir("build");
  run.mkdir(wdir);
  run.mkdir(wdir + "src"); // Needed for npy_test
  run.symlink(run.root + "/website/", wdir + "static");
  run.symlink(run.root + "/website/img", wdir + "img");
  run.symlink(run.root + "/deps/data/", wdir + "data");
  // Needed for npy_test
  run.symlink(run.root + "/src/testdata/", wdir + "src/testdata");

  const opts = {
    cache: true,
    hmr: false,
    logLevel: process.env.CI ? 1 : null,
    minify: prodFlag,
    outDir: wdir,
    production: prodFlag,
    publicUrl: "/",
    watch: !build
  };

  let b = new Bundler("website/sandbox.ts", opts);
  await b.bundle();

  b = new Bundler("tools/test_dl.ts", opts);
  await b.bundle();

  b = new Bundler("tools/test_website.ts", opts);
  await b.bundle();

  run.gendoc(wdir + "docs.json");

  const indexBunder = new Bundler("website/index.html", opts);
  return indexBunder;
}

const port = 8080;
async function devWebsiteServer(build) {
  const b = await bundler(build);
  return await b.serve(port);
}
exports.devWebsiteServer = devWebsiteServer;

if (require.main === module) {
  devWebsiteServer(false);
  console.log(`Propel http://localhost:${port}/`);
}
