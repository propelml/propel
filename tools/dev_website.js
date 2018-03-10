#!/usr/bin/env node
// This is an alternative to ./tools/build_website which does fast
// incremental builds. The main difference is that the production website
// pre-generates static html for the documentation.
// Eventually we intend to merge the two scripts.
const run = require("./run");
const fs = require("fs");
const Bundler = require("parcel-bundler");
require("ts-node").register({"typeCheck": true });
const gendoc = require("./gendoc.ts");

(async() => {
  run.mkdir("build");
  run.mkdir("build/website");  // for docs.json
  run.mkdir("build/dev_website");
  run.mkdir("build/dev_website/docs");
  run.mkdir("build/dev_website/notebook");
  run.mkdir("build/dev_website/src"); // Needed for npy_test

  run.symlink(run.root + "/website/", "build/dev_website/static");
  run.symlink(run.root + "/website/img", "build/dev_website/img");
  run.symlink(run.root + "/deps/data/", "build/dev_website/data");
  // Needed for npy_test
  run.symlink(run.root + "/src/testdata/", "build/dev_website/src/testdata");

  const opts = {
    cache: true,
    hmr: false,
    logLevel: process.env.CI ? 1 : null,
    minify: false,
    outDir: "build/dev_website/",
    production: false,
    publicUrl: "/",
    watch: true
  }

  const sandboxBunder = new Bundler("website/sandbox.ts", opts);
  await sandboxBunder.bundle();

  const docs = gendoc.genJSON();
  const docsJson = JSON.stringify(docs, null, 2);
  fs.writeFileSync("build/dev_website/docs.json", docsJson);
  // There's a readFileSync for the docs.json that looks in build/website/
  // so we write it there too.
  fs.writeFileSync("build/website/docs.json", docsJson);

  const indexBunder = new Bundler("website/index.html", opts);
  const port = 8080
  indexBunder.serve(port);

  console.log(`Propel dev http://localhost:${port}/`);
})();

