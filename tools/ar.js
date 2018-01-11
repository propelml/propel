#!/usr/bin/env node
const run = require("./run");
const path = require("path");
const fs = require("fs");
const url = require("url");
const {execSync} = require("child_process");

function makeIndex(dir="/") {
  function page(body) {
    return `<!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Propel Archive</title>
        <meta id="viewport" name="viewport" content="width=device-width,
          minimum-scale=1.0, maximum-scale=1.0, user-scalable=no"/>
        <link rel="stylesheet" href="http://propelml.org/normalize.css"/>
        <link rel="stylesheet" href="http://propelml.org/skeleton.css"/>
        <link rel="stylesheet" href="http://propelml.org/style.css"/>
        <link rel="icon" type="image/png" href="http://propelml.org/favicon.png">
        <body>
          <div class="container">
            <section class="splash">
              <p>Archive</p>
              ${body}
            </section>
          </div>
        </html>
      `;
  }

  let s3dir = "s3://" + path.join("ar.propelml.org", dir);
  let output = execSync("aws s3 ls " + s3dir).toString();
  let files = [];
  for (const line of output.split("\n")) {
    const fn = line.split(/\s/).pop();
    if (fn.length && fn !== "index.html") {
      files.push(fn);
      console.log(fn);
    }
  }

  let body = files.map(fn => `<li><a href="${fn}">${fn}</a></li>`).join("\n");
  let html = page(body);
  let indexFn = url.resolve(s3dir, "index.html");
  let cmd = `aws s3 cp - ${indexFn} --content-type text/html `;
  console.log(cmd);
  execSync(cmd, { input: html } )
  console.log(path.join("ar.propelml.org", dir, "index.html"));
}

// Main:

const srcFiles = process.argv.slice(2);
if (srcFiles.length === 0) {
  console.log("This tool uploads a file to archive, ar.propelml.org");
  console.log("Example: ./tools/ar.js build/propel/propel-2.99.2.tgz");
}

const debug = false;
if (!debug) {
  for (const src of srcFiles) {
    const fn = path.basename(src)
    const dst = "s3://ar.propelml.org/" + fn;
    console.log("src", src, "dst", dst);
    const isDir = fs.statSync(src).isDirectory();
    run.sh(`aws s3 sync ${src} ${dst} --exclude="*/*.tgz" --follow-symlinks`); 
    if (isDir) {
      makeIndex(fn);
    }
  }
}

makeIndex('/');
