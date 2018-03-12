#!/usr/bin/env node
// This program generates static html from the react website by using
// puppeteer (chrome) and the parcel development webserver.
// Available command line arguments: prod, gendoc
// Use ./tools/dev_website.js for fast incremental builds.
let ncp = require("ncp");
let run = require("./run");
const { pages } = require("../website/website");
let fs = require("fs");
let path = require("path");
let { format } = require("util");
let puppeteer = require("puppeteer");
const { devWebsiteServer } = require("./dev_website");
require("../src/util"); // So it throws on unhandled promises.

const headless = process.env.CI != null;

async function puppetRender(browser, url) {
  return new Promise(async (resolve, reject) => {
    const page = await browser.newPage();

    async function onMessage(msg) {
      const values = msg.args.map(
        v =>
          v._remoteObject.value !== undefined
            ? v._remoteObject.value
            : `[[${v._remoteObject.type}]]`
      );
      const text = format.apply(null, values);

      console.log(text);

      if (text.match(/Propel onload/)) {
        let html = await page.evaluate(
          () => document.documentElement.innerHTML
        );
        resolve(html);
      }
    }

    page.on("console", onMessage);
    page.on("pageerror", err => {
      console.log("pagerror");
      reject(err);
    });
    page.goto(url, { timeout: 0 });
  });
}

function cp(src, dst) {
  return new Promise((resolve, reject) => {
    ncp(src, dst, err => {
      if (err) {
        reject(err);
      } else {
        resolve();
      }
    });
  });
}

host = "http://localhost:8080/";

async function writePages(browser, dir) {
  for (const page of pages) {
    const html = await puppetRender(
      browser,
      host + page.path.replace("index.html", "")
    );
    const fn = path.join(dir, page.path);
    const parentDir = path.dirname(fn);
    run.mkdir(parentDir);
    fs.writeFileSync(fn, html);
    console.log("write", fn);
  }
}

(async () => {
  let dir = "build/website_render";
  run.rmrf(dir);
  let server = await devWebsiteServer();
  const browser = await puppeteer.launch({
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
    headless
  });
  await cp("build/dev_website", dir);

  await writePages(browser, dir);
  await browser.close();

  process.exit(0);
})();
