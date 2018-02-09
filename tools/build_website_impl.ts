import * as fs from "fs";
import { JSDOM } from "jsdom";
import { join } from "path";
import { renderSync } from "sass";
import { drainExecuteQueue } from "../website/notebook";
import * as website from "../website/website";
import * as run from "./run";

// tslint:disable:no-reference
/// <reference path="deps/firebase/firebase.d.ts" />

const websiteRoot = run.root + "/build/website/";

async function renderToHtmlWithJsdom(page: website.Page): Promise<string> {
  const jsdomConfig = { };
  const window = new JSDOM("", jsdomConfig).window;

  global["window"] = window;
  global["self"] = window;
  global["document"] = window.document;
  global["navigator"] = window.navigator;
  global["Node"] = window.Node;
  global["getComputedStyle"] = window.getComputedStyle;

  website.renderPage(page);

  const p = new Promise<string>((resolve, reject) => {
    window.addEventListener("load", async() => {
      try {
        await drainExecuteQueue();
        const bodyHtml = document.body.innerHTML;
        const html =  website.getHTML(page.title, bodyHtml);
        resolve(html);
      } catch (e) {
        reject(e);
      }
    });
  });
  return p;
}

async function writePages() {
  for (const page of website.pages) {
    const html = await renderToHtmlWithJsdom(page);
    const fn = join(run.root, "build", page.path);
    fs.writeFileSync(fn, html);
    console.log("Wrote", fn);
  }
}

function scss(inFile, outFile) {
  const options = {
    file: inFile,
    includePaths: ["./website"],
  };
  const result = renderSync(options).css.toString("utf8");
  console.log("scss", inFile, outFile);
  fs.writeFileSync(outFile, result);
}

run.mkdir("build");
run.mkdir("build/website");
run.mkdir("build/website/docs");
run.mkdir("build/website/notebook");

run.symlink(run.root + "/website/", "build/website/static");

scss("website/main.scss", join(websiteRoot, "bundle.css"));

writePages().then(() => {
  run.parcel("website/website_main.ts", "build/website");
  console.log("Website built in", websiteRoot);
  // Firebase keeps network connections open, so we have force exit the process.
  process.exit(0);
});
