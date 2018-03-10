import * as fs from "fs";
import { JSDOM, VirtualConsole } from "jsdom";
import { join, resolve } from "path";
import { renderSync } from "sass";
import * as toArrayBuffer from "to-arraybuffer";
import { URL } from "url";
import { drainExecuteQueue, initSandbox } from "../website/nb";
import * as website from "../website/website";
import * as gendoc from "./gendoc";
import * as run from "./run";

// tslint:disable:no-reference
/// <reference path="deps/firebase/firebase.d.ts" />

const websiteRoot = run.root + "/build/website/";

// A minimal polyfill for fetch() that is used in the notebook sandbox while
// rendering pages to static HTML.
async function fetchFromSandbox(
    input: RequestInfo, init?: RequestInit): Promise<Response> {
  if (typeof input !== "string" || /^\[a-z]+:|^[\\\/]{2}/.test(input)) {
    throw new Error("fetch() only supports relative URLs");
  }
  const path = resolve(`${websiteRoot}/${input}`);
  console.log(`fetch: ${path}`);
  const data = fs.readFileSync(path);

  return {
    async arrayBuffer() { return toArrayBuffer(data); },
    async text() { return data.toString("utf8"); }
  } as any as Response;
}

async function renderToHtmlWithJsdom(page: website.Page): Promise<string> {
  const window = new JSDOM("", {}).window;

  global["window"] = window;
  global["self"] = window;
  global["document"] = window.document;
  global["navigator"] = window.navigator;
  global["Node"] = window.Node;
  global["getComputedStyle"] = window.getComputedStyle;

  // The notebook normally uses an iframe to create a sandbox to run notebook
  // cells in. Although it is possible to make this work in JSDOM, it requires
  // some workarounds in the notebook code, and the iframe ends up in the
  // static rendered HTML. Instead, create a separate JSDOM context and tell
  // the sandbox to use that instead.

  // Create JSDOM console object and forward output to the node.js console.
  // We do this explicitly in order to suppress "errors" (really just warnings)
  // about JSDOM not supporting canvas.
  const virtualConsole = new VirtualConsole();
  virtualConsole.sendTo(console, { omitJSDOMErrors: true });

  const { window: sbWindow } = new JSDOM("", {
    resources: "usable",
    runScripts: "dangerously",
    url: new URL(`file:///${__dirname}/../build/website/sandbox`).href,
    virtualConsole,
    beforeParse(sbWindow: any) {
      // Add a fake parent window reference - the sandbox needs to be able
      // to call window.parent.postMessage() to communicate with the host.
      // In JSDOM window.parent is a getter, but setting _parent works.
      sbWindow._parent = window;
      // Inject a minimal fetch() shim.
      sbWindow.fetch = fetchFromSandbox;
    }
  });
  const sandboxScript =
    fs.readFileSync(`${__dirname}/../build/website/sandbox.js`, "utf8");
  sbWindow.eval(sandboxScript);
  initSandbox(sbWindow);

  website.renderPage(page);
  await new Promise(res => window.addEventListener("load", res));
  await drainExecuteQueue();

  const bodyHtml = document.body.innerHTML;
  const html = website.getHTML(page.title, bodyHtml);
  return html;
}

async function writePages() {
  for (const page of website.pages) {
    console.log(`rendering: ${page.path}`);
    const html = await renderToHtmlWithJsdom(page);
    const fn = join(run.root, "build", page.path);
    fs.writeFileSync(fn, html);
  }
}

function scss(inFile, outFile) {
  const options = {
    file: inFile,
    includePaths: ["./website"],
  };
  let result = renderSync(options).css.toString("utf8");
  // Due to how parcel works, we have to include relative URLs in the sass
  // file website/main.scss
  // Example:
  //  background-image: url(./img/deleteButtonOutline.svg);
  // When bundle.scss is built in ./build/website/ it then can't resolve these
  // relative path names. Like if it's in /docs.
  // So just hack it here... All this should be replaced with parcel soon.
  result = result.replace(/.\/img\//g, "/static/img/");
  result = result.replace(/\(img\//g, "(/static/img/");
  result = result.replace(/\("img\//g, `("/static/img/`);
  console.log("scss", inFile, outFile);
  fs.writeFileSync(outFile, result);
}

process.on("unhandledRejection", e => { throw e; });

(async() => {
  run.mkdir("build");
  run.mkdir("build/website");
  run.mkdir("build/website/docs");
  run.mkdir("build/website/notebook");
  run.mkdir("build/website/src"); // Needed for npy_test

  run.symlink(run.root + "/website/", "build/website/static");
  run.symlink(run.root + "/deps/data/", "build/website/data");
  // Needed for npy_test
  run.symlink(run.root + "/src/testdata/", "build/website/src/testdata");

  gendoc.writeJSON("build/website/docs.json");

  scss("website/main.scss", join(websiteRoot, "bundle.css"));

  // Bundle all scripts that are used on the website (except the sandbox).
  await run.parcel("website/main.ts", "build/website");
  // Bundle all scripts that run in the notebook sandbox iframe.
  await run.parcel("website/sandbox.ts", "build/website");

  // Render all pages to static HTML. This needs to be run *after* parcel,
  // because sandbox.js is used to render notebook cells.
  await writePages();

  console.log("Website built in", websiteRoot);

  // Firebase keeps network connections open, so we have force exit the process.
  process.exit(0);
})();
