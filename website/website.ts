// tslint:disable:variable-name
// This is the propelml.org website. It is used both server-side and
// client-side for generating HTML.
import { readFileSync } from "fs";
import { h, render } from "preact";
import * as nb from "./notebook";
const { version } = require("../package.json");

export interface DocEntry {
  kind: "class" | "method" | "property";
  name: string;
  typestr?: string;
  docstr?: string;
  args?: ArgEntry[];
  retType?: string;
  sourceUrl?: string;
}

export interface ArgEntry {
  name: string;
  typestr?: string;
  docstr?: string;
}

export interface Page {
  title: string;
  path: string;
  root: any;
  route: RegExp;
}

export function renderPage(p: Page): void {
  render(h(p.root, null), document.body, document.body.children[0]);
}

function p(...children) {
  return h("p", null, ...children);
}

function b(...children) {
  return h("b", null, ...children);
}

function i(...children) {
  return h("i", null, ...children);
}

export function div(className, ...children) {
  return h("div", { "class": className }, ...children);
}

function headerButton(href, text) {
  return h("a", { "class": "button header-button", "href": href }, text);
}

function link(href, ...children) {
  return h("a", {href}, ...children);
}

function fixed(code: string): JSX.Element {
  return h(nb.FixedCell, { code });
}

function toTagName(s: string): string {
  return s.replace(/[.$]/g, "_");
}

function isIndented(s: string): boolean {
  return s.match(/^  +[^\s]/) != null;
}

function unindent(s: string): string {
  return s.replace(/^  /, "");
}

// Given some bit of documentation text, this function can detect indented
// portions denoting examples and mark them up with <script type="notebook">.
export function markupDocStr(docstr: string): JSX.Element {
  const input = docstr.split("\n");
  let buf = [];
  const elements = [];

  let state: "normal" | "code" = "normal";

  function evalState(line) {
    switch (state) {
      case "normal":
        if (line == null || isIndented(line)) {
          state = "code";
          const src = buf.join("\n");
          elements.push(p(src));
          buf = [];
        }
        break;
      case "code":
        if (line == null || !isIndented(line)) {
          state = "normal";
          const src = buf.map(unindent).join("\n");
          elements.push(nb.notebook(src));
          buf = [];
        }
        break;
    }
  }

  for (let i = 0; i < input.length; ++i) {
    const line = input[i];
    evalState(line);
    buf.push(line);
  }
  evalState(null);

  return div("docstr", elements);
}

const DocIndex = ({docs}) => {
  const list = docs.map(entry => {
    const tag = toTagName(entry.name);
    const className = "name " + entry.kind;
    return h("li", null, h("a", { href: "#" + tag, "class": className },
      entry.name));
  });
  return h("ol", { "class": "docindex" }, list);
};

const DocEntries = ({docs}) => {
  const entries = docs.map(entry => {
    const tag = toTagName(entry.name);
    const out = [];

    out.push(h("h2", { id: tag, "class": "name" }, entry.name,
      entry.sourceUrl
        ? h("a", { "class": "source-link", "href": entry.sourceUrl }, " source")
        : null));
    if (entry.typestr) {
      out.push(div("typestr", entry.typestr));
    }

    if (entry.docstr) {
      out.push(markupDocStr(entry.docstr));
    }

    return div("doc-entry", ...out);
  });
  return div("doc-entries", ...entries);
};

function startsWithUpperCase(s: string): boolean {
  return s[0].toLowerCase() !== s[0];
}

const Docs = (props) => {
  let docs: DocEntry[] = require("./docs.json");
  docs = docs.sort((a, b) => {
    // Special case "T" to be at the top of the docs.
    if (a.name === "T") return -1;
    if (b.name === "T") return 1;
    if (!startsWithUpperCase(a.name) && startsWithUpperCase(b.name)) {
      return -1;
    }
    if (startsWithUpperCase(a.name) && !startsWithUpperCase(b.name)) {
      return 1;
    }
    if (a.name < b.name) return -1;
    if (a.name > b.name) return 1;
    return 0;
  });

  return div("docs",
    div("panel",
      h("h1", null, "Propel"),
      h(DocIndex, { docs }),
    ),
    h(DocEntries, { docs }),
  );
};

export const References = (props) => {
  const refhtml = readFileSync(__dirname + "/references.html", "utf8");
  return div("references",
    h(nb.GlobalHeader, null),
    h("header", null,
      h("h1", null, "References"),
      p("This work is inspired by and built upon great technologies."),
    ),
    h("div", {
      dangerouslySetInnerHTML: { __html: refhtml },
    }),
  );
};

export const PropelIndex = (props) => {
  return div("index",
    h(Splash, null),
    h(Explainer, null),
    h(ReferencesFooter, null),
  );
};

const ReferencesFooter = (props) => {
  return h("section", { "class": "footer" }, link("references.html",
    "References"));
};

const Splash = (props) => {
  return h("section", { "class": "splash" },
    h("header", null,
      h("h1", null, "Propel"),
      h("p", null, "Differential Programming in JavaScript")
    ),
    p(
      headerButton("/docs", "API Ref"),
      // Hide notebook link until more developed.
      // headerButton("/notebook", "Notebook"),
      headerButton("http://github.com/propelml/propel", "Github")
    ),
    h("p", { "class": "snippet-title" }, "Use it in Node:"),
    fixed("npm install propel\nimport { grad } from \"propel\";"),
    h("p", { "class": "snippet-title" }, "Use it in a browser:"),
    fixed(`<script src="https://unpkg.com/propel@${version}"></script>`),
  );
};

const Explainer = (props) => {
  return div("explainer-container",
    div("explainer",
      div("explainer-text",
        p(b("Propel"), ` provides a GPU-backed numpy-like infrastructure
          for scientific computing in JavaScript.  JavaScript is a fast,
          dynamic language which, we think, could act as an ideal workflow
          for scientific programmers of all sorts.`),
        p(`Propel runs both in the browser and natively from Node. In
          both environments Propel is able to use GPU hardware for
          computations.  In the browser it utilizes WebGL through `,
          link("https://deeplearnjs.org/", "deeplearn.js"),
          " and on Node it uses TensorFlow's ",
          link("https://www.tensorflow.org/install/install_c", "C API"),
          "."),
        p("Propel has an imperative ",
          link("https://github.com/HIPS/autograd", "autograd"),
          `-style API, unlike TensorFlow.  Computation graphs are traced as
          you run them -- a general purpose `,
          i("gradient function"),
          ` provides an elegant interface to backpropagation. `),
        p(`Browsers are great for demos, but they are not a great numerics
          platform. WebGL is a far cry from CUDA. By running Propel outside
          of the browser, users will be able to target multiple GPUs and
          make TCP connections. The models developed server-side will be
          much easier to deploy as HTML demos.`),
        p(`The basic propel npm package is javascript only,
          without TensorFlow bindings. To upgrade your speed dramatically
          install`),
        fixed([
          "npm install propel_mac",
          "npm install propel_windows",
          "npm install propel_linux",
          "npm install propel_linux_gpu",
        ].join("\n")),
      ),
      div("explainer-notebook", nb.notebook(tanhGrads))
    )
  );
};

const tanhGrads = `
import { grad, linspace, plot } from "propel";

f = x => x.tanh();
x = linspace(-4, 4, 200);
plot(x, f(x),
     x, grad(f)(x),
     x, grad(grad(f))(x),
     x, grad(grad(grad(f)))(x),
     x, grad(grad(grad(grad(f))))(x))
`;

export let firebaseUrls = [
  "https://www.gstatic.com/firebasejs/4.9.0/firebase.js",
  "https://www.gstatic.com/firebasejs/4.9.0/firebase-auth.js",
  "https://www.gstatic.com/firebasejs/4.9.0/firebase-firestore.js",
];

// Called by tools/build_website.ts
export function getHTML(title, markup) {
  const scriptTags = firebaseUrls.map(u =>
    `<script src="${u}"></script>`).join("\n");
  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>${title}</title>
    <meta id="viewport" name="viewport" content="width=device-width,
      minimum-scale=1.0, maximum-scale=1.0, user-scalable=no"/>
    <link rel="stylesheet" href="/bundle.css"/>
    ${scriptTags}
    <script src="/website_main.js"></script>
    <link rel="icon" type="image/png" href="/static/favicon.png">
  </head>
  <body>${markup}
  <script async
    src="https://www.googletagmanager.com/gtag/js?id=UA-112187805-1"></script>
  <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'UA-112187805-1');
  </script>
  </body>
</html>`;
}

export function route(pathname: string): Page {
  for (const page of pages) {
    if (pathname.match(page.route)) {
      return page;
    }
  }
  // TODO 404 page
  return null;
}

export const pages: Page[] = [
  {
    title: "Propel ML",
    path: "website/index.html",
    root: PropelIndex,
    route: /^\/(index.html)?$/,
  },
  {
    title: "Propel Notebook",
    path: "website/notebook/index.html",
    root: nb.NotebookRoot,
    route: /^\/notebook/,
  },
  {
    title: "Propel References",
    path: "website/references.html",
    root: References,
    route: /^\/references/,
  },
  {
    title: "Propel Docs",
    path: "website/docs/index.html",
    root: Docs,
    route: /^\/docs/,
  },
];
