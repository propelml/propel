/*!
   Copyright 2018 Propel http://propel.site/.  All rights reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */
// tslint:disable:variable-name
// This is the propelml.org website. It is used both server-side and
// client-side for generating HTML.
import { readFileSync } from "fs";
import { Component, h, render } from "preact";
import { div, GlobalHeader, p, UserMenu } from "./common";
import * as db from "./db";
import { Docs } from "./docs";
import * as nb from "./notebook";
const { version } = require("../package.json");

export interface Page {
  title: string;
  path: string;
  root: any;
  route: RegExp;
}

export function renderPage(p: Page): void {
  render(h(p.root, null), document.body, document.body.children[0]);
}

function b(...children) {
  return h("b", null, ...children);
}

function i(...children) {
  return h("i", null, ...children);
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

export const References = (props) => {
  const refhtml = readFileSync(__dirname + "/references.html", "utf8");
  return div("references",
    h(GlobalHeader, { subtitle: "References" }),
    p("This work is inspired by and built upon great technologies."),
    h("div", {
      dangerouslySetInnerHTML: { __html: refhtml },
    }),
  );
};

export const PropelIndex = (props) => {
  return div("index",
    h(Splash, props),
    h(Intro, null),
    h(UseIt, null),
    h(Perks, null),
    h(ReferencesFooter, null),
  );
};

const ReferencesFooter = () =>
  div("footer", link("references.html", "References"));

const Splash = (props) =>
  div("splash",
    // TODO "header" should be inside GlobalHeader.
    h("header", null,
      h(GlobalHeader, null,
        h("a", { href: "/notebook" }, "Notebook"),
        h(UserMenu, props)
      ),
    )
  );

const Intro = () =>
  div("intro flex-row",
    div("flex-cell",
      h("h2", {}, "Differential Programming in JavaScript"),
      p(
        b("Propel"), ` provides a GPU-backed numpy-like infrastructure
        for scientific computing in JavaScript.  JavaScript is a fast,
        dynamic language which, we think, could act as an ideal workflow
        for scientific programmers of all sorts.`),
      p(
        headerButton("/docs", "API Ref"),
        // Hide notebook link until more developed.
        // headerButton("/notebook", "Notebook"),
        headerButton("http://github.com/propelml/propel", "Github")
      )
    ),
    div("intro-notebook flex-cell", nb.notebook(tanhGrads))
  );

const UseIt = () =>
  div("use-it",
    div("use-it-inner",
      h("p", { "class": "snippet-title" }, "Use it in Node:"),
      fixed("npm install propel\nimport { grad } from \"propel\";"),
      h("p", { "class": "snippet-title" }, "Use it in a browser:"),
      fixed(`<script src="https://unpkg.com/propel@${version}"></script>`)
    )
  );

const Perks = () =>
  div("perks",
    div("flex-row",
      div("flex-cell",
        h("h2", { "class": "world" }, "Run anywhere."),
        p(
          `Propel runs both in the browser and natively from Node. In
          both environments Propel is able to use GPU hardware for
          computations.  In the browser it utilizes WebGL through `,
          link("https://deeplearnjs.org/", "deeplearn.js"),
          " and on Node it uses TensorFlow's ",
          link("https://www.tensorflow.org/install/install_c", "C API"),
          "."
        ),
      ),
      div("flex-cell",
        h("h2", { "class": "upward" }, "Phd optional."),
        p(
          "Propel has an imperative ",
          link("https://github.com/HIPS/autograd", "autograd"),
          `-style API.  Computation graphs are traced as
          you run them -- a general purpose `,
          i("gradient function"),
          ` provides an elegant interface to backpropagation.`
        ),
      ),
    ),
    div("flex-row",
      div("flex-cell",
        h("h2", { "class": "chip" }, "Did somebody say GPUs?"),
        p(
          `Browsers are great for demos, but they are not a great numerics
          platform. WebGL is a far cry from CUDA. By running Propel outside
          of the browser, users will be able to target multiple GPUs and
          make TCP connections. The models developed server-side will be
          much easier to deploy as HTML demos.`
        ),
      ),
      div("flex-cell",
        h("h2", { "class": "lightning" }, "Let's do this."),
        p(`The basic propel npm package is javascript only,
          without TensorFlow bindings. To upgrade your speed dramatically
          install`),
        fixed([
          "npm install propel_mac",
          "npm install propel_windows",
          "npm install propel_linux",
          "npm install propel_linux_gpu",
        ].join("\n"))
      ),
    )
  );

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

export interface RouterState {
  userInfo?: db.UserInfo;
  loadingAuth: boolean;
}

// The root of all pages of the propel website.
// Handles auth.
export class Router extends Component<any, RouterState> {
  constructor(props)  {
    super(props);
    this.state = {
      userInfo: null,
      loadingAuth: true,
    };
  }

  unsubscribe: db.UnsubscribeCb;
  componentWillMount() {
    this.unsubscribe = db.active.subscribeAuthChange((userInfo) => {
      this.setState({ loadingAuth: false, userInfo });
    });
  }

  componentWillUnmount() {
    this.unsubscribe();
  }

  render() {
    const page = route(document.location.pathname);
    return h(page.root, { userInfo: this.state.userInfo });
  }
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
