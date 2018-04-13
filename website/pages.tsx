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
import { Footer, GlobalHeader } from "./common";
import * as db from "./db";
import { Docs } from "./docs";
import { PropelIndex } from "./index";
import * as nb from "./nb";

export interface Page {
  title: string;
  path: string;
  root: any;
  route: RegExp;
}

export function renderPage(p: Page): void {
  render(h(p.root, null), document.body, document.body.children[0]);
}

export const References = (props) => {
  const refhtml = readFileSync(__dirname + "/references.html", "utf8");
  return (
    <div class="references">
      <GlobalHeader subtitle="References" />
      <p class="references__details">
      This work is inspired by and built upon great technologies.</p>
      <div dangerouslySetInnerHTML={ { __html: refhtml } } />
      <Footer />
    </div>
  );
};

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
    <script src="/main.js"></script>
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
  constructor(props) {
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
    console.log("document.location.pathname", document.location.pathname);
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
    path: "index.html",
    root: PropelIndex,
    route: /^\/(index.html)?$/,
  },
  {
    title: "Propel Notebook",
    path: "notebook/index.html",
    root: nb.NotebookRoot,
    route: /^\/notebook/,
  },
  {
    title: "Propel References",
    path: "references/index.html",
    root: References,
    route: /^\/references(.html)?/,
  },
  {
    title: "Propel Docs",
    path: "docs/index.html",
    root: Docs,
    route: /^\/docs/,
  },
];
