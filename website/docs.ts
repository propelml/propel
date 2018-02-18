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
import { h } from "preact";
import { div, p, PropelLogo } from "./common";
import * as nb from "./notebook";

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

    out.push(h("h2", { id: tag, "class": "name" },
      h("a", { "href": "#" + tag }, entry.name)));

    if (entry.typestr) {
      out.push(div("typestr", entry.typestr));
    }
    if (entry.docstr) {
      out.push(markupDocStr(entry.docstr));
    }

    const sourceLink = !entry.sourceUrl ? null
      : h("a", { "class": "source-link", "href": entry.sourceUrl }, " source");

    return div("doc-entry", sourceLink, ...out);
  });
  return div("doc-entries", ...entries);
};

function startsWithUpperCase(s: string): boolean {
  return s[0].toLowerCase() !== s[0];
}

export function Docs(props) {
  let docs: DocEntry[] = require("./docs.json");
  docs = docs.sort((a, b) => {
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
      h(PropelLogo, { subtitle: "Docs" }),
      h(DocIndex, { docs }),
    ),
    h(DocEntries, { docs }),
  );
}
