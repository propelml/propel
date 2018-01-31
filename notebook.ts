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
// tslint:disable:no-reference
/// <reference path="node_modules/@types/codemirror/index.d.ts" />
// React Notebook cells.
// Note that these are rendered and executed server-side using JSDOM and may
// are re-rendered client-side when users click "Run".
// The Propel code in the cells are executed server-side so they don't have to
// be run again on page load.
import { Component, h } from "preact";
import * as propel from "./api";
import * as matplotlib from "./matplotlib";
import * as mnist from "./mnist";
import { transpile } from "./nb_transpiler";
import { delay, IS_WEB } from "./util";

let cellTable = new Map<number, Cell>(); // Maps id to Cell.
let nextCellId = 1;
let lastExecutedCellId = null;
// If you use the eval function indirectly, by invoking it via a reference
// other than eval, as of ECMAScript 5 it works in the global scope rather than
// the local scope. This means, for instance, that function declarations create
// global functions, and that the code being evaluated doesn't have access to
// local variables within the scope where it's being called.
const globalEval = eval;

// FIXME This is a hack. When rendering server-side the ids keep increasing
// over all pages - that makes it not line up with how client-side ids will be
// generated.
export function resetIds() {
  nextCellId = 1;
  cellTable = new Map<number, Cell>();
}

export function getNextId(): number {
  return nextCellId++;
}

export function lookupCell(id: number) {
  return cellTable.get(id);
}

// When rendering HTML server-side, all of the notebook cells are executed so
// their output can be placed in the generated HTML. This queue tracks the
// execution promises for each cell.
export let notebookExecuteQueue: Array<Promise<void>> = [];

const codemirrorOptions = {
  lineNumbers: false,
  lineWrapping: true,
  mode: "javascript",
  scrollbarStyle: null,
  theme: "syntax",
  viewportMargin: Infinity,
};

export interface Props {
  code: string;
  outputHTML: string;
  id: number;
}
export interface State { outputHTML: string; }

export class Cell extends Component<Props, State> {
  input: Element;
  output: Element;
  editor: CodeMirror.Editor;

  constructor(props) {
    super(props);
    cellTable.set(this.id, this);
  }

  get id(): number {
    return this.props.id;
  }

  _console: Console;
  get console(): Console {
    if (!this._console) {
      this._console = new Console(this);
    }
    return this._console;
  }

  get code(): string {
    return (this.editor ? this.editor.getValue() : this.props.code).trim();
  }

  shouldComponentUpdate() {
    console.log(" shouldComponentUpdate() ", this.id);
    return false;
  }

  async execute(): Promise<void> {
    lastExecutedCellId = this.id;
    let rval, error;
    try {
      rval = await evalCell(this.code, this.id);
    } catch (e) {
      error = e instanceof Error ? e : new Error(e);
    }
    if (error) {
      this.console.error(error.stack);
    } else if (rval !== undefined) {
      this.console.log(rval);
    }
    // When running tests, rethrow any errors. This ensures that errors
    // occurring during notebook cell evaluation result in test failure.
    if (error && window.navigator.webdriver) {
      throw error;
    }
  }

  componentDidMount() {
    const options = Object.assign({}, codemirrorOptions, {
      mode: "javascript",
      value: this.code,
    });

    // If we're in node, doing server-side rendering, we don't enable
    // CodeMirror as its not clear if it can re-initialize its state after
    // being serialized.
    if (IS_WEB) {
      // tslint:disable:variable-name
      const CodeMirror = require("codemirror");
      require("codemirror/mode/javascript/javascript.js");
      this.input.innerHTML = "" ; // Delete the <pre>
      this.editor = CodeMirror(this.input, options);
      this.editor.on("focus", this.focus.bind(this));
      this.editor.on("blur", this.blur.bind(this));
      this.editor.setOption("extraKeys", {
        "Ctrl-Enter": () =>  { this.update(); return true; },
        "Shift-Enter": () => { this.update(); this.nextCell(); return true; }
      });
    }
    // Execute the cell automatically.
    notebookExecuteQueue.push(this.update());
  }

  async update() {
    this.output.innerHTML = ""; // Clear output.
    const classList = (this.input.parentNode as HTMLElement).classList;
    classList.add("notebook-cell-running");
    try {
      await this.execute();
    } finally {
      classList.add("notebook-cell-updating");
      await delay(100);
      classList.remove("notebook-cell-updating");
      classList.remove("notebook-cell-running");
    }
  }

  async nextCell() {
    const currentId = "cell" + this.id;
    const noteBooks = document.querySelectorAll(".notebook-cell");
    let found = false, i;
    for (i = 0; i < noteBooks.length; i++) {
      if (noteBooks[i].id === currentId) {
        found = true;
        i++;
        break;
      }
    }
    if (!found) {
      return;
    }
    let cell = noteBooks[i];
    if (!cell) {
      // create a new cell
      const newBtn = document.querySelector("#newCell");
      if (!newBtn) {
        return;
      }
      newBtn.click();
      await delay(100);
      cell = document.querySelector(".notebook-cell:last-child");
    }
    // scroll & focus
    // TODO: Scroll with a nice animation
    cell.scrollIntoView();
    document.querySelector("#" + cell.id + " textarea").focus();
  }

  clickedRun() {
    console.log("Run was clicked.");
    this.update();
  }

  blur() {
    this.input.classList.remove("focus");
  }

  focus() {
    this.input.classList.add("focus");
  }

  render() {
    return h("div", {
        "class": "notebook-cell",
        "id": `cell${this.id}`
      },
      h("div", {
        "class": "input",
        "ref": (ref => { this.input = ref; }),
      },
        // This pre is replaced by CodeMirror if users have JavaScript enabled.
        h("pre", { }, this.code)
      ),
      h("div", { "class": "buttons" },
        h("button", {
          "class": "run-button",
          "onClick": this.clickedRun.bind(this),
        }, "Run"),
        // h("button", { "class": "", "onClick": null, }, "Delete"),
      ),
      h("div", {
        "class": "output",
        "dangerouslySetInnerHTML": { __html: this.props.outputHTML },
        "id": "output" + this.id,
        "ref": (ref => { this.output = ref; }),
      }),
    );
  }
}

export interface FixedProps {
  code: string;
}

// FixedCell is for non-executing notebook-lookalikes. Usually will be used for
// non-javascript code samples.
// TODO share more code with Cell.
export class FixedCell extends Component<FixedProps, State> {
  render() {
    // Render as a pre in case people don't have javascript turned on.
    return h("div", { "class": "notebook-cell", },
      h("div", { "class": "input" },
        h("pre", { }, this.props.code.trim()),
      )
    );
  }
}

export class Console {
  constructor(private cell: Cell) { }

  private common(...args) {
    const output = this.cell.output;
    // .toString() will fail if any of the arguments is null or undefined. Using
    // ("" + a) instead.
    let s = args.map((a) => "" + a).join(" ");
    const last = output.lastChild;
    if (last && last.nodeType !== Node.TEXT_NODE) {
      s = "\n" + s;
    }
    const t = document.createTextNode(s + "\n");
    output.appendChild(t);
  }

  log(...args) {
    return this.common(...args);
  }

  warn(...args) {
    return this.common(...args);
  }

  error(...args) {
    return this.common(...args);
  }
}

export async function evalCell(source: string, cellId: number): Promise<any> {
  source = transpile(source);
  source += `\n//# sourceURL=__cell${cellId}__.js`;
  const fn = globalEval(source);
  const g = IS_WEB ? window : global;
  const cell = lookupCell(cellId);
  return await fn(g, importModule, cell.console);
}

// This is to handle asynchronous output situations.
// Try to guess which cell we are executing in by looking at the stack trace.
// If there isn't a cell in there, then default to the lastExecutedCellId.
export function guessCellId(): number {
  const stacktrace = (new Error()).stack.split("\n");
  for (let i = stacktrace.length - 1; i >= 0; --i) {
    const line = stacktrace[i];
    const m = line.match(/__cell(\d+)__/);
    if (m) return Number(m[1]);
  }
  return lastExecutedCellId;
}

export function outputEl(): Element {
  const id = guessCellId();
  const cell = lookupCell(id);
  if (cell) {
    return cell.output;
  } else {
    return null;
  }
}

matplotlib.register(outputEl);

async function importModule(target) {
  // console.log("require", target);
  const m = {
    matplotlib,
    mnist,
    propel,
  }[target];
  if (m) {
    return m;
  }
  throw new Error("Unknown module: " + target);
}
