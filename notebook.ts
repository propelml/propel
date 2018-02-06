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

// Propel Notebooks.
// Note that this is rendered and executed server-side using JSDOM and then is
// re-rendered client-side. The Propel code in the cells are executed
// server-side so the results can be displayed even if javascript is disabled.

// tslint:disable:no-reference
/// <reference path="node_modules/@types/codemirror/index.d.ts" />

import { Component, h } from "preact";
import * as propel from "./api";
import * as db from "./db";
import * as matplotlib from "./matplotlib";
import * as mnist from "./mnist";
import { transpile } from "./nb_transpiler";
import { assert, delay, IS_WEB } from "./util";
import { div, GlobalHeader } from "./website";

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

// Convenience function to create Notebook JSX element.
export function notebook(code: string, props: CellProps = {}): JSX.Element {
  props.id = props.id || getNextId();
  // console.log("create notebook", props.id);
  props.code = code;
  return h(Cell, props);
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

export interface CellProps {
  code?: string;
  id?: number;
  onRun?: (code: null | string) => void;
  // If onDelete or onInsertCell is null, it hides the button.
  onDelete?: () => void;
  onInsertCell?: () => void;
}
export interface CellState { }

export class Cell extends Component<CellProps, CellState> {
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

  clearOutput() {
    this.output.innerHTML = "";
  }

  componentWillReceiveProps(nextProps: CellProps) {
    const nextCode = nextProps.code.trim();
    if (nextCode !== this.code) {
      this.editor.setValue(nextCode);
      this.clearOutput();
    }
  }

  // Never update the component, because CodeMirror has complex state.
  // Code updates are done in componentWillReceiveProps.
  shouldComponentUpdate() {
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
    // Execute the cell automatically.
    this.clearOutput();
    notebookExecuteQueue.push(this.execute());

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
        "Ctrl-Enter": () =>  { this.run(); return true; },
        "Shift-Enter": () => { this.run(); this.nextCell(); return true; }
      });
    }
  }

  async update() {
    this.clearOutput();
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

  nextCell() {
    // TODO
  }

  run() {
    this.update();
    if (this.props.onRun) this.props.onRun(this.code);
  }

  clickedDelete() {
    console.log("Delete was clicked.");
    if (this.props.onDelete) this.props.onDelete();
  }

  clickedInsertCell() {
    console.log("NewCell was clicked.");
    if (this.props.onInsertCell) this.props.onInsertCell();
  }

  blur() {
    this.input.classList.remove("focus");
  }

  focus() {
    this.input.classList.add("focus");
  }

  render() {
    const buttons = [
      h("button", {
        "class": "run-button",
        "onClick": this.run.bind(this),
      }, "Run")
    ];

    if (this.props.onDelete) {
      buttons.unshift(h("button", {
          "class": "delete-button",
          "onClick": this.clickedDelete.bind(this),
      }, "Delete"));
    }

    if (this.props.onInsertCell) {
      buttons.unshift(h("button", {
          "class": "delete-button",
          "onClick": this.clickedInsertCell.bind(this),
      }, "Insert Cell"));
    }

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
      h("div", { "class": "buttons" }, ...buttons),
      h("div", {
        "class": "output",
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
export class FixedCell extends Component<FixedProps, CellState> {
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
  assert(cell != null);
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

interface NotebookProps {
  nbId?: string;
}

interface NotebookState {
  loadingAuth: boolean;
  userInfo?: db.UserInfo;
  doc?: db.NotebookDoc;
  nbId: string;
  errorMsg?: string;
}

const exampleNbId = "2f0lhH2kP1ySYtY4sK9y";

// This defines the Notebook page which loads Cells from Firebase.
export class Notebook extends Component<NotebookProps, NotebookState> {
  handle: db.Handle;

  constructor(props) {
    super(props);

    // Extract the nbId from props or query string.
    let nbId;
    if (this.props.nbId) {
      nbId = this.props.nbId;
    } else {
      const matches = document.location.search.match(/nbId=(\w+)/);
      if (matches) {
        nbId = matches[1];
      } else {
        nbId = exampleNbId;
      }
    }

    this.state = {
      loadingAuth: true,
      userInfo: null,
      doc: null,
      nbId,
    };
  }

  onError(errorMsg: string) {
    this.setState({ errorMsg });
  }

  async onRun(updatedCode, i) {
    if (this.handle) {
      const doc = this.state.doc;
      doc.cells[i] = updatedCode;
      this.setState({ doc });
      this.handle.update(doc.cells);
    }
  }

  async onDelete(i) {
    const doc = this.state.doc;
    doc.cells.splice(i, 1);
    this.setState({ doc });
    this.handle.update(doc.cells);
  }

  async onInsertCell(i) {
    const doc = this.state.doc;
    doc.cells.splice(i + 1, 0, "");
    this.setState({ doc });
    this.handle.update(doc.cells);
  }

  loading() {
    return this.state.loadingAuth || this.state.doc == null;
  }

  removeAuthListener: () => void;
  componentDidMount() {
    let hasSetStateFromDoc = false;
    if (IS_WEB) {
      this.handle = new db.Handle({
        nbId: this.state.nbId,
        onAuthStateChange: userInfo =>
            this.setState({ loadingAuth: false, userInfo }),
        onDocChange: doc => {
          // We only take the very first change....
          // Hacky.
          if (!hasSetStateFromDoc) { this.setState({ doc }),
          hasSetStateFromDoc = true;
          }
          console.log("Doc change from database.", doc);
        },
        onError: this.onError.bind(this),
      });
      this.handle.subscribe();
    }
  }

  componentWillUnmount() {
    if (this.handle) this.handle.dispose();
  }

  signIn() {
    console.log("Click signIn");
    if (this.handle && !this.state.userInfo && !this.state.loadingAuth) {
      this.handle.signIn();
      this.setState({ loadingAuth: true });
    }
  }

  async clone() {
    console.log("Click clone");
    const clonedId = await this.handle.clone(this.state.doc);
    // Redirect to new cloned notebook.
    window.location.href = "/notebook?nbId=" + clonedId;
  }

  signOut() {
    console.log("Click signOut");
    this.handle.signOut();
    this.setState({ loadingAuth: true });
  }

  stateToCells(): JSX.Element[] {
    return this.state.doc.cells.map((code, i) => {
      return notebook(code, {
        // TODO(rld) i + 1 is because nextCellId starts at 1. So the statically
        // generated version starts at 1. In order to consistantly update those
        // cells, we use the same ids here. This brittle and should be fixed.
        id: i + 1,
        onRun: (updatedCode) => { this.onRun(updatedCode, i); },
        onDelete: () => { this.onDelete(i); },
        onInsertCell: () => { this.onInsertCell(i); },
      });
    });
  }

  render() {
    let menuItems;
    let body;

    if (this.state.errorMsg) {
      menuItems = [];
      body = [
        h("h1", null, "Error"),
        h("b", null, this.state.errorMsg),
      ];

    } else if (this.loading()) {
      menuItems = [];
      body = [h("b", null, "loading")];

    } else {
      const nbCells = this.stateToCells();

      if (this.state.userInfo) {
        menuItems = [
          h(Avatar, { userInfo: this.state.userInfo }),
          h("button", {
            "onclick": this.signOut.bind(this),
          }, "Sign out")
        ];

        // If we don't own the doc, show a clone button.
        menuItems.push(h("button", {
          "class": "clone",
          "onclick": this.clone.bind(this),
        }, "Clone"));

      } else {
        menuItems = [
          h("button", {
            "class": "clone",
            "onclick": this.signIn.bind(this),
          }, "Sign in to Clone")
        ];
      }

      body = [
        h("header", null,
          h(Avatar, { userInfo: this.state.doc.owner }),
          h("b", null,
            ` Notebook created
              by ${this.state.doc.owner.displayName}
              on ${fmtDate(this.state.doc.created)}.
          `),
        ),
        div("cells", ...nbCells),
        div("container nb-footer", null),
      ];
    }

    return div("notebook",
      h(GlobalHeader, null, ...menuItems),
      ...body,
    );
  }
}

const Avatar = (props: { size?: number, userInfo: db.UserInfo }) => {
  const size = props.size || 50;
  return h("img", {
    src: props.userInfo.photoURL + "&size=" + size,
  });
};

function fmtDate(d: Date): string {
  return d.toISOString();
}
