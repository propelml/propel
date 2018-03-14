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

// Propel Notebooks. nb = notebook.
// Note that this is rendered and executed server-side using JSDOM and then is
// re-rendered client-side. The Propel code in the cells are executed
// server-side so the results can be displayed even if javascript is disabled.

import { escape } from "he";
import { Component, h } from "preact";
import { OutputHandlerDOM } from "../src/output_handler";
import { assert, IS_WEB, URL } from "../src/util";
import { CMComponent } from "./codemirror";
import { Avatar, GlobalHeader, Loading, normalizeCode, UserMenu }
  from "./common";
import * as db from "./db";
import { SandboxRPC } from "./sandbox_rpc";

const cellTable = new Map<string, Cell>(); // Maps id to Cell.
const nextCellId = 1;

// Maps cellId to outputHTML. See website/main.ts.
const prerenderedOutputs = new Map<string, string>();

export function registerPrerenderedOutput(output) {
  const cellId = output.id.replace("output", "");
  prerenderedOutputs.set(cellId, output.innerHTML);
}

// An anonymous notebook doc for when users aren't logged in
const anonDoc = {
  anonymous: true,
  cellDocs: db.blankCells([ "// New Notebook. Insert code here." ]),
  created: new Date(),
  owner: {
    displayName: "Anonymous",
    photoURL: "/static/img/anon_profile.png?",
    uid: "",
  },
  title: "Anonymous Notebook. Changes will not be saved.",
  updated: new Date(),
};

// Given a cell's id, which can either be an integer or
// a string of the form "cell5" (where 5 is the id), look up
// the component in the global table.
export function lookupCell(id: string) {
  const cell = cellTable.get(id);
  if (!cell) console.log("lookupCell failed for", id);
  return cell;
}

function createIframe(): Window {
  const base = new URL("/sandbox", window.document.baseURI).href;
  const html = `<!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <base href="${escape(base, { isAttributeValue: true })}">
        <script async type="text/javascript" src="/sandbox.js">
        </script>
      </head>
      <body>
      </body>
    </html>`;

  const iframe = document.createElement("iframe");
  iframe.setAttribute("sandbox", "allow-scripts");
  iframe.setAttribute("srcdoc", `${html}`);
  iframe.style.display = "none";
  document.body.appendChild(iframe);

  return iframe.contentWindow;
}

function createSandbox(context: Window): SandboxRPC {
  const sandbox = new SandboxRPC(context, {
    console(cellId: string, ...args: string[]): void {
      const cell = lookupCell(cellId);
      cell.console(...args);
    },

    plot(cellId: string, data: any): void {
      const cell = lookupCell(cellId);
      cell.plot(data);
    },

    imshow(cellId: string, data: any): void {
      const cell = lookupCell(cellId);
      cell.imshow(data);
    }
  });

  return sandbox;
}

let sandbox_: SandboxRPC = null;

export function initSandbox(context?: Window): void {
  if (context == null) {
    context = createIframe();
  }
  sandbox_ = createSandbox(context);
}

function sandbox(): SandboxRPC {
  if (sandbox_ === null) {
    initSandbox();
  }
  return sandbox_;
}

// Convenience function to create Notebook JSX element.
export function cell(code: string, props: CellProps = {}): JSX.Element {
  props.doc = {
    id: "StaticCell" + hash(code),
    input: code,
    outputHTML: null,
  };
  return h(Cell, props);
}

// When rendering HTML server-side, all of the notebook cells are executed so
// their output can be placed in the generated HTML. This queue tracks the
// execution promises for each cell.
const cellExecuteQueue: Cell[] = [];

export async function drainExecuteQueue() {
  while (cellExecuteQueue.length > 0) {
    const cell = cellExecuteQueue.shift();
    await cell.run();
  }
}

export interface CellProps {
  doc?: db.CellDoc;
  onRun?: (cellDoc: db.CellDoc) => void;
  // If onDelete or onInsertCell is null, it hides the button.
  onDelete?: () => void;
  onInsertCell?: () => void;
}

export interface CellState {
  outputHTML: string;
  focused: boolean;
}

export class Cell extends Component<CellProps, CellState> {
  parentDiv: HTMLElement;
  output: Element;
  cm: CMComponent;

  constructor(props) {
    super(props);
    if (prerenderedOutputs.has(this.id)) {
      this.state.outputHTML = prerenderedOutputs.get(this.id);
    } else {
      this.state.outputHTML = props.doc.outputHTML;
    }
  }

  componentWillMount() {
    console.log("register cell", this.id);
    cellTable.set(this.id, this);
    if (!this.state.outputHTML) {
      console.log("cellExecuteQueue", this.id);
      cellExecuteQueue.push(this);
    }
  }

  setValue(code: string): void {
    this.cm.setValue(code);
  }

  get id(): string {
    return this.props.doc.id;
  }

  // TODO rename this getter to cell.input.
  get code(): string {
    return normalizeCode(this.cm ? this.cm.code
                                 : this.props.doc.input);
  }

  saveOutputState() {
    this.props.doc.outputHTML = this.output.innerHTML;
    this.setState({ outputHTML: this.output.innerHTML });
  }

  console(...args: string[]) {
    const output = this.output;
    const last = output.lastChild;
    let s = (last && last.nodeType !== Node.TEXT_NODE) ? "\n" : "";
    s += args.join(" ") + "\n";
    const el = document.createTextNode(s);
    output.appendChild(el);
    this.saveOutputState();
  }

  plot(data) {
    const o = new OutputHandlerDOM(this.output);
    o.plot(data);
    this.saveOutputState();
  }

  imshow(data) {
    const o = new OutputHandlerDOM(this.output);
    o.imshow(data);
    this.saveOutputState();
  }

  clearOutput() {
    this.setState({ outputHTML: "" });
  }

  focusNext() {
    const cellsEl = this.parentDiv.parentElement;
    // Don't focus next if we're in the docs.
    if (cellsEl.className !== "cells") return;

    const nbCells = cellsEl.getElementsByClassName("notebook-cell");
    assert(nbCells.length > 0);

    // NodeListOf<Element> doesn't have indexOf. We loop instead.
    for (let i = 0; i < nbCells.length - 1; i++) {
      if (nbCells[i] === this.parentDiv) {
        const nextCellElement = nbCells[i + 1];
        const next = lookupCell(nextCellElement.id);
        assert(next != null);
        next.focus();
        return;
      }
    }
  }

  focus() {
    this.cm.focus();
    this.parentDiv.scrollIntoView();
  }

  // This method executes the code in the cell, and updates the output div with
  // the result. The onRun callback is called if provided.
  async run() {
    this.clearOutput();
    const classList = this.parentDiv.classList;
    classList.add("notebook-cell-running");

    await sandbox().call("runCell", this.code, this.id);

    classList.add("notebook-cell-updating");
    // Use setTimeout instead of delay here to not block
    // onRun and drainExecuteQueue.
    setTimeout(() => {
      classList.remove("notebook-cell-updating");
      classList.remove("notebook-cell-running");
    }, 100);

    const cellDoc = Object.assign(this.props.doc, {
      outputHTML: this.state.outputHTML
    });
    if (this.props.onRun) this.props.onRun(cellDoc);
  }

  clickedDelete() {
    if (this.props.onDelete) this.props.onDelete();
  }

  clickedInsertCell() {
    if (this.props.onInsertCell) this.props.onInsertCell();
  }

  render() {
    const runButton = h("button", {
      "class": "run-button",
      "onClick": this.run.bind(this),
    }, "");

    let deleteButton = null;
    if (this.props.onDelete) {
      deleteButton = h("button", {
          "class": "delete-button",
          "onClick": this.clickedDelete.bind(this),
      }, "");
    }

    let insertButton = null;
    if (this.props.onInsertCell) {
      insertButton = h("button", {
          "class": "insert-button",
          "onClick": this.clickedInsertCell.bind(this),
      }, "");
    }

    // If supplied outputHTML, use that in the output div.
    const outputDiv = h("div", {
      "class": "output",
      "dangerouslySetInnerHTML": {
        __html: this.state.outputHTML,
      },
      "id": "output" + this.id,
      "ref": ref => { this.output = ref; },
    });

    let className = "notebook-cell";
    if (this.state.focused) {
      className += " notebook-cell-focus";
    }

    return h("div", {
        "class": className,
        "id": this.id,
        "ref": (ref => { this.parentDiv = ref as HTMLElement; }),
      },
      h("div", {
        "class": "input",
      },
        deleteButton,
        runButton,
        h(CMComponent, {
          code: this.code,
          onCtrlEnter: () => this.run(),
          onFocusChange: (focused) => this.setState({ focused }),
          onShiftEnter: async() => {
            this.cm.blur();
            await this.run();
            this.focusNext();
          },
          ref: (ref => { this.cm = ref as CMComponent; }),
        }),
      ),
      h("div", { "class": "output-container" },
        outputDiv,
        insertButton,
      )
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
        h("pre", { }, normalizeCode(this.props.code)),
      )
    );
  }
}

export interface NotebookRootProps {
  userInfo?: db.UserInfo;
  nbId?: string;
}

export interface NotebookRootState {
  nbId?: string;
}

export class NotebookRoot extends Component<NotebookRootProps,
                                            NotebookRootState> {
  constructor(props) {
    super(props);

    let nbId;
    if (this.props.nbId) {
      nbId = this.props.nbId;
    } else {
      const matches = window.location.search.match(/nbId=(\w+)/);
      nbId = matches ? matches[1] : null;
    }

    this.state = { nbId };
  }

  render() {
    let body;
    if (this.state.nbId) {
      body = h(Notebook, {
        nbId: this.state.nbId,
        userInfo: this.props.userInfo,
      });
    } else {
      body = h(MostRecent, null);
    }

    return h("div", { "class": "notebook" },
      h(GlobalHeader, {
        subtitle: "Notebook",
        subtitleLink: "/notebook",
      }, h(UserMenu, { userInfo: this.props.userInfo })),
      body,
    );
  }
}

export interface MostRecentState {
  latest: db.NbInfo[];
}

function nbUrl(nbId: string): string {
  // Careful, S3 is finicy about what URLs it serves. So
  // /notebook?nbId=blah  will get redirect to /notebook/
  // because it is a directory with an index.html in it.
  const u = window.location.origin + "/notebook/?nbId=" + nbId;
  return u;
}

export class MostRecent extends Component<any, MostRecentState> {
  async componentWillMount() {
    // Only query firebase when in the browser.
    // This is to avoiding calling into firebase during static HTML generation.
    if (IS_WEB) {
      const latest = await db.active.queryLatest();
      this.setState({latest});
    }
  }

  async onCreate() {
    const nbId = await db.active.create();
    // Redirect to new notebook.
    window.location.href = nbUrl(nbId);
  }

  render() {
    if (!this.state.latest) {
      return h(Loading, null);
    }
    const notebookList = this.state.latest.map((info) => {
      let snippit = "";
      if (info.doc.cellDocs != null) {
        snippit = info.doc.cellDocs
          .map(({input}) => normalizeCode(input))
          .join("\n")
          .slice(0, 100);
      }
      const href = nbUrl(info.nbId);
      return h("a", { href },
        h("li", null,
          h("div", { "class": "code-snippit" }, snippit),
          notebookBlurb(info.doc, false),
        )
      );
    });
    return h("div", { "class": "most-recent" },
      h("div", {"class": "most-recent-header"},
        h("div", {"class": "most-recent-header-title"},
          h("h2", null, "Recently Updated"),
        ),
        h("div", {"class": "most-recent-header-cta"},
          h("button", { "class": "create-notebook",
                        "onClick": () => this.onCreate(),
          }, "+ New Notebook"),
        ),
      ),
      h("ol", null, ...notebookList),
    );
  }
}

export interface NotebookProps {
  nbId: string;
  onReady?: () => void;
  userInfo?: db.UserInfo;  // Info about the currently logged in user.
}

export interface NotebookState {
  doc?: db.NotebookDoc;
  errorMsg?: string;
  isCloningInProgress: boolean;
  editingTitle: boolean;
}

// This defines the Notebook cells component.
export class Notebook extends Component<NotebookProps, NotebookState> {
  private titleInput: HTMLInputElement;

  constructor(props) {
    super(props);
    this.state = {
      doc: null,
      errorMsg: null,
      isCloningInProgress: false,
      editingTitle: false,
    };
  }

  async componentWillMount() {
    try {
      if (this.props.nbId === "anonymous") {
        this.setState({ doc: anonDoc });
      } else {
        const doc = await db.active.getDoc(this.props.nbId);
        if (doc.cellDocs == null) {
          doc.cellDocs = db.blankCells(doc.cells);
          delete doc.cells;
        }
        this.setState({ doc });
      }
    } catch (e) {
      this.setState({ errorMsg: e.message });
    }
  }

  // TODO Rename to Save to database.
  private async update(doc: db.NotebookDoc): Promise<void> {
    this.setState({ doc });
    if (doc.anonymous) return; // don't persist anonymous notebooks
    try {
      await db.active.updateDoc(this.props.nbId, doc);
    } catch (e) {
      this.setState({ errorMsg: e.message });
    }
  }

  async onSave(updatedCode: string, i: number) {
    this.update(this.state.doc);
  }

  async onRun(cellDoc: db.CellDoc, i: number) {
    const doc = this.state.doc;
    doc.cellDocs[i] = cellDoc;
    this.setState({ doc });
  }

  async onSaveTitle() {
    this.setState({ editingTitle: false });
    this.state.doc.title = this.titleInput.value;
    this.update(this.state.doc);
  }

  async onDelete(i) {
    const doc = this.state.doc;
    doc.cellDocs.splice(i, 1);
    this.update(this.state.doc);
  }

  async onInsertCell(i) {
    const doc = this.state.doc;
    doc.cellDocs.splice(i + 1, 0, {
      id: "ViaInsert" + Math.random().toFixed(5).slice(2),
      input: "",
      outputHTML: "",
    });
    this.update(this.state.doc);
  }

  async onClone() {
    if (this.state.isCloningInProgress) {
      return;
    }
    this.setState({ isCloningInProgress: true });
    const clonedId = await db.active.clone(this.state.doc);
    this.setState({ isCloningInProgress: false });
    // Redirect to new cloned notebook.
    window.location.href = nbUrl(clonedId);
  }

  get loading() {
    return this.state.doc == null;
  }

  renderCells(doc): JSX.Element {
    assert(doc.cellDocs.length > 0);
    return h("div", { "class": "cells" }, doc.cellDocs.map((cellDoc, i) => {
      // Only display the delete button if there is more than one
      // cell.
      const onDelete = doc.cellDocs.length > 1 ? () => this.onDelete(i)
                                                : null;
      return h(Cell, {
        doc: cellDoc,
        onDelete,
        onInsertCell: () => this.onInsertCell(i),
        onRun: () => this.onRun(cellDoc, i),
      });
    }));
  }

  render() {
    let body;

    if (this.state.errorMsg) {
      body = [
        h("h1", null, "Error"),
        h("b", null, this.state.errorMsg),
      ];

    } else if (this.state.doc == null) {
      body = [
        h(Loading, null)
      ];

    } else {
      const doc = this.state.doc;

      const titleEdit = h("div", { class: "title" },
        h("input", {
          "class": "title-input",
          "ref": ref => { this.titleInput = ref as HTMLInputElement; },
          "value": doc.title,
        }),
        h("button", {
          "class": "save-title green-button",
          "onClick": () => this.onSaveTitle()
        }, "Save"),
        h("button", {
          "class": "cancel-edit-title",
          "onClick": () => this.setState({ editingTitle: false })
        }, "Cancel")
      );

      const editButton = h("button", {
        class: "edit-title",
        onClick: () => this.setState({ editingTitle: true })
      }, "Edit");

      const titleDisplay = h("div", { class: "title" }, [
        h("h2", {
          class: doc.title && doc.title.length ? "" : "untitled",
          value: doc.title
        }, doc.title || "Untitled Notebook"),
        db.ownsDoc(this.props.userInfo, doc) ? editButton : null
      ]);

      const title = this.state.editingTitle ? titleEdit : titleDisplay;

      let cloneButton, saveButton;
      if (db.ownsDoc(this.props.userInfo, doc)) {
        saveButton = h("button", {
          "class": "save-notebook",
          "onClick": () => this.onSave(),
        }, "Save");
      } else {
        cloneButton = h("button", {
          "class": "green-button",
          "onClick": () => this.onClone(),
        }, "Clone");
      }

      body = [
        h("div", { "class": "notebook-container" },
          h("header", null, notebookBlurb(doc), title, cloneButton, saveButton),
          this.renderCells(doc),
        ),
      ];
    }

    return h("div", null, ...body);
  }

  async componentDidUpdate() {
    await drainExecuteQueue();

    // We've rendered the Notebook with either a document or errorMsg.
    if (this.state.doc || this.state.errorMsg) {
      if (this.props.onReady) this.props.onReady();
    }
  }
}

function notebookBlurb(doc: db.NotebookDoc, showDates = true): JSX.Element {
  const dates = !showDates ? [] : [
    h("div", { "class": "date-created" },
      h("p", { "class": "created" }, `Created ${fmtDate(doc.created)}.`),
    ),
    h("div", { "class": "date-updated" },
      h("p", { "class": "updated" }, `Updated ${fmtDate(doc.updated)}.`),
    ),
  ];
  return h("div", { "class": "blurb" }, null, [
    h("div", { "class": "blurb-avatar" },
      h(Avatar, { userInfo: doc.owner }),
    ),
    h("div", { "class": "blurb-name" },
      h("p", { "class": "displayName" }, doc.owner.displayName),
    ),
    ...dates
  ]);
}

function fmtDate(d: Date): string {
  return d.toISOString();
}

function hash(s: string): number {
    let hash = 0;
    if (s.length === 0) {
        return hash;
    }
    for (let i = 0; i < s.length; i++) {
        const char = s.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32bit integer
    }
    return hash;
}
