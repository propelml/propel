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

import { escape } from "he";
import { Component, h } from "preact";
import { OutputHandlerDOM } from "../src/output_handler";
import {
  assert,
  delay,
  randomString,
  URL
} from "../src/util";
import { CodeMirrorComponent } from "./codemirror";
import {
  Avatar,
  GlobalHeader,
  Loading,
  normalizeCode,
  UserMenu,
} from "./common";
import * as db from "./db";
import { RPC, WindowRPC } from "./rpc";

const cellTable = new Map<number, Cell>(); // Maps id to Cell.
let nextCellId = 1;

export function resetNotebook() {
  destroySandbox();
  cellTable.clear();
  nextCellId = 1;
}

const prerenderedOutputs = new Map<number, string>();

export function registerPrerenderedOutput(output) {
  const cellId = Number(output.id.replace("output", ""));
  prerenderedOutputs.set(cellId, output.innerHTML);
}

// An anonymous notebook doc for when users aren't logged in
const anonDoc = {
  anonymous: true,
  cells: ["// New Notebook. Insert code here."],
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
export function lookupCell(id: string | number): Cell {
  let numId;
  if (typeof id === "string") {
    numId = Number(id.replace("cell", ""));
  } else {
    numId = id;
  }
  return cellTable.get(numId);
}

export function lookupOutputHandler(id: string | number) {
  const cell = lookupCell(id);
  return cell.getOutputHandler();
}

const rpcHandlers = {
  plot(cellId: number, data: any): any {
    lookupOutputHandler(cellId).plot(data);
  },

  print(cellId: number, data: any): any {
    return lookupOutputHandler(cellId).print(data);
  },

  imshow(cellId: number, data: any): any {
    return lookupOutputHandler(cellId).imshow(data);
  },

  downloadProgress(cellId: number, data: any): void {
    return lookupOutputHandler(cellId).downloadProgress(data);
  }
};

function createIframe(rpcChannelId): HTMLIFrameElement {
  const base = new URL("/sandbox", window.document.baseURI).href;
  const html = `<!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8"/>
        <meta name="rpc-channel-id" content="${escape(rpcChannelId)}"/>
        <base href="${escape(base)}">
        <script async type="text/javascript" src="/sandbox.js">
        </script>
      </head>
      <body>
      </body>
    </html>`;

  const iframe = document.createElement("iframe");
  iframe.setAttribute("sandbox", "allow-scripts");
  iframe.setAttribute("srcdoc", `${html}`);
  // Edge doesn't support "srcdoc", it'll use a data url instead.
  iframe.setAttribute("src", `data:text/html,${html}`);
  iframe.style.display = "none";
  document.body.appendChild(iframe);

  return iframe;
}

let sandboxIframe: HTMLIFrameElement = null;
let sandboxRpc: RPC = null;

function createSandbox(): void {
  const rpcChannelId = randomString();
  sandboxIframe = createIframe(rpcChannelId);
  sandboxRpc = new WindowRPC(sandboxIframe.contentWindow, rpcChannelId);
  sandboxRpc.start(rpcHandlers);
}

function destroySandbox(): void {
  if (!sandboxRpc) return;

  sandboxRpc.stop();
  if (sandboxIframe.parentNode) {
    sandboxIframe.parentNode.removeChild(sandboxIframe);
  }
  sandboxRpc = null;
  sandboxIframe = null;
}

export function sandbox(): RPC {
  if (sandboxRpc === null) {
    createSandbox();
  }
  return sandboxRpc;
}

// Convenience function to create Notebook JSX element.
export function cell(code: string, props: CellProps = {}): JSX.Element {
  props.code = code.trim();
  return <Cell { ...props } />;
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
  code?: string;
  onRun?: (code: null | string) => void;
  // If onDelete or onInsertCell is null, it hides the button.
  onDelete?: () => void;
  onInsertCell?: () => void;
}
export interface CellState { }

export class Cell extends Component<CellProps, CellState> {
  parentDiv: Element;
  input: Element;
  output: Element;
  private outputHandler: OutputHandlerDOM;
  cm: CodeMirrorComponent;
  readonly id: number;
  outputHTML?: string;

  constructor(props) {
    super(props);
    this.id = nextCellId++;
    if (prerenderedOutputs.has(this.id)) {
      this.outputHTML = prerenderedOutputs.get(this.id);
    }
    cellTable.set(this.id, this);
  }

  componentWillMount() {
    if (this.outputHTML == null) {
      cellExecuteQueue.push(this);
    }
  }

  get code(): string {
    return normalizeCode(this.cm ? this.cm.code : this.props.code);
  }

  getOutputHandler() {
    if (!this.outputHandler) {
      this.outputHandler = new OutputHandlerDOM(this.output);
    }
    return this.outputHandler;
  }

  downloadProgress(data) {
    const o = new OutputHandlerDOM(this.output);
    o.downloadProgress(data);
  }

  clearOutput() {
    this.output.innerHTML = "";
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
    this.parentDiv.classList.add("notebook-cell-focus");
    this.parentDiv.scrollIntoView();
  }

  // This method executes the code in the cell, and updates the output div with
  // the result. The onRun callback is called if provided.
  async run() {
    this.clearOutput();
    const classList = (this.input.parentNode as HTMLElement).classList;
    classList.add("notebook-cell-running");

    await sandbox().call("runCell", this.code, this.id);

    classList.add("notebook-cell-updating");
    await delay(100);
    classList.remove("notebook-cell-updating");
    classList.remove("notebook-cell-running");

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

  render() {
    const runButton = (
      <button class="run-button" onClick={ this.run.bind(this) } />
    );

    let deleteButton = null;
    if (this.props.onDelete) {
      deleteButton = (
        <button
          class="delete-button"
          onClick={ this.clickedDelete.bind(this) } />
      );
    }

    let insertButton = null;
    if (this.props.onInsertCell) {
      insertButton = (
        <button
          class="insert-button"
          onClick={ this.clickedInsertCell.bind(this) } />
      );
    }

    // If supplied outputHTML, use that in the output div.
    const outputDivAttr = {
      "class": "output",
      "id": "output" + this.id,
      "ref": (ref => { this.output = ref; }),
    };
    if (this.outputHTML) {
      outputDivAttr["dangerouslySetInnerHTML"] = {
        __html: this.outputHTML,
      };
    }
    const outputDiv = <div { ...outputDivAttr } />;

    const runCellAndFocusNext = () => {
      this.run();
      this.cm.blur();
      this.focusNext();
    };

    return (
      <div
        class="notebook-cell"
        id={ `cell${this.id}` }
        ref={ ref => { this.parentDiv = ref; } } >
        <div
          class="input"
          ref={ ref => { this.input = ref; } } >
          <CodeMirrorComponent
            code={ this.code }
            ref={ ref => { this.cm = ref; } }
            onFocus={ () => {
              this.parentDiv.classList.add("notebook-cell-focus");
            } }
            onBlur={ () => {
              this.parentDiv.classList.remove("notebook-cell-focus");
            } }
            onAltEnter={ runCellAndFocusNext }
            onShiftEnter={ runCellAndFocusNext }
            onCtrlEnter={ () => { this.run(); } }
          />
          { deleteButton }
          { runButton }
        </div>
        <div class="progress-bar" />
        <div class="output-container">
          { outputDiv }
          { insertButton }
        </div>
      </div>
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
    return (
      <div class="notebook-cell">
        <div class="input">
          <pre>{ normalizeCode(this.props.code) }</pre>
        </div>
      </div>
    );
  }
}

export interface NotebookRootProps {
  userInfo?: db.UserInfo; // The current user who is logged in.
  // If nbId is specified, it will be queried, and set in doc.
  nbId?: string;
  // If profileId is specified, it will be queried.
  profileUid?: string;
  // If neither nbId nor profileUid is specified, NotebookRoot will
  // use the current URL's query string to search fo nbId and profile.
  // If those are not found, NotebookRoot will query the most recent.
  onReady: () => void;
}

export interface NotebookRootState {
  // Same as in props, but after checking window.location.
  nbId?: string;
  profileUid?: string;

  // If set a Notebook for this doc will be displayed.
  doc?: db.NotebookDoc;
  // If set the most-recent page will be displayed.
  mostRecent?: db.NbInfo[];
  // If set the profile page will be displayed.
  profileLatest?: db.NbInfo[];

  errorMsg?: string;
}

export class NotebookRoot extends Component<NotebookRootProps,
                                            NotebookRootState> {
  notebookRef?: Notebook; // Hook for testing.

  constructor(props) {
    super(props);
    let nbId;
    if (this.props.nbId) {
      nbId = this.props.nbId;
    } else {
      const matches = window.location.search.match(/nbId=(\w+)/);
      nbId = matches ? matches[1] : null;
    }
    let profileUid;
    if (this.props.profileUid) {
      profileUid = this.props.profileUid;
    } else {
      const matches = window.location.search.match(/profile=(\w+)/);
      profileUid = matches ? matches[1] : null;
    }
    this.state = { nbId, profileUid };
  }

  async componentWillMount() {
    // Here is where we query firebase for all sorts of messages.
    const { nbId, profileUid } = this.state;
    try {
      if (nbId) {
        // nbId specified. Query the the notebook.
        const doc = await (nbId === "anonymous"
                           ? Promise.resolve(anonDoc)
                           : db.active.getDoc(nbId));
        this.setState({ doc });

      } else if (profileUid) {
        // profileUid specified. Query the profile.
        const profileLatest =
          await db.active.queryProfile(profileUid, 100);
        this.setState({ profileLatest });

      } else {
        // Neither specified. Show the most-recent.
        // TODO potentially these two queries can be combined into one.
        const mostRecent = await db.active.queryLatest();
        this.setState({ mostRecent });
      }
    } catch (e) {
      this.setState({ errorMsg: e.message });
    }
  }

  async componentDidUpdate() {
    // Call the onReady callback for testing.
    if (this.state.errorMsg || this.state.mostRecent ||
        this.state.profileLatest || this.state.doc) {
      // Make sure the notebook has fully executed.
      await drainExecuteQueue();
      if (this.props.onReady) this.props.onReady();
    }
  }

  render() {
    let body;
    if (this.state.errorMsg) {
      body = (
        <div class="notification-screen">
          <div class="notebook-container">
            <p class="error-header">Error</p>
            <p>{ this.state.errorMsg }</p>
          </div>
        </div>
      );
    } else if (this.state.profileLatest) {
      body = (
        <Profile
          profileLatest={ this.state.profileLatest }
          userInfo={ this.props.userInfo } />
      );

    } else if (this.state.doc) {
      body = (
        <Notebook
          nbInfo={ { nbId: this.state.nbId, doc: this.state.doc } }
          ref= { ref => this.notebookRef = ref }
          userInfo= { this.props.userInfo } />
      );
    } else if (this.state.mostRecent) {
      body = (
        <MostRecent
          mostRecent={ this.state.mostRecent }
          userInfo={ this.props.userInfo } />
      );

    } else {
      body = <Loading />;
    }

    return (
      <div class="notebook">
        <GlobalHeader subtitle="Notebook" subtitleLink="/notebook" >
          <UserMenu userInfo={ this.props.userInfo } />
        </GlobalHeader>
        { body }
      </div>
    );
  }
}

export interface MostRecentProps {
  mostRecent: db.NbInfo[];
  userInfo?: db.UserInfo;
}

export interface MostRecentState { }

export class MostRecent extends Component<MostRecentProps, MostRecentState> {
  render() {
    let profileLinkEl = null;
    if (this.props.userInfo) {
      // TODO This is ugly - we're reusing most-recent-header just to get a line
      // break between the link to "Your Notebooks" and "Most Recent".
      profileLinkEl = (
        <div class="most-recent-header">
          <h2>{ profileLink(this.props.userInfo, "Your Notebooks") }</h2>
        </div>
      );
    }

    return (
      <div class="most-recent">
        { profileLinkEl }
        <div class="most-recent-header">
          <div class="most-recent-header-title">
            <h2>Recently Updated</h2>
          </div>
          <div class="most-recent-header-cta">
            { newNotebookButton() }
          </div>
        </div>
        <ol>
          { ...notebookList(this.props.mostRecent) }
        </ol>
      </div>
    );
  }
}

function newNotebookButton() {
  return (
    <button
      class="create-notebook"
      onClick={async() => {
        // Redirect to new notebook.
        const nbId = await db.active.create();
        window.location.href = nbUrl(nbId);
      }} >
      + New Notebook
    </button>
  );
}

export interface ProfileProps {
  profileLatest: db.NbInfo[];
  userInfo?: db.UserInfo;
}

export interface ProfileState { }

export class Profile extends Component<ProfileProps, ProfileState> {
  render() {
    if (this.props.profileLatest.length === 0) {
      return <h1>User has no notebooks</h1>;
    }
    const doc = this.props.profileLatest[0].doc;

    // TODO Profile is reusing the most-recent css class, because it's a very
    // similar layout. The CSS class should be renamed something appropriate
    // for both of them, maybe nb-listing.
    return (
      <div class="most-recent">
        <div class="most-recent-header">
          <UserTitle userInfo={ doc.owner } />
          { newNotebookButton() }
        </div>
        <ol>
          {...notebookList(this.props.profileLatest, {
            showDates: false,
            showName: false,
            showTitle: true,
          })}
        </ol>
      </div>
    );
  }
}

export interface NotebookProps {
  nbInfo: db.NbInfo;
  userInfo?: db.UserInfo;  // Info about the currently logged in user.
}

export interface NotebookState {
  doc: db.NotebookDoc;
  isCloningInProgress: boolean;
  editingTitle: boolean;
}

// This defines the Notebook cells component.
export class Notebook extends Component<NotebookProps, NotebookState> {
  private titleInput: HTMLInputElement;

  constructor(props) {
    super(props);
    this.state = {
      doc: this.props.nbInfo.doc,
      isCloningInProgress: false,
      editingTitle: false,
    };
  }

  private async update(doc): Promise<void> {
    this.setState({ doc });
    if (doc.anonymous) return; // don't persist anonymous notebooks
    try {
      await db.active.updateDoc(this.props.nbInfo.nbId, doc);
    } catch (e) {
      // TODO updating the database should be moved out of Notebook because
      // errors are handled in NotebookRoot. Clearly we need Redux or something
      // similar.
      // We should be doing this.setState({ errorMsg: e.message });
      throw Error(e);
    }
  }

  async onRun(updatedCode: string, i: number) {
    const doc = this.state.doc;
    updatedCode = normalizeCode(updatedCode);
    // Save updated code in database if different.
    if (normalizeCode(doc.cells[i]) !== updatedCode) {
      doc.cells[i] = updatedCode;
      this.update(doc);
    }
  }

  async onSaveTitle(doc) {
    this.setState({ editingTitle: false });
    doc.title = this.titleInput.value;
    this.update(doc);
  }

  async onDelete(i) {
    const doc = this.state.doc;
    doc.cells.splice(i, 1);
    this.update(doc);
  }

  async onInsertCell(i) {
    const doc = this.state.doc;
    doc.cells.splice(i + 1, 0, "");
    this.update(doc);
  }

  async onClone() {
    if (this.state.isCloningInProgress) {
      return;
    }
    console.log("Click clone");
    this.setState({ isCloningInProgress: true });
    const clonedId = await db.active.clone(this.state.doc);
    this.setState({ isCloningInProgress: false });
    // Redirect to new cloned notebook.
    window.location.href = nbUrl(clonedId);
  }

  renderCells(doc): JSX.Element {
    const codes = db.getInputCodes(doc);
    return (
      <div class="cells">
        {codes.map((code, i) => {
          // Only display the delete button if there is more than one
          // cell.
          const onDelete = doc.cells.length > 1 ? () => this.onDelete(i)
                                                : null;
          return cell(code, {
            onRun: (updatedCode) => this.onRun(updatedCode, i),
            onDelete,
            onInsertCell: () => this.onInsertCell(i),
          });
        })}
      </div>
    );
  }

  render() {
    const doc = this.state.doc;

    const titleEdit = (
      <div class="title">
        <input
          class="title-input"
          ref={ ref => { this.titleInput = ref as HTMLInputElement; } }
          value={ doc.title } />
        <button
          class="save-title green-button"
          onClick={ () => this.onSaveTitle(doc) } >
          Save
        </button>
        <button
          class="cancel-edit-title"
          onClick={ () => this.setState({ editingTitle: false }) } >
          Cancel
        </button>
      </div>
    );

    const editButton = (
      <button
        class="edit-title"
        onClick={ () => this.setState({ editingTitle: true }) } >
        Edit
      </button>
    );

    const titleDisplay = (
      <div class="title">
        <h2
          class={ doc.title && doc.title.length ? "" : "untitled" }
          value={ doc.title } >
          { docTitle(doc) }
        </h2>
        { db.ownsDoc(this.props.userInfo, doc) ? editButton : null }
      </div>
    );

    const title = this.state.editingTitle ? titleEdit : titleDisplay;

    const cloneButton = this.props.userInfo == null ? "" : (
      <button
        class="green-button"
        onClick={ () => this.onClone() } >
        Clone
      </button>
    );

    return (
      <div class="notebook-container">
        <UserTitle userInfo={ doc.owner } />
        <div class="notebook-header">
          { title }
          { cloneButton }
        </div>
        { this.renderCells(doc) }
      </div>
    );
  }
}

function UserTitle(props) {
  return (
    <div class="most-recent-header-title">
      <Avatar userInfo={ props.userInfo } />
      <h2>{ profileLink(props.userInfo) }</h2>
    </div>
  );
}

function docTitle(doc: db.NotebookDoc): string {
  return doc.title || "Untitled Notebook";
}

function notebookList(notebooks: db.NbInfo[], {
  showName = true,
  showTitle = false,
  showDates = false,
} = {}): JSX.Element[] {
  return notebooks.map(info => {
    const snippit = db.getInputCodes(info.doc).join("\n\n");
    const href = nbUrl(info.nbId);
    return (
      <a href={ href } >
        <li>
            <div class="code-snippit">{ snippit }</div>
            { notebookBlurb(info.doc, { showName, showTitle, showDates }) }
        </li>
      </a>
    );
  });
}

function profileLink(u: db.UserInfo, text: string = null): JSX.Element {
  const href = window.location.origin + "/notebook/?profile=" + u.uid;
  return (
    <a class="profile-link" href={ href } >
      { text ? text : u.displayName }
    </a>
  );
}

function notebookBlurb(doc: db.NotebookDoc, {
  showName = true,
  showTitle = false,
  showDates = false,
} = {}): JSX.Element {
  let body = [];
  if (showDates) {
    body = body.concat([
      <div class="date-created">
        <p class="created">
          Created { fmtDate(doc.created) }
        </p>
      </div>,
      <div class="date-updated">
        <p class="updated">
          Updated { fmtDate(doc.updated) }
        </p>
      </div>
    ]);
  }
  if (showName) {
    body = body.concat([
      <div class="blurb-avatar">
        <Avatar userInfo={ doc.owner } />
      </div>,
      <p class="blurb-name">
        { doc.owner.displayName }
      </p>
    ]);
  }
  if (showTitle) {
    body.push(<p class="blurb-title">{ docTitle(doc) }</p>);
  }
  return <div class="blurb">{ ...body }</div>;
}

function fmtDate(d: Date): string {
  return d.toISOString();
}

function nbUrl(nbId: string): string {
  // Careful, S3 is finicky about what URLs it serves. So
  // /notebook?nbId=blah  will get redirect to /notebook/
  // because it is a directory with an index.html in it.
  const u = window.location.origin + "/notebook/?nbId=" + nbId;
  return u;
}
