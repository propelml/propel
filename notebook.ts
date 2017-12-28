import * as CodeMirror from "codemirror";
import "codemirror/mode/htmlmixed/htmlmixed.js";
import "codemirror/mode/javascript/javascript.js";

import * as propel from "./api";
import * as matplotlib from "./matplotlib";
import * as mnist from "./mnist";
import { transpile } from "./nb_transpiler";
import { assert } from "./utils";

const cellTable = new Map<number, Cell>(); // Maps id to Cell.
const cellsElement = null;
const _log = console.log;
const _error = console.error;
// If you use the eval function indirectly, by invoking it via a reference
// other than eval, as of ECMAScript 5 it works in the global scope rather than
// the local scope. This means, for instance, that function declarations create
// global functions, and that the code being evaluated doesn't have access to
// local variables within the scope where it's being called.
const globalEval = eval;

let lastExecutedCellId = null;

export async function evalCell(source, cellId): Promise<any> {
  source = transpile(source);
  source += `\n//# sourceURL=__cell${cellId}__.js`;
  const fn = globalEval(source);
  return await fn(window, importModule);
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

export function outputEl(): HTMLElement {
  const id = guessCellId();
  const cell = cellTable.get(id);
  return cell.output;
}

export function outputId(): string {
  const id = guessCellId();
  const cell = cellTable.get(id);
  return cell.output.id;
}

async function importModule(target) {
  // _log("require", target);
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

// Override console.log so that it goes to the correct cell output.
console.log = (...args) => {
  const output = outputEl();
  // messy

  // .toString() will fail if any of the arguments is null or undefined. Using
  // ("" + a) instead.
  let s = args.map((a) => "" + a).join(" ");

  const last = output.lastChild;
  if (last && last.nodeType !== Node.TEXT_NODE) {
    s = "\n" + s;
  }
  const t = document.createTextNode(s + "\n");
  output.appendChild(t);
  _log(...args);
};

// Override console.error so that it goes to the correct cell output.
console.error = (...args) => {
  const output = outputEl();
  // messy

  // .toString() will fail if any of the arguments is null or undefined. Using
  // ("" + a) instead.
  let s = args.map((a) => "" + a).join(" ");

  const last = output.lastChild;
  if (last && last.nodeType !== Node.TEXT_NODE) {
    s = "\n" + s;
  }
  const t = document.createElement("b");
  t.innerText = s + "\n";
  output.appendChild(t);
  _error(...args);
};

class Cell {
  isLoad: boolean;
  output: HTMLElement;
  editor: CodeMirror.Editor;
  runButton: HTMLElement;
  id: number;
  static nextId = 1;

  constructor(source: string, parentDiv: HTMLElement) {
    this.id = Cell.nextId++;
    cellTable.set(this.id, this);
    parentDiv.classList.add("notebook-cell");

    this.editor = CodeMirror(parentDiv, {
      lineNumbers: false,
      mode: "javascript",
      value: source ? source.trim() : "",
      viewportMargin: Infinity,
    });
    this.editor.setOption("extraKeys", {
      "Shift-Enter": this.update.bind(this),
    });

    const runButton = document.createElement("button");
    this.runButton = runButton;
    runButton.innerText = "Run";
    runButton.className = "run-button";
    runButton.onclick = this.update.bind(this);
    parentDiv.appendChild(runButton);

    this.output = document.createElement("div");
    this.output.className = "output";
    this.output.id = `output${this.id}`;
    parentDiv.appendChild(this.output);
  }

  focus() {
    this.editor.focus();
  }

  update(cm) {
    _log("update");
    this.output.innerText = ""; // Clear output.
    this.execute();
    return false;
  }

  appendOutput(svg) {
    this.output.appendChild(svg);
  }

  outputId(): string {
    return this.output.id;
  }

  async execute() {
    lastExecutedCellId = this.id;
    const source = this.editor.getValue();
    let rval;
    try {
      rval = await evalCell(source, this.id);
    } catch (e) {
      console.error(e.stack);
    }
    if (rval !== undefined ) {
      console.log(rval);
    }
  }
}

function newCellClick() {
  _log("Button click");
  const cellsDiv = document.getElementById("cells");
  const parentDiv = document.createElement("div");
  cellsDiv.appendChild(parentDiv);
  const cell = new Cell("", parentDiv);
  window.scrollBy(0, 500); // scroll down.
  cell.focus();
}

window.onload = async() => {
  const newCell = document.getElementById("newCell");
  if (newCell) newCell.onclick = newCellClick;

  // Use CodeMirror to syntax highlight read-only <pre> elements.
  for (const p of Array.from(document.getElementsByTagName("pre"))) {
    _log("pre", p);
    const code = p.innerText;
    p.innerHTML = "";
    CodeMirror(p, {
      lineNumbers: false,
      mode: p.getAttribute("lang") || "javascript",
      readOnly: true,
      value: code,
      viewportMargin: Infinity,
    });
  }

  // Pre-existing cells are stored as <script type=notebook> elements.
  // These script tags are promptly removed from the DOM and
  // replaced with CodeMirror-handled textareas containing
  // the source code. We use <script type=notebook> in order to get
  // proper syntax highlighting in editors.

  const cells = [];
  const scripts = Array.from(document.scripts).filter(
    s => s.type === "notebook");
  for (const s of scripts) {
    const code = s.innerText;
    const parentDiv = document.createElement("div");
    replaceWith(s, parentDiv);
    cells.push(new Cell(code, parentDiv));
  }

  for (const cell of cells) {
    await cell.execute();
  }
};

// Replaces oldElement with newElement at the same place
// in the DOM tree.
function replaceWith(oldElement: HTMLElement,
                     newElement: HTMLElement): void {
  oldElement.parentNode.replaceChild(newElement, oldElement);
}
