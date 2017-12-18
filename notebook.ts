import * as CodeMirror from "codemirror";
import "codemirror/mode/javascript/javascript.js";

import * as propel from "./api";
import * as matplotlib from "./matplotlib";
import * as mnist from "./mnist";
import { assert } from "./util";

const cellTable = new Map<number, Cell>(); // Maps id to Cell.
let cellsElement = null;
const _log = console.log;
const _error = console.error;
// If you use the eval function indirectly, by invoking it via a reference
// other than eval, as of ECMAScript 5 it works in the global scope rather than
// the local scope. This means, for instance, that function declarations create
// global functions, and that the code being evaluated doesn't have access to
// local variables within the scope where it's being called.
const globalEval = eval;

let lastExecutedCellId = null;

export function evalCell(source, cellId) {
  source += `\n//# sourceURL=__cell${cellId}__.js`;
  globalEval(source);
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

window["require"] = function require(target) {
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
};

window["exports"] = {};

// Override console.log so that it goes to the correct cell output.
console.log = (...args) => {
  const output = outputEl();
  // messy
  let s = args.map((a) => a.toString()).join(" ");
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
  let s = args.map((a) => a.toString()).join(" ");
  const last = output.lastChild;
  if (last && last.nodeType !== Node.TEXT_NODE) {
    s = "\n" + s;
  }
  const t = document.createElement("b");
  t.innerText = s + "\n";
  output.appendChild(t);
  _error(...args);
};

// TODO(ry) Replace this with typescript compiler at some point.
function parseImportLine(line) {
  const words = line.split(/\s/);
  if (words.shift() !== "import") return null;
  const imports = [];
  let state = "curly-open";
  while (true) {
    let word = words.shift();
    switch (state) {
      case null:
        return null;
      case "curly-open":
        if (word !== "{") return null;
        state = "imported";
        break;
      case "imported":
        if (word === "}") {
          state = "from";
        } else {
          if (word[word.length - 1] !== ",") {
            state = "curly-close";
          }
          imports.push(word.replace(/,$/, ""));
        }
        break;
      case "curly-close":
        if (word !== "}") return null;
        state = "from";
        break;
      case "from":
        if (word !== "from") return null;
        state = "modname";
        break;
      case "modname":
        const q = word[0];
        word = word.replace(/;$/, "");
        if (word[word.length - 1] !== q) return null;
        // success
        return {
          imports,
          modname: word,
        };

      default:
        assert(false);
    }
  }
}

// TODO Yes, extremely hacky.
function transpile(source: string): string {
  const lines = source.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const r = parseImportLine(line);
    // Replace line with require() calls, if it's an import line.
    if (r) {
      lines[i] = "";
      for (const m of r.imports) {
        lines[i] += `${m} = require(${r.modname}).${m}; `;
      }
    }
  }
  return lines.join("\n");
}

class Cell {
  isLoad: boolean;
  output: HTMLElement;
  editor: CodeMirror.Editor;
  runButton: HTMLElement;
  id: number;
  static nextId = 1;

  constructor(source?: string) {
    this.id = Cell.nextId++;
    cellTable.set(this.id, this);

    this.editor = CodeMirror(cellsElement, {
      lineNumbers: false,
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
    cellsElement.appendChild(runButton);

    this.output = document.createElement("div");
    this.output.className = "output";
    this.output.id = `output${this.id}`;
    cellsElement.appendChild(this.output);
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

  execute(done?: () => void) {
    lastExecutedCellId = this.id;
    const source = transpile(this.editor.getValue());
    let rval;
    try {
      rval = evalCell(source, this.id);
    } catch (e) {
      console.error(e.stack);
    }
    if (rval) { console.log(rval); }
    if (done) { done(); }
  }
}

function newCellClick() {
  _log("Button click");
  const cell = new Cell();
  window.scrollBy(0, 500); // scroll down.
  cell.focus();
}

window.onload = () => {
  cellsElement = document.getElementById("cells");
  document.getElementById("newCell").onclick = newCellClick;

  // Pre-existing cells are stored as <script type=notebook> elements.
  // These script tags are promptly removed from the DOM and
  // replaced with CodeMirror-handled textareas containing
  // the source code. We use <script type=notebook> in order to get
  // proper syntax highlighting in editors.

  const cells = [];
  const scripts = Array.from(document.scripts).filter(
    (s) => s.type === "notebook");
  for (const s of scripts) {
    s.remove();
    cells.push(new Cell(s.innerText));
  }

  let execCounter = 0;
  function execNext() {
    if (execCounter < cells.length) {
      const cell = cells[execCounter++];
      cell.execute(execNext);
    }
  }
  execNext();
};
