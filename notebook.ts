import * as CodeMirror from "codemirror";
import "codemirror/mode/javascript/javascript.js";

import * as propel from "./api";
import * as matplotlib from "./matplotlib";

import { Context } from "./ts_context";
import { assert } from "./util";

let cellsElement = null;
const _log = console.log;
// If you use the eval function indirectly, by invoking it via a reference
// other than eval, as of ECMAScript 5 it works in the global scope rather than
// the local scope. This means, for instance, that function declarations create
// global functions, and that the code being evaluated doesn't have access to
// local variables within the scope where it's being called.
const globalEval = eval;

let _REPLAppendOutput = null;
let _REPLOutputId = null;

export function appendOutput(svg) {
  _REPLAppendOutput(svg);
}

export function outputId(): string {
  return _REPLOutputId();
}

const ctx = new Context({ builtins: { propel, matplotlib } });

class Cell {
  isLoad: boolean;
  output: HTMLElement;
  editor: CodeMirror.Editor;
  runButton: HTMLElement;
  id: number;
  static nextId = 1;

  constructor(source?: string) {
    this.id = Cell.nextId++;

    this.editor = CodeMirror(cellsElement, {
      lineNumbers: false,
      value: source ? source.trim() : "",
      viewportMargin: Infinity
    });
    this.editor.setOption("extraKeys", {
      "Shift-Enter": this.update.bind(this)
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

  log(...args) {
    // messy
    let s = args.map(a => "" + a).join(" ");
    const last = this.output.lastChild;
    if (last && last.nodeType !== Node.TEXT_NODE) {
      s = "\n" + s;
    }
    const t = document.createTextNode(s + "\n");
    this.output.appendChild(t);
    _log(...args);
  }

  error(...args) {
    // messy
    let s = args.map(a => a.toString()).join(" ");
    const last = this.output.lastChild;
    if (last && last.nodeType !== Node.TEXT_NODE) {
      s = "\n" + s;
    }
    const t = document.createElement("b");
    t.innerText = s + "\n";
    this.output.appendChild(t);
    _log(...args);
  }

  appendOutput(svg) {
    this.output.appendChild(svg);
  }

  outputId(): string {
    return "#" + this.output.id;
  }

  execute(done = null) {
    console.log = this.log.bind(this);
    _REPLAppendOutput = this.appendOutput.bind(this);
    _REPLOutputId = this.outputId.bind(this);

    const source = this.editor.getValue();

    ctx.eval(source).then(({ result, error }) => {
      if (error) this.error(error.stack);
      if (done) done();
    });
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
    s => s.type === "notebook"
  );
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
