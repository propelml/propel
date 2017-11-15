import * as hljs from "highlight.js";

import propel from "./propel";
import matplotlib from "./matplotlib";

let cellsElement = document.createElement('div');
let _log = console.log;
// If you use the eval function indirectly, by invoking it via a reference
// other than eval, as of ECMAScript 5 it works in the global scope rather than
// the local scope. This means, for instance, that function declarations create
// global functions, and that the code being evaluated doesn't have access to
// local variables within the scope where it's being called.
let globalEval = eval;

let _REPLAppendOutput = null;
let _REPLOutputId = null;

export function appendOutput(svg) {
  _REPLAppendOutput(svg);
}

export function outputId(): string {
  return _REPLOutputId();
}

window["define"] = function define(x) {
  _log("define", x);
}

window["require"] = function require(target) {
  //_log("require", target);
  let m = {
    "propel": propel,
    "matplotlib": matplotlib
  }[target];
  if (m) {
    return m;
  }
  throw new Error("Unknown module: " + target);
}

window["exports"] = {};

// TODO Yes, extremely hacky.
function transpile(source: string): string {
  return source.replace(/import (\w+) from ("[^"]+"|'[^']+')/g, (m, p1, p2) => {
    let x = `${p1} = require(${p2})`;
    //_log("replace", p1, p2, x);
    return x;
  });
}

class Cell {
  isLoad: boolean;
  script: HTMLScriptElement;
  output: HTMLElement;
  input: HTMLElement;
  id: number;
  static nextId = 1;

  constructor(script?: HTMLScriptElement) {
    this.id = Cell.nextId++;
    this.script = script;
    //_log("create cell");

    let input = document.createElement("pre");
    input.style.padding = "10px";
    this.input = input;
    input.onkeydown = this.onkeydown.bind(this);
    cellsElement.appendChild(input);

    if (this.script.src) {
      input.innerText = `<script src="${this.script.src}"/>`;
      input.className = "language-html";
      this.isLoad = true;
    } else {
      input.innerText = this.script.innerText.replace(/^\s+/, '');
      input.className = "language-javascript";
      input.contentEditable = "true";
      this.isLoad = false;
    }
    hljs.highlightBlock(input);

    this.output = document.createElement("div");
    this.output.id = `output${this.id}`;
    this.output.style.padding = "0 10px";
    this.output.style["font-family"] = "monospace";
    this.output.style["white-space"] = "pre-wrap";
    cellsElement.appendChild(this.output);
  }

  onkeydown(e) {
    if (e.which == 13 && e.shiftKey) {
      _log("return shift");
      hljs.highlightBlock(this.input); // rehighlight.
      this.output.innerText = ""; // Clear output.
      this.execute();
      return false;
    }
  }

  log(...args) {
    // messy
    let s = args.map(a => a.toString()).join(" ");
    let last = this.output.lastChild; 
    if (last && last.nodeType != Node.TEXT_NODE) {
      s = "\n" + s;
    }
    var t = document.createTextNode(s + "\n");
    this.output.appendChild(t);
    _log(...args);
  }

  error(...args) {
    // messy
    let s = args.map(a => a.toString()).join(" ");
    let last = this.output.lastChild; 
    if (last && last.nodeType != Node.TEXT_NODE) {
      s = "\n" + s;
    }
    var t = document.createElement("b");
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
    this.script.remove();
    this.script['type'] = "text/javascript";
    if (this.isLoad) {
      _log("execute", this.script.src);
      document.head.appendChild(this.script);
      this.script.onload = done;
    } else {
      let source = this.script.innerText = this.input.innerText;

      let jsSource = transpile(source);

      let rval = undefined;
      try {
        rval = globalEval(jsSource);
      } catch(e) {
        this.error(e.stack);
      }
      if (rval) this.log(rval);
      if (done) done();
    }
  }
}

function createNewCellButton() {
  let b = document.createElement("button");
  b.innerText = "New Cell";
  b.style.margin = "50px 10px";
  document.body.appendChild(b);
  b.onclick = () => {
    _log("Button click");
    let s = document.createElement('script');
    s['type'] = 'repl';
    new Cell(s);
    window.scrollBy(0, 500); // scroll down.
  };
}

window.onload = () => {
  document.body.style.margin = "0";
  document.body.appendChild(cellsElement);

  createNewCellButton();

  let cells = [];
  for (let i = 0; i < document.scripts.length; ++i) {
    let s = document.scripts[i];
    if (s.type != "repl") continue;
    cells.push(new Cell(s));
  }

  let execCounter = 0;
  function execNext() {
    if (execCounter < cells.length) {
      let cell = cells[execCounter++];
      cell.execute(execNext);
    }
  }
  execNext();
};

