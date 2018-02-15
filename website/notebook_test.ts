import { h, render } from "preact";
import { assert, delay, objectsEqual } from "../src/util";
import { testBrowser } from "../tools/tester";
import { enableMock } from "./db";
import * as nb from "./notebook";

testBrowser(function notebook_NotebookRoot() {
  const mdb = enableMock();
  document.body.innerHTML = "";
  const el = h(nb.NotebookRoot, { });
  render(el, document.body);
  console.log("mdb.counts", mdb.counts);
  assert(objectsEqual(mdb.counts, {
    queryLatest: 1,
  }));
  const c = document.body.children[0];
  assert(c.className === "notebook");
});

function createNotebook(): Promise<void> {
  document.body.innerHTML = "";
  return new Promise((resolve) => {
    const el = h(nb.Notebook, {
      nbId: "default",
      onReady: resolve,
    });
    render(el, document.body);
  });
}

function getOutput(cellEl) {
  const output = cellEl.getElementsByClassName("output")[0];
  return output.innerHTML.trim();
}

async function rerender(): Promise<void> {
  // TODO This is a race condition. Replace this with something like
  // await rerender();
  await delay(100);
}

testBrowser(async function notebook_Notebook() {
  const mdb = enableMock();
  await createNotebook();
  assert(objectsEqual(mdb.counts, { getDoc: 1 }));
  // Check that we rendered the blurb.
  const blurbs = document.getElementsByClassName("blurb");
  assert(1 === blurbs.length);
  // Because we aren't logged in, we shouldn't see any clone button.
  const clones = document.getElementsByClassName("clone");
  assert(0 === clones.length);
});

testBrowser(async function notebook_focusNextCell() {
  const mdb = enableMock();
  await createNotebook();

  assert(objectsEqual(mdb.counts, { getDoc: 1 }));
  // Test focusNextCell transitions.
  const cellEls = document.getElementsByClassName("notebook-cell");
  assert(cellEls.length >= 2);
  const first = nb.lookupCell(cellEls[0].id);
  const second = nb.lookupCell(cellEls[1].id);

  first.focus();
  await rerender();

  assert(cellEls[0].classList.contains("notebook-cell-focus"));
  assert(!cellEls[1].classList.contains("notebook-cell-focus"));

  // Simulate Shift+Enter
  first.editor.triggerOnKeyDown({
    keyCode: 13,  // Enter
    shiftKey: true,
  });

  await rerender();

  assert(!cellEls[0].classList.contains("notebook-cell-focus"));
  assert(cellEls[1].classList.contains("notebook-cell-focus"));
});

testBrowser(async function notebook_InsertCell() {
  const mdb = enableMock();
  await createNotebook();
  let cellEls = document.getElementsByClassName("notebook-cell");
  const origNumCells = cellEls.length;

  // First check that the output of the first two cells is there.
  const output0 = getOutput(cellEls[0]);
  const output1 = getOutput(cellEls[1]);
  assert(output0.length > 0);
  assert(output1.length > 0);

  // Grab the insert cell button on the first cell.
  const insertButton = cellEls[0].getElementsByClassName("insert-button")[0];
  insertButton.click(); .
  await rerender();

  // Check that we added a cell element.
  cellEls = document.getElementsByClassName("notebook-cell");
  assert(cellEls.length === origNumCells + 1);

  // Check that we didn't delete any other outputs in the process.
  assert(getOutput(cellEls[0]) === output0);
  assert(getOutput(cellEls[1]).length === 0);
  assert(getOutput(cellEls[2]) === output1);
  // TODO assert that the first and third cells weren't re-executed.
  // Currently cellEls[2] is executed after insert.
});
