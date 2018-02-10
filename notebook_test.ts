import { h, render } from "preact";
import { enableMock } from "./db";
import * as nb from "./notebook";
import { testBrowser } from "./test";
import { assert, objectsEqual } from "./util";

testBrowser(function notebook_NotebookRoot() {
  const mdb = enableMock();
  document.body.innerHTML = "";
  const el = h(nb.NotebookRoot, { });
  render(el, document.body);
  assert(objectsEqual(mdb.counts, {
    queryLatest: 1,
    subscribeAuthChange: 1,
  }));
  const c = document.body.children[0];
  assert(c.className === "notebook");
});

testBrowser(async function notebook_Notebook() {
  const mdb = enableMock();
  document.body.innerHTML = "";

  const readyPromise = new Promise((resolve) => {
    const el = h(nb.Notebook, {
      nbId: "default",
      onReady: resolve,
    });
    render(el, document.body);
  });

  await readyPromise;

  console.log(mdb.counts);
  assert(objectsEqual(mdb.counts, { getDoc: 1 }));

  // Check that we rendered the blurb.
  const blurbs = document.getElementsByClassName("blurb");
  assert(1 === blurbs.length);

  // Because we aren't logged in, we shouldn't see any clone button.
  const clones = document.getElementsByClassName("clone");
  assert(0 === clones.length);
});

testBrowser(async function cell_focusNextCell() {
  const mdb = enableMock();
  document.body.innerHTML = "";

  const readyPromise = new Promise((resolve) => {
    const el = h(nb.Notebook, {
      nbId: "default",
      onReady: resolve,
    });
    render(el, document.body);
  });

  await readyPromise;
  assert(objectsEqual(mdb.counts, { getDoc: 1 }));
  // Test focusNextCell transitions.
  const cells = document.getElementsByClassName("notebook-cell");
  assert(0 < cells.length);
  const insertedId = Number(cells[0].id.replace("cell", ""));
  const first = nb.lookupCell(insertedId);
  const second = nb.lookupCell(insertedId + 1);

  first.focus();
  assert(first.input.classList.contains("focus"));
  assert(!second.input.classList.contains("focus"));
  first.focusNextCell();
  assert(!first.input.classList.contains("focus"));
  assert(second.input.classList.contains("focus"));
});
