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

testBrowser(async function notebook_focusNextCell() {
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
  const cellEls = document.getElementsByClassName("notebook-cell");
  assert(cellEls.length >= 2);
  const first = nb.lookupCell(cellEls[0].id);
  const second = nb.lookupCell(cellEls[1].id);

  first.focus();

  assert(cellEls[0].classList.contains("notebook-cell-focus"));
  assert(!cellEls[1].classList.contains("notebook-cell-focus"));

  // Simulate Shift+Enter
  first.editor.triggerOnKeyDown({
    keyCode: 13,  // Enter
    shiftKey: true,
  });

  // TODO This is a race condition. Replace this with something like
  // await rerender();
  await delay(100);

  assert(!cellEls[0].classList.contains("notebook-cell-focus"));
  assert(cellEls[1].classList.contains("notebook-cell-focus"));
});
