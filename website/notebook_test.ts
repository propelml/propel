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

  // Check that we rendered the title.
  const title = document.querySelectorAll("div.title > h2");
  assert(1 === title.length);
  assert("Sample Notebook" === title[0].innerHTML);

  // Because we aren't logged in, we shouldn't see an edit button for the title.
  const editButtons = document.getElementsByClassName("edit-title");
  assert(0 === editButtons.length);

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

testBrowser(async function notebook_NotebookLoggedIn() {
  resetPage();

  await new Promise((resolve) => {
    const el = h(nb.Notebook, {
      nbId: "default",
      onReady: resolve,
      userInfo: {
        displayName: "owner",
        photoURL: "https://avatars1.githubusercontent.com/u/80?v=4",
        uid: "abc",
      },
    });
    render(el, document.body);
  });

  // Check that we rendered the title.
  const title = document.querySelectorAll("div.title > h2");
  assert(1 === title.length);
  assert("Sample Notebook" === title[0].innerHTML);

  /* TODO Test needs Firebase auth
  // Because we are logged in, we should see an edit button for the title.
  const editButton = document.getElementsByClassName("edit-title");
  assert(1 === editButton.length);
  */

  // The edit button hasn't been clicked yet, so we shouldn't see the
  // title-input.
  const titleInput = document.getElementsByClassName("title-input");
  assert(0 === titleInput.length);

  /* TODO The follow test has a race condition. Disabling for now.
  editButton[0].click();

  // The edit button has been clicked, so we should see the title-input.
  titleInput = document.getElementsByClassName("title-input");
  assert(null !== titleInput);
  assert("Sample Notebook" === titleInput[0].value);

  titleInput[0].value = "New Title";
  document.getElementsByClassName("edit-title")[0].click();

  title = document.querySelectorAll("div.title > h2");
  assert(1 === title.length);
  assert("New Title" === title[0].innerHTML);
  */

});
