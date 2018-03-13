import { h, render, rerender } from "preact";
import { assert, objectsEqual } from "../src/util";
import { testBrowser } from "../tools/tester";
import * as db from "./db";
import * as nb from "./nb";

testBrowser(function notebook_NotebookRoot() {
  const mdb = db.enableMock();
  resetPage();
  const el = h(nb.NotebookRoot, { });
  render(el, document.body);
  assert(objectsEqual(mdb.counts, {
    queryLatest: 1,
  }));
  const c = document.body.getElementsByTagName("div")[0];
  assert(c.className === "notebook");
});

testBrowser(async function notebook_Notebook() {
  const mdb = db.enableMock();
  await renderAnonNotebook();
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
  const mdb = db.enableMock();
  await renderAnonNotebook();
  assert(objectsEqual(mdb.counts, { getDoc: 1 }));
  // Test focusNextCell transitions.
  const cellEls = document.querySelectorAll(".notebook-cell");
  assert(cellEls.length >= 2);
  const first = nb.lookupCell(cellEls[0].id);
  assert(first != null);
  const second = nb.lookupCell(cellEls[1].id);
  assert(second != null);
  first.focus();
  assert(cellEls[0].classList.contains("notebook-cell-focus"));
  assert(!cellEls[1].classList.contains("notebook-cell-focus"));
  /* FIXME Flaky....
  // Simulate Shift+Enter
  first.editor['triggerOnKeyDown']({ keyCode: 13, shiftKey: true });
  await flush();
  cellEls = document.querySelectorAll(".notebook-cell");
  assert(!cellEls[0].classList.contains("notebook-cell-focus"));
  assert(cellEls[1].classList.contains("notebook-cell-focus"));
  */
});

testBrowser(async function notebook_titleEdit() {
  const mdb = db.enableMock();
  mdb.signIn();
  await renderOwnerNotebook();
  assert(db.defaultDoc.title.length > 1);
  // Check that we rendered the title.
  let title: HTMLElement = document.querySelector(".title > h2");
  assert(db.defaultDoc.title === title.innerText);
  // Because we are logged in, we should see an edit button for the title.
  const editButton: HTMLButtonElement = document.querySelector(".edit-title");
  assert(editButton != null);
  // The edit button hasn't been clicked yet, so we shouldn't see the
  // title-input.
  let titleInput: HTMLInputElement = document.querySelector(".title-input");
  assert(titleInput == null);
  // Click the edit button.
  editButton.click();
  await flush();
  // The edit button has been clicked, so we should see the title-input.
  titleInput = document.querySelector(".title-input");
  assert(null !== titleInput);
  assert(titleInput.value === db.defaultDoc.title);
  // The save button should be shown.
  const saveTitle: HTMLButtonElement = document.querySelector(".save-title");
  assert(saveTitle != null);
  // Edit the title.
  titleInput.value = "New Title";
  // Before the save the db counts look like:
  assert(objectsEqual(mdb.counts, { signIn: 1, getDoc: 1 }));
  // Click the save button.
  saveTitle.click();
  await flush();
  // Check the database saw an updateDoc.
  assert(objectsEqual(mdb.counts, {
    getDoc: 1,
    signIn: 1,
    updateDoc: 1,
  }));
  // Check the title got updated.
  title = document.querySelector(".title > h2");
  assert(title != null);
  assert("New Title" === title.innerText);
});

testBrowser(async function notebook_deleteLastCell() {
  await renderOwnerNotebook();
  const numCells = db.defaultDoc.cells.length;
  assert(numCells > 2);
  // Should have same number of delete buttons, as we have cells in the
  // default doc.
  const deleteButtons = document.getElementsByClassName("delete-button");
  assert(deleteButtons.length === numCells);
  let cells = document.getElementsByClassName("notebook-cell");
  assert(cells.length === numCells);
  // Now click all but one of the delete buttons.
  for (let i = 0; i < numCells - 1; i++) {
    const deleteButton: HTMLButtonElement =
        document.querySelector(".delete-button");
    deleteButton.click();
    await flush();
  }
  cells = document.getElementsByClassName("notebook-cell");
  assert(cells.length === 1);
  // Now that there is only one cell left, the delete button should
  // not be shown.
  const deleteButton = document.querySelector(".delete-button");
  assert(deleteButton == null);
});

// Call this to ensure that the DOM has been updated after events.
function flush(): Promise<void> {
  rerender();
  return Promise.resolve();
}

function resetPage() {
  nb.destroySandbox();
  document.body.innerHTML = "";
}

function renderOwnerNotebook() {
  resetPage();
  return new Promise((resolve) => {
    const el = h(nb.Notebook, {
      nbId: "default",
      onReady: resolve,
      userInfo: db.defaultOwner, // Owns "default" doc.
    });
    render(el, document.body);
  });
}

function renderAnonNotebook() {
  resetPage();
  return new Promise((resolve) => {
    const el = h(nb.Notebook, {
      nbId: "default",
      onReady: resolve,
    });
    render(el, document.body);
  });
}
