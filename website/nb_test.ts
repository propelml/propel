import { h, render, rerender } from "preact";
import { assert, assertEqual, createResolvable } from "../src/util";
import { testBrowser } from "../tools/tester";
import * as db from "./db";
import * as nb from "./nb";

testBrowser(async function notebook_NotebookRoot() {
  const mdb = db.enableMock();
  resetPage();
  const el = h(nb.NotebookRoot, { });
  render(el, document.body);
  await flush();
  assertObjectsEqual(mdb.counts, {
    queryLatest: 1,
  });
  const c = document.body.getElementsByTagName("div")[0];
  assertEqual(c.className, "notebook");
});

testBrowser(async function notebook_Notebook() {
  const mdb = db.enableMock();
  await renderAnonNotebook();
  assertEqual(mdb.counts, { getDoc: 1 });
  // Check that we rendered the title.
  const title = document.querySelectorAll("div.title > h2");
  assertEqual(1, title.length);
  assertEqual("Sample Notebook", title[0].innerHTML);
  // Because we aren't logged in, we shouldn't see an edit button for the title.
  const editButtons = document.getElementsByClassName("edit-title");
  assertEqual(0, editButtons.length);
  // Because we aren't logged in, we shouldn't see any clone button.
  const clones = document.getElementsByClassName("clone");
  assertEqual(0, clones.length);
});

testBrowser(async function notebook_focusNextCell() {
  const mdb = db.enableMock();
  await renderAnonNotebook();
  assertObjectsEqual(mdb.counts, { getDoc: 1 });
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
  assertEqual(db.defaultDoc.title, title.innerText);
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
  assertEqual(titleInput.value, db.defaultDoc.title);
  // The save button should be shown.
  const saveTitle: HTMLButtonElement = document.querySelector(".save-title");
  assert(saveTitle != null);
  // Edit the title.
  titleInput.value = "New Title";
  // Before the save the db counts look like:
  assertObjectsEqual(mdb.counts, { signIn: 1, getDoc: 1 });
  // Click the save button.
  saveTitle.click();
  await flush();
  // Check the database saw an updateDoc.
  assertObjectsEqual(mdb.counts, {
    getDoc: 1,
    signIn: 1,
    updateDoc: 1,
  });
  // Check the title got updated.
  title = document.querySelector(".title > h2");
  assert(title != null);
  assertEqual("New Title", title.innerText);
});

testBrowser(async function notebook_deleteLastCell() {
  await renderOwnerNotebook();
  const numCells = db.defaultDoc.cells.length;
  assert(numCells > 2);
  // Should have same number of delete buttons, as we have cells in the
  // default doc.
  const deleteButtons = document.getElementsByClassName("delete-button");
  assertEqual(deleteButtons.length, numCells);
  let cells = document.getElementsByClassName("notebook-cell");
  assertEqual(cells.length, numCells);
  // Now click all but one of the delete buttons.
  for (let i = 0; i < numCells - 1; i++) {
    const deleteButton: HTMLButtonElement =
        document.querySelector(".delete-button");
    deleteButton.click();
    await flush();
  }
  cells = document.getElementsByClassName("notebook-cell");
  assertEqual(cells.length, 1);
  // Now that there is only one cell left, the delete button should
  // not be shown.
  const deleteButton = document.querySelector(".delete-button");
  assert(deleteButton == null);
});

testBrowser(async function notebook_progressBar() {
  resetPage();

  let notebookRoot: nb.NotebookRoot;
  const promise = createResolvable();
  const el = h(nb.NotebookRoot, {
    nbId: "anonymous",
    onReady: promise.resolve,
    ref: ref => notebookRoot = ref,
  });
  render(el, document.body);
  await promise;

  const notebook = notebookRoot.notebookRef;

  // We need at least two cells for this test (which have cellId 1 and 2).
  notebook.onInsertCell(0);
  notebook.onInsertCell(0);

  const progressBar =
      document.querySelector(".notebook-cell .progress-bar") as HTMLElement;
  assert(progressBar != null);

  // tslint:disable-next-line:ban
  const percent = () => parseFloat(progressBar.style.width);
  const visible = () => progressBar.style.display === "block";

  // Call util.downloadProgress in the notebook sandbox.
  const downloadProgress = async(job, loaded, total, cellId = 1) => {
    const sandbox = nb.sandbox();
    await sandbox.call(
      "runCell",
      `import { downloadProgress } from "test_internals";
       downloadProgress(...${JSON.stringify([job, loaded, total])});`,
      cellId);
    await flush();
  };

  // Should not be visible initially.
  assert(!visible());
  // Start one download job, size not specified yet, will be 10kb.
  await downloadProgress("job1", 0, null);
  assert(visible());
  assertEqual(percent(), 0);
  // Start another, size 30k bytes.
  await downloadProgress("job2", 0, 30e3);
  assert(visible());
  assertEqual(percent(), 0);
  // Make progress on both jobs.
  await downloadProgress("job1", 1e3, 10e3);
  assertEqual(percent(), 100 * 1e3 / 40e3);
  await downloadProgress("job2", 1e3, 30e3);
  assertEqual(percent(), 100 * 2e3 / 40e3);
  await downloadProgress("job2", 15e3, 30e3);
  assertEqual(percent(), 100 * 16e3 / 40e3);
  // Set job1 to 100% from cellId 2.
  await downloadProgress("job1", 10e3, 10e3, 2);
  assertEqual(percent(), 100 * 25e3 / 40e3);
  // Finish job1.
  await downloadProgress("job1", null, null);
  // Since job1 is no longer active, and job2 is half done, the progress bar
  // is now back at 50%.
  // TODO: this is kinda weird.
  assert(visible());
  assertEqual(percent(), 50);
  // Set job2 to 100%.
  await downloadProgress("job2", 30e3, 30e3);
  assert(visible());
  assertEqual(percent(), 100);
  // Remove job2 from cell 2.
  await downloadProgress("job2", null, null, 2);
  assert(!visible());
  assertEqual(percent(), 0);
});

testBrowser(async function notebook_profile() {
  const mdb = db.enableMock();
  await renderProfile("non-existant");
  let avatars = document.querySelectorAll(".avatar");
  assert(avatars.length === 0);
  let notebooks = document.querySelectorAll(".most-recent ol li");
  assert(notebooks.length === 0);
  assertObjectsEqual(mdb.counts, { queryProfile: 1 });

  // Try again with a real uid.
  await renderProfile(db.defaultOwner.uid);
  avatars = document.querySelectorAll(".avatar");
  assert(avatars.length === 1);
  notebooks = document.querySelectorAll(".most-recent ol li");
  assert(notebooks.length === 1);
  assertObjectsEqual(mdb.counts, { queryProfile: 2 });
});

// Call this to ensure that the DOM has been updated after events.
function flush(): Promise<void> {
  rerender();
  return Promise.resolve();
}

function resetPage() {
  nb.resetNotebook();
  document.body.innerHTML = "";
}

function renderProfile(profileUid: string) {
  const promise = createResolvable();
  resetPage();
  const el = h(nb.NotebookRoot, {
    onReady: promise.resolve,
    profileUid,
  });
  render(el, document.body);
  return promise;
}

function renderOwnerNotebook() {
  resetPage();
  return new Promise((resolve) => {
    const el = h(nb.NotebookRoot, {
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
    const el = h(nb.NotebookRoot, {
      nbId: "default",
      onReady: resolve,
    });
    render(el, document.body);
  });
}
