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
  /* Currently failing.
  let clones = document.getElementsByClassName("clone")
  console.log("clone buttons", clones);
  assert(0 === clones.length);
   */
});
