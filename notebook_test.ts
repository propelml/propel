import { h, render } from "preact";
import { enableMock } from "./db";
import * as nb from "./notebook";
import { testBrowser } from "./test";
import { assert, objectsEqual } from "./util";

testBrowser(function notebook_NotebookRoot() {
  const mdb = enableMock();
  const el = h(nb.NotebookRoot, { });
  render(el, document.body);
  assert(objectsEqual(mdb.counts, {
    queryLatest: 1,
    subscribeAuthChange: 1,
  }));
});

testBrowser(function notebook_Notebook() {
  const mdb = enableMock();
  const el = h(nb.Notebook, { nbId: "abc" });
  render(el, document.body);
  console.log(mdb.counts);
  assert(objectsEqual(mdb.counts, {
    getDoc: 1,
  }));
});
