import { assert, IS_WEB } from "../src/util";
import { enableFirebase } from "./db";
import { drainExecuteQueue } from "./notebook";
import { renderPage, route } from "./website";

assert(IS_WEB);

enableFirebase();

window.addEventListener("load", () => {
  const page = route(document.location.pathname);
  renderPage(page);
  drainExecuteQueue();
});
