import { h, render } from "preact";
import { assert, IS_WEB } from "../src/util";
import { enableFirebase } from "./db";
import { drainExecuteQueue } from "./notebook";
import { Router } from "./website";

assert(IS_WEB);

enableFirebase();

window.addEventListener("load", () => {
  render(h(Router, null), document.body, document.body.children[0]);
  drainExecuteQueue();
});
