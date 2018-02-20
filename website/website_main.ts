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

  const match = /(?:^#|&)overlay=(.*?)(?:&|$)/.exec(window.location.hash);
  if (match) {
    const image = match[1];
    const origStyle = Object.assign({}, document.body.style);
    document.addEventListener("keydown", e => {
      if (e.key !== "Control") return;
      Object.assign(document.body.style, {
        opacity: .5,
        backgroundImage: `url("/static/${image}")`,
        backgroundRepeat: "no-repeat",
        backgroundPositionX: "center",
        backgroundPositionY: "top",
      });
      e.stopPropagation();
    });
    document.addEventListener("keyup", e => {
      if (e.key !== "Control") return;
      Object.assign(document.body.style, origStyle);
      e.stopPropagation();
    });
  }
});
