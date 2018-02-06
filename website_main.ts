import { enableFirebase } from "./db";
import { assert, IS_WEB } from "./util";
import { renderPage, route } from "./website";

assert(IS_WEB);

enableFirebase();

window.addEventListener("load", () => {
  const page = route(document.location.pathname);
  renderPage(page);
});
