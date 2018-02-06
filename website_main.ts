import { assert, IS_WEB } from "./util";
import { renderPage, route } from "./website";

assert(IS_WEB);

window.addEventListener("load", () => {
  const page = route(document.location.pathname);
  renderPage(page);
});
