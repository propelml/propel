import * as gendoc from "./gendoc";
import { assert } from "./util";

function testBasic() {
  const docs = gendoc.genJSON();
  assert(docs.length > 5);
  assert(docs.map(e => e.name).indexOf("Tensor") >= 0);
  const html = gendoc.toHTML(docs);
  assert(html.length > 0);
}

testBasic();
