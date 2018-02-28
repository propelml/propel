import "./test_isomorphic";

import "../src/disk_experiment_test";

// Only on Node/TF should we run the binding_test.
import { backend } from "../src/api";
if (backend === "tf") {
  import("../src/binding_test");
}
