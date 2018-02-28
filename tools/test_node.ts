import "../src/disk_experiment_test";
import "./test_isomorphic";

// Only on Node/TF should we run the binding_test.
import { backend } from "../src/api";
if (backend === "tf") {
  import("../src/binding_test");
}
