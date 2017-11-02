import * as util from './util';
import {assert, assertEqual} from './util';

function testCounterMap() {
  let m = new util.CounterMap();
  assertEqual(m.get(0), 0);
  m.inc(0);
  m.inc(0);
  assertEqual(m.get(0), 2);
  assertEqual(m.get(1), 0);
  m.inc(1);
  assertEqual(m.get(1), 1);
  m.dec(0);
  m.inc(1);
  assertEqual(m.get(0), 1);
  assertEqual(m.get(1), 2);
  let k = m.keys();
  console.log(k)
}


testCounterMap();
