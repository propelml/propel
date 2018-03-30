import {
  assertEqual,
  IS_NODE,
  nodeRequire
} from "../src/util";
import { test } from "../tools/tester";
import { RPC, WebSocketRPC } from "./rpc";

type ChannelInfo = { rpc1: RPC; rpc2: RPC; cleanup: () => void };
let makeChannel: () => Promise<ChannelInfo>;

if (IS_NODE) {
  makeChannel = async(): Promise<ChannelInfo> => {
    // tslint:disable-next-line:variable-name
    const WebSocket = nodeRequire("ws");
    const server = new WebSocket.Server({ host: `127.0.0.1`, port: 0 });
    await new Promise(res => server.once("listening", res));
    const port = server.address().port;
    const [socket1, socket2] = await Promise.all([
      new Promise<WebSocket>(res => {
        const s = new WebSocket(`ws://127.0.0.1:${port}`);
        s.once("open", () => res(s));
      }),
      new Promise<WebSocket>(res => {
        server.once("connection", s => res(s as WebSocket));
      })
    ]);
    const rpc1 = new WebSocketRPC(socket1);
    const rpc2 = new WebSocketRPC(socket2);
    const cleanup = () => {
      server.close();
      socket1.close();
      socket2.close();
    };
    return { rpc1, rpc2, cleanup };
  };

  test(async function rpc_return() {
    const { rpc1, rpc2, cleanup } = await makeChannel();
    rpc1.start({
      getNumber() {
        return 3.14;
      },
      getString() {
        return "hello";
      },
      getObject() {
        return { i: "am an object" };
      },
      getArray() {
        return [1, 2, 3];
      }
    });
    rpc2.start({});
    assertEqual((await rpc2.call("getNumber")), 3.14);
    assertEqual((await rpc2.call("getString")), "hello");
    assertEqual(await rpc2.call("getObject"), { i: "am an object" });
    assertEqual(await rpc2.call("getArray"), [1, 2, 3]);
    rpc1.stop();
    rpc2.stop();
    cleanup();
  });

  test(async function rpc_recursiveCalls() {
    const { rpc1, rpc2, cleanup } = await makeChannel();
    rpc1.start({
      async addOne(val: number) {
        val += 1;
        if (val < 42) {
          val = await rpc1.call("addTwo", val);
        }
        return val;
      }
    });
    rpc2.start({
      async addTwo(val: number) {
        val += 2;
        if (val < 42) {
          val = await rpc2.call("addOne", val);
        }
        return val;
      }
    });
    assertEqual((await rpc1.call("addTwo", 0)), 42);
    assertEqual((await rpc2.call("addOne", 0)), 42);
    rpc1.stop();
    rpc2.stop();
    cleanup();
  });

} else {
  // TODO: how to test this on the web?
}
