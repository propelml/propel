import { Server } from "ws";
import { OutputHandler } from "../src/output_handler";
import { setOutputHandler } from "../src/util";
import { RPC, WebSocketRPC } from "./rpc";

interface BufferedOutput {
  type: keyof OutputHandler;
  data: any;
}

export class BufferingOutputHandler implements OutputHandler {
  // Currently we just store the last 20 outputs. This is to become more
  // advanced when we allow updating existing plots.
  private maxBuffer = 10;
  private buffer = new Array<BufferedOutput>();
  // RPC outputs that we are forwarding to.
  remotes = new Array<RPC>();

  // TODO: OutputHandler should probably have a different signature, because
  // we keep repeating these four methods with all four having pretty much
  // the same implementation. See also sandbox.ts and nb.tsx.
  imshow(data: any): void {
    this.push({ type: "imshow", data });
  }
  plot(data: any): void {
    this.push({ type: "plot", data });
  }
  print(data: any): void {
    this.push({ type: "print", data });
  }
  downloadProgress(data: any): void {
    this.push({ type: "downloadProgress", data });
  }

  private send(rpc: RPC, o: BufferedOutput) {
    rpc.call(o.type, null, o.data).catch(() => {});
  }

  private push(o: BufferedOutput) {
    this.buffer = this.buffer.slice(-(this.maxBuffer - 1)).concat(o);
    for (const rpc of this.remotes) {
      this.send(rpc, o);
    }
  }

  connect(rpc: RPC) {
    this.remotes.push(rpc);
    for (const o of this.buffer) {
      this.send(rpc, o);
    }
  }

  disconnect(rpc: RPC) {
    const index = this.remotes.indexOf(rpc);
    if (index < 0) return;
    this.remotes.splice(index, 1);
  }
}

const rpcHandlers = {
  async runCell() {
    return "Not supported";
  }
};

let server;

export async function startServer() {
  if (server) {
    // Only start one server.
    return;
  }

  // Buffer outputs so we can display them in the nodebook when the user
  // connects.
  const outputHandler = new BufferingOutputHandler();
  setOutputHandler(outputHandler);

  server = new Server({ host: "127.0.0.1", port: 12345 });

  server.on("connection", async socket => {
    // TODO-IMPORTANT: check host/origin.
    const rpc = new WebSocketRPC(socket);
    rpc.start(rpcHandlers);
    outputHandler.connect(rpc);
    await new Promise(res => {
      socket.onclose = res;
      socket.onerror = res;
    });
    outputHandler.disconnect(rpc);
    rpc.stop();
  });

  await new Promise(res => server.once("listening", res));

  const port = server.address().port;
  console.log(`websocket listening at ws://127.0.0.1:${port}/`);
  console.log(`View output at http://propelml.org/notebook?port=${port}`);
}
