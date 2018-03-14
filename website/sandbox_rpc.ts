/*!
   Copyright 2018 Propel http://propel.site/.  All rights reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

import { createResolvable, nodeRequire, IS_NODE, Resolvable } from "../src/util";

const WebSocket = IS_NODE ? nodeRequire("ws") : window.WebSocket;

export type RpcHandler = (...args: any[]) => any;
export type RpcHandlers = { [name: string]: RpcHandler };

export interface HandshakeMessage {
  type: "syn" | "ack";
}

export interface CallMessage {
  type: "call";
  id: string;
  handler: string;
  args: any[];
}

export interface ReturnMessage {
  type: "return";
  id: string;
  result?: any;
  exception?: any;
}

export type Message = HandshakeMessage | CallMessage | ReturnMessage;

export abstract class SandboxRPCBase {
  // TODO: better solution for filtering messages intended for other frames.
  private unique = Math.floor(Math.random() * 1 << 30).toString(16);
  private counter = 0;
  private ready = createResolvable();
  private returnHandlers = new Map<string, Resolvable<any>>();

  protected abstract send(message: Message);

  constructor(private handlers: RpcHandlers) {
  }

  async call(handler: string, ...args: any[]): Promise<any> {
    await this.ready;

    const id = `${this.unique}_${this.counter++}`;
    const message: CallMessage = {
      type: "call",
      id,
      handler,
      args
    };

    const resolver = createResolvable<any>();
    this.returnHandlers.set(id, resolver);

    try {
      this.send(message);
      return await resolver;
    } finally {
      this.returnHandlers.delete(id);
    }
  }

  protected onMessage(message: Message): void {
    const { type } = message;
    switch (type) {
      case "syn":
      case "ack":
        this.onHandshake(message as HandshakeMessage);
        break;
      case "call":
        this.onCall(message as CallMessage);
        break;
      case "return":
        this.onReturn(message as ReturnMessage);
        break;
    }
  }

  protected sendHandshake() {
    this.send({ type: "syn" });
  }

  private onHandshake(message: HandshakeMessage) {
    const {type} = message;
    if (type === "syn") {
      this.send({ type: "ack" });
    }
    this.ready.resolve();
  }

  private async onCall(message: CallMessage) {
    const {id, handler, args} = message;
    const ret: ReturnMessage = {
      type: "return",
      id
    };
    try {
      const result = await this.handlers[handler](...args);
      this.send({ result, ...ret });
    } catch (exception) {
      if (exception instanceof Error) {
        // Convert to a normal object.
        const { message, stack } = exception;
        exception = { message, stack, __error__: true };
      }
      this.send({ exception, ...ret });
    }
  }

  private onReturn(message: ReturnMessage) {
    const id = message.id;
    const resolver = this.returnHandlers.get(id);
    if (resolver === undefined) {
        return; // Not for us.
    }
    if (message.hasOwnProperty("exception")) {
      let { exception } = message;
      if (exception.__error__) {
        // Convert to Error object.
        exception = Object.assign(new Error(exception.message),
                                  { stack: exception.stack });
      }
      resolver.reject(exception);
    } else {
      resolver.resolve(message.result);
    }
  }
}

export class SandboxRPC extends SandboxRPCBase {
  constructor(private remote: Window, handlers: RpcHandlers) {
    super(handlers);
    // TODO: remove event listener when remote window disappears.
    window.addEventListener("message", event => this.onMessage(event.data));
    this.sendHandshake();
  }

  protected send(message: Message) {
    this.remote.postMessage(message, "*");
  }
}

export class SandboxRPCWebSocket extends SandboxRPCBase {
  constructor(private remote: WebSocket, handlers: RpcHandlers) {
    super(handlers);
    remote.addEventListener("message", event => this.onMessage(JSON.parse(event.data)));
    if (remote.readyState === WebSocket.CONNECTING) {
      remote.addEventListener("open", () => this.sendHandshake());
    } else {
      this.sendHandshake();
    }
  }

  protected send(message: Message) {
    this.remote.send(JSON.stringify(message));
  }
}
