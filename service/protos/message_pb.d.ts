// package: rlenv
// file: message.proto

import * as jspb from "google-protobuf";
import * as action_pb from "./action_pb";

export class WorkerMessage extends jspb.Message {
  getWorkerindex(): number;
  setWorkerindex(value: number): void;

  getMessagetype(): WorkerMessageTypeMap[keyof WorkerMessageTypeMap];
  setMessagetype(value: WorkerMessageTypeMap[keyof WorkerMessageTypeMap]): void;

  hasAction(): boolean;
  clearAction(): void;
  getAction(): action_pb.Action | undefined;
  setAction(value?: action_pb.Action): void;

  getGameid(): number;
  setGameid(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): WorkerMessage.AsObject;
  static toObject(includeInstance: boolean, msg: WorkerMessage): WorkerMessage.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: WorkerMessage, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): WorkerMessage;
  static deserializeBinaryFromReader(message: WorkerMessage, reader: jspb.BinaryReader): WorkerMessage;
}

export namespace WorkerMessage {
  export type AsObject = {
    workerindex: number,
    messagetype: WorkerMessageTypeMap[keyof WorkerMessageTypeMap],
    action?: action_pb.Action.AsObject,
    gameid: number,
  }
}

export interface WorkerMessageTypeMap {
  START: 0;
  ACTION: 1;
}

export const WorkerMessageType: WorkerMessageTypeMap;

