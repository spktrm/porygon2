// package: rlenv
// file: message.proto

import * as jspb from "google-protobuf";
import * as action_pb from "./action_pb";

export class WorkerMessage extends jspb.Message {
  getWorkerIndex(): number;
  setWorkerIndex(value: number): void;

  getMessageType(): WorkerMessageTypeEnumMap[keyof WorkerMessageTypeEnumMap];
  setMessageType(value: WorkerMessageTypeEnumMap[keyof WorkerMessageTypeEnumMap]): void;

  hasAction(): boolean;
  clearAction(): void;
  getAction(): action_pb.Action | undefined;
  setAction(value?: action_pb.Action): void;

  getGameId(): number;
  setGameId(value: number): void;

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
    workerIndex: number,
    messageType: WorkerMessageTypeEnumMap[keyof WorkerMessageTypeEnumMap],
    action?: action_pb.Action.AsObject,
    gameId: number,
  }
}

export interface WorkerMessageTypeEnumMap {
  WORKER_MESSAGE_TYPE_ENUM___UNSPECIFIED: 0;
  WORKER_MESSAGE_TYPE_ENUM__START: 1;
  WORKER_MESSAGE_TYPE_ENUM__ACTION: 2;
}

export const WorkerMessageTypeEnum: WorkerMessageTypeEnumMap;

