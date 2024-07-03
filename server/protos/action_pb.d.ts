// package: rlenv
// file: action.proto

import * as jspb from "google-protobuf";

export class Action extends jspb.Message {
  getGameid(): number;
  setGameid(value: number): void;

  getPlayerindex(): boolean;
  setPlayerindex(value: boolean): void;

  getIndex(): number;
  setIndex(value: number): void;

  getText(): string;
  setText(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Action.AsObject;
  static toObject(includeInstance: boolean, msg: Action): Action.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Action, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Action;
  static deserializeBinaryFromReader(message: Action, reader: jspb.BinaryReader): Action;
}

export namespace Action {
  export type AsObject = {
    gameid: number,
    playerindex: boolean,
    index: number,
    text: string,
  }
}

