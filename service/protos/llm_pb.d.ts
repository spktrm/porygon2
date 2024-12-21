// package: 
// file: llm.proto

import * as jspb from "google-protobuf";

export class LLMState extends jspb.Message {
  clearLegalmaskList(): void;
  getLegalmaskList(): Array<boolean>;
  setLegalmaskList(value: Array<boolean>): void;
  addLegalmask(value: boolean, index?: number): boolean;

  getRequest(): string;
  setRequest(value: string): void;

  getLog(): string;
  setLog(value: string): void;

  getMyteam(): string;
  setMyteam(value: string): void;

  getOppteam(): string;
  setOppteam(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): LLMState.AsObject;
  static toObject(includeInstance: boolean, msg: LLMState): LLMState.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: LLMState, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): LLMState;
  static deserializeBinaryFromReader(message: LLMState, reader: jspb.BinaryReader): LLMState;
}

export namespace LLMState {
  export type AsObject = {
    legalmaskList: Array<boolean>,
    request: string,
    log: string,
    myteam: string,
    oppteam: string,
  }
}

