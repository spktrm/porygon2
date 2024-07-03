// package: rlenv
// file: state.proto

import * as jspb from "google-protobuf";

export class LegalActions extends jspb.Message {
  getMove1(): boolean;
  setMove1(value: boolean): void;

  getMove2(): boolean;
  setMove2(value: boolean): void;

  getMove3(): boolean;
  setMove3(value: boolean): void;

  getMove4(): boolean;
  setMove4(value: boolean): void;

  getSwitch1(): boolean;
  setSwitch1(value: boolean): void;

  getSwitch2(): boolean;
  setSwitch2(value: boolean): void;

  getSwitch3(): boolean;
  setSwitch3(value: boolean): void;

  getSwitch4(): boolean;
  setSwitch4(value: boolean): void;

  getSwitch5(): boolean;
  setSwitch5(value: boolean): void;

  getSwitch6(): boolean;
  setSwitch6(value: boolean): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): LegalActions.AsObject;
  static toObject(includeInstance: boolean, msg: LegalActions): LegalActions.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: LegalActions, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): LegalActions;
  static deserializeBinaryFromReader(message: LegalActions, reader: jspb.BinaryReader): LegalActions;
}

export namespace LegalActions {
  export type AsObject = {
    move1: boolean,
    move2: boolean,
    move3: boolean,
    move4: boolean,
    switch1: boolean,
    switch2: boolean,
    switch3: boolean,
    switch4: boolean,
    switch5: boolean,
    switch6: boolean,
  }
}

export class Info extends jspb.Message {
  getGameid(): number;
  setGameid(value: number): void;

  getDone(): boolean;
  setDone(value: boolean): void;

  getPlayeronereward(): number;
  setPlayeronereward(value: number): void;

  getPlayertworeward(): number;
  setPlayertworeward(value: number): void;

  getPlayerindex(): boolean;
  setPlayerindex(value: boolean): void;

  getTurn(): number;
  setTurn(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Info.AsObject;
  static toObject(includeInstance: boolean, msg: Info): Info.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Info, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Info;
  static deserializeBinaryFromReader(message: Info, reader: jspb.BinaryReader): Info;
}

export namespace Info {
  export type AsObject = {
    gameid: number,
    done: boolean,
    playeronereward: number,
    playertworeward: number,
    playerindex: boolean,
    turn: number,
  }
}

export class State extends jspb.Message {
  hasInfo(): boolean;
  clearInfo(): void;
  getInfo(): Info | undefined;
  setInfo(value?: Info): void;

  hasLegalactions(): boolean;
  clearLegalactions(): void;
  getLegalactions(): LegalActions | undefined;
  setLegalactions(value?: LegalActions): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): State.AsObject;
  static toObject(includeInstance: boolean, msg: State): State.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: State, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): State;
  static deserializeBinaryFromReader(message: State, reader: jspb.BinaryReader): State;
}

export namespace State {
  export type AsObject = {
    info?: Info.AsObject,
    legalactions?: LegalActions.AsObject,
  }
}

