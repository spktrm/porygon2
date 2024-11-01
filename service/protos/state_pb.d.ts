// package: rlenv
// file: state.proto

import * as jspb from "google-protobuf";
import * as history_pb from "./history_pb";
import * as enums_pb from "./enums_pb";

export class Info extends jspb.Message {
  getGameid(): number;
  setGameid(value: number): void;

  getDone(): boolean;
  setDone(value: boolean): void;

  getWinreward(): number;
  setWinreward(value: number): void;

  getHpreward(): number;
  setHpreward(value: number): void;

  getPlayerindex(): boolean;
  setPlayerindex(value: boolean): void;

  getTurn(): number;
  setTurn(value: number): void;

  getTurnssinceswitch(): number;
  setTurnssinceswitch(value: number): void;

  getHeuristicaction(): number;
  setHeuristicaction(value: number): void;

  getLastaction(): number;
  setLastaction(value: number): void;

  getLastmove(): number;
  setLastmove(value: number): void;

  getFaintedreward(): number;
  setFaintedreward(value: number): void;

  getHeuristicdist(): Uint8Array | string;
  getHeuristicdist_asU8(): Uint8Array;
  getHeuristicdist_asB64(): string;
  setHeuristicdist(value: Uint8Array | string): void;

  getSwitchreward(): number;
  setSwitchreward(value: number): void;

  getTs(): number;
  setTs(value: number): void;

  getLongevityreward(): number;
  setLongevityreward(value: number): void;

  getDrawratio(): number;
  setDrawratio(value: number): void;

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
    winreward: number,
    hpreward: number,
    playerindex: boolean,
    turn: number,
    turnssinceswitch: number,
    heuristicaction: number,
    lastaction: number,
    lastmove: number,
    faintedreward: number,
    heuristicdist: Uint8Array | string,
    switchreward: number,
    ts: number,
    longevityreward: number,
    drawratio: number,
  }
}

export class State extends jspb.Message {
  hasInfo(): boolean;
  clearInfo(): void;
  getInfo(): Info | undefined;
  setInfo(value?: Info): void;

  getLegalactions(): Uint8Array | string;
  getLegalactions_asU8(): Uint8Array;
  getLegalactions_asB64(): string;
  setLegalactions(value: Uint8Array | string): void;

  hasHistory(): boolean;
  clearHistory(): void;
  getHistory(): history_pb.History | undefined;
  setHistory(value?: history_pb.History): void;

  getMoveset(): Uint8Array | string;
  getMoveset_asU8(): Uint8Array;
  getMoveset_asB64(): string;
  setMoveset(value: Uint8Array | string): void;

  getTeam(): Uint8Array | string;
  getTeam_asU8(): Uint8Array;
  getTeam_asB64(): string;
  setTeam(value: Uint8Array | string): void;

  getKey(): string;
  setKey(value: string): void;

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
    legalactions: Uint8Array | string,
    history?: history_pb.History.AsObject,
    moveset: Uint8Array | string,
    team: Uint8Array | string,
    key: string,
  }
}

