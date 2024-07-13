// package: history
// file: history.proto

import * as jspb from "google-protobuf";

export class History extends jspb.Message {
  getActive(): Uint8Array | string;
  getActive_asU8(): Uint8Array;
  getActive_asB64(): string;
  setActive(value: Uint8Array | string): void;

  getBoosts(): Uint8Array | string;
  getBoosts_asU8(): Uint8Array;
  getBoosts_asB64(): string;
  setBoosts(value: Uint8Array | string): void;

  getSideconditions(): Uint8Array | string;
  getSideconditions_asU8(): Uint8Array;
  getSideconditions_asB64(): string;
  setSideconditions(value: Uint8Array | string): void;

  getVolatilestatus(): Uint8Array | string;
  getVolatilestatus_asU8(): Uint8Array;
  getVolatilestatus_asB64(): string;
  setVolatilestatus(value: Uint8Array | string): void;

  getHyphenargs(): Uint8Array | string;
  getHyphenargs_asU8(): Uint8Array;
  getHyphenargs_asB64(): string;
  setHyphenargs(value: Uint8Array | string): void;

  getWeather(): Uint8Array | string;
  getWeather_asU8(): Uint8Array;
  getWeather_asB64(): string;
  setWeather(value: Uint8Array | string): void;

  getPseudoweather(): Uint8Array | string;
  getPseudoweather_asU8(): Uint8Array;
  getPseudoweather_asB64(): string;
  setPseudoweather(value: Uint8Array | string): void;

  getTurncontext(): Uint8Array | string;
  getTurncontext_asU8(): Uint8Array;
  getTurncontext_asB64(): string;
  setTurncontext(value: Uint8Array | string): void;

  getLength(): number;
  setLength(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): History.AsObject;
  static toObject(includeInstance: boolean, msg: History): History.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: History, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): History;
  static deserializeBinaryFromReader(message: History, reader: jspb.BinaryReader): History;
}

export namespace History {
  export type AsObject = {
    active: Uint8Array | string,
    boosts: Uint8Array | string,
    sideconditions: Uint8Array | string,
    volatilestatus: Uint8Array | string,
    hyphenargs: Uint8Array | string,
    weather: Uint8Array | string,
    pseudoweather: Uint8Array | string,
    turncontext: Uint8Array | string,
    length: number,
  }
}

export interface ActionTypeEnumMap {
  MOVE: 0;
  SWITCH: 1;
}

export const ActionTypeEnum: ActionTypeEnumMap;

