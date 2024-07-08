// package: history
// file: history.proto

import * as jspb from "google-protobuf";
import * as pokemon_pb from "./pokemon_pb";
import * as enums_pb from "./enums_pb";
import * as messages_pb from "./messages_pb";

export class Boost extends jspb.Message {
  getIndex(): enums_pb.BoostsEnumMap[keyof enums_pb.BoostsEnumMap];
  setIndex(value: enums_pb.BoostsEnumMap[keyof enums_pb.BoostsEnumMap]): void;

  getValue(): number;
  setValue(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Boost.AsObject;
  static toObject(includeInstance: boolean, msg: Boost): Boost.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Boost, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Boost;
  static deserializeBinaryFromReader(message: Boost, reader: jspb.BinaryReader): Boost;
}

export namespace Boost {
  export type AsObject = {
    index: enums_pb.BoostsEnumMap[keyof enums_pb.BoostsEnumMap],
    value: number,
  }
}

export class Sidecondition extends jspb.Message {
  getIndex(): enums_pb.SideconditionsEnumMap[keyof enums_pb.SideconditionsEnumMap];
  setIndex(value: enums_pb.SideconditionsEnumMap[keyof enums_pb.SideconditionsEnumMap]): void;

  getValue(): number;
  setValue(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Sidecondition.AsObject;
  static toObject(includeInstance: boolean, msg: Sidecondition): Sidecondition.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Sidecondition, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Sidecondition;
  static deserializeBinaryFromReader(message: Sidecondition, reader: jspb.BinaryReader): Sidecondition;
}

export namespace Sidecondition {
  export type AsObject = {
    index: enums_pb.SideconditionsEnumMap[keyof enums_pb.SideconditionsEnumMap],
    value: number,
  }
}

export class Volatilestatus extends jspb.Message {
  getIndex(): enums_pb.VolatilestatusEnumMap[keyof enums_pb.VolatilestatusEnumMap];
  setIndex(value: enums_pb.VolatilestatusEnumMap[keyof enums_pb.VolatilestatusEnumMap]): void;

  getValue(): number;
  setValue(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Volatilestatus.AsObject;
  static toObject(includeInstance: boolean, msg: Volatilestatus): Volatilestatus.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Volatilestatus, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Volatilestatus;
  static deserializeBinaryFromReader(message: Volatilestatus, reader: jspb.BinaryReader): Volatilestatus;
}

export namespace Volatilestatus {
  export type AsObject = {
    index: enums_pb.VolatilestatusEnumMap[keyof enums_pb.VolatilestatusEnumMap],
    value: number,
  }
}

export class HistorySide extends jspb.Message {
  hasActive(): boolean;
  clearActive(): void;
  getActive(): pokemon_pb.Pokemon | undefined;
  setActive(value?: pokemon_pb.Pokemon): void;

  clearBoostsList(): void;
  getBoostsList(): Array<Boost>;
  setBoostsList(value: Array<Boost>): void;
  addBoosts(value?: Boost, index?: number): Boost;

  clearSideconditionsList(): void;
  getSideconditionsList(): Array<Sidecondition>;
  setSideconditionsList(value: Array<Sidecondition>): void;
  addSideconditions(value?: Sidecondition, index?: number): Sidecondition;

  clearVolatilestatusList(): void;
  getVolatilestatusList(): Array<Volatilestatus>;
  setVolatilestatusList(value: Array<Volatilestatus>): void;
  addVolatilestatus(value?: Volatilestatus, index?: number): Volatilestatus;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): HistorySide.AsObject;
  static toObject(includeInstance: boolean, msg: HistorySide): HistorySide.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: HistorySide, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): HistorySide;
  static deserializeBinaryFromReader(message: HistorySide, reader: jspb.BinaryReader): HistorySide;
}

export namespace HistorySide {
  export type AsObject = {
    active?: pokemon_pb.Pokemon.AsObject,
    boostsList: Array<Boost.AsObject>,
    sideconditionsList: Array<Sidecondition.AsObject>,
    volatilestatusList: Array<Volatilestatus.AsObject>,
  }
}

export class HistoryStep extends jspb.Message {
  hasP1(): boolean;
  clearP1(): void;
  getP1(): HistorySide | undefined;
  setP1(value?: HistorySide): void;

  hasP2(): boolean;
  clearP2(): void;
  getP2(): HistorySide | undefined;
  setP2(value?: HistorySide): void;

  getWeather(): enums_pb.WeathersEnumMap[keyof enums_pb.WeathersEnumMap];
  setWeather(value: enums_pb.WeathersEnumMap[keyof enums_pb.WeathersEnumMap]): void;

  hasPseudoweather(): boolean;
  clearPseudoweather(): void;
  getPseudoweather(): messages_pb.PseudoweatherMessage | undefined;
  setPseudoweather(value?: messages_pb.PseudoweatherMessage): void;

  getAction(): ActionTypeEnumMap[keyof ActionTypeEnumMap];
  setAction(value: ActionTypeEnumMap[keyof ActionTypeEnumMap]): void;

  getMove(): enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap];
  setMove(value: enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap]): void;

  hasHyphenargs(): boolean;
  clearHyphenargs(): void;
  getHyphenargs(): messages_pb.HyphenargsMessage | undefined;
  setHyphenargs(value?: messages_pb.HyphenargsMessage): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): HistoryStep.AsObject;
  static toObject(includeInstance: boolean, msg: HistoryStep): HistoryStep.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: HistoryStep, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): HistoryStep;
  static deserializeBinaryFromReader(message: HistoryStep, reader: jspb.BinaryReader): HistoryStep;
}

export namespace HistoryStep {
  export type AsObject = {
    p1?: HistorySide.AsObject,
    p2?: HistorySide.AsObject,
    weather: enums_pb.WeathersEnumMap[keyof enums_pb.WeathersEnumMap],
    pseudoweather?: messages_pb.PseudoweatherMessage.AsObject,
    action: ActionTypeEnumMap[keyof ActionTypeEnumMap],
    move: enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap],
    hyphenargs?: messages_pb.HyphenargsMessage.AsObject,
  }
}

export interface ActionTypeEnumMap {
  MOVE: 0;
  SWITCH: 1;
}

export const ActionTypeEnum: ActionTypeEnumMap;

