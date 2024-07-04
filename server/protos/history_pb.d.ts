// package: history
// file: history.proto

import * as jspb from "google-protobuf";
import * as pokemon_pb from "./pokemon_pb";
import * as enums_pb from "./enums_pb";

export class Boost extends jspb.Message {
  getStat(): enums_pb.BoostsEnumMap[keyof enums_pb.BoostsEnumMap];
  setStat(value: enums_pb.BoostsEnumMap[keyof enums_pb.BoostsEnumMap]): void;

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
    stat: enums_pb.BoostsEnumMap[keyof enums_pb.BoostsEnumMap],
    value: number,
  }
}

export class SideCondition extends jspb.Message {
  getType(): enums_pb.SideconditionsEnumMap[keyof enums_pb.SideconditionsEnumMap];
  setType(value: enums_pb.SideconditionsEnumMap[keyof enums_pb.SideconditionsEnumMap]): void;

  getValue(): number;
  setValue(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): SideCondition.AsObject;
  static toObject(includeInstance: boolean, msg: SideCondition): SideCondition.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: SideCondition, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): SideCondition;
  static deserializeBinaryFromReader(message: SideCondition, reader: jspb.BinaryReader): SideCondition;
}

export namespace SideCondition {
  export type AsObject = {
    type: enums_pb.SideconditionsEnumMap[keyof enums_pb.SideconditionsEnumMap],
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
  getSideconditionsList(): Array<SideCondition>;
  setSideconditionsList(value: Array<SideCondition>): void;
  addSideconditions(value?: SideCondition, index?: number): SideCondition;

  clearVolatilestatusList(): void;
  getVolatilestatusList(): Array<enums_pb.VolatilestatusEnumMap[keyof enums_pb.VolatilestatusEnumMap]>;
  setVolatilestatusList(value: Array<enums_pb.VolatilestatusEnumMap[keyof enums_pb.VolatilestatusEnumMap]>): void;
  addVolatilestatus(value: enums_pb.VolatilestatusEnumMap[keyof enums_pb.VolatilestatusEnumMap], index?: number): enums_pb.VolatilestatusEnumMap[keyof enums_pb.VolatilestatusEnumMap];

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
    sideconditionsList: Array<SideCondition.AsObject>,
    volatilestatusList: Array<enums_pb.VolatilestatusEnumMap[keyof enums_pb.VolatilestatusEnumMap]>,
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

  clearPseudoweatherList(): void;
  getPseudoweatherList(): Array<enums_pb.PseudoweatherEnumMap[keyof enums_pb.PseudoweatherEnumMap]>;
  setPseudoweatherList(value: Array<enums_pb.PseudoweatherEnumMap[keyof enums_pb.PseudoweatherEnumMap]>): void;
  addPseudoweather(value: enums_pb.PseudoweatherEnumMap[keyof enums_pb.PseudoweatherEnumMap], index?: number): enums_pb.PseudoweatherEnumMap[keyof enums_pb.PseudoweatherEnumMap];

  getAction(): ActionTypeEnumMap[keyof ActionTypeEnumMap];
  setAction(value: ActionTypeEnumMap[keyof ActionTypeEnumMap]): void;

  getMove(): enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap];
  setMove(value: enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap]): void;

  clearHyphenargsList(): void;
  getHyphenargsList(): Array<enums_pb.HyphenargsEnumMap[keyof enums_pb.HyphenargsEnumMap]>;
  setHyphenargsList(value: Array<enums_pb.HyphenargsEnumMap[keyof enums_pb.HyphenargsEnumMap]>): void;
  addHyphenargs(value: enums_pb.HyphenargsEnumMap[keyof enums_pb.HyphenargsEnumMap], index?: number): enums_pb.HyphenargsEnumMap[keyof enums_pb.HyphenargsEnumMap];

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
    pseudoweatherList: Array<enums_pb.PseudoweatherEnumMap[keyof enums_pb.PseudoweatherEnumMap]>,
    action: ActionTypeEnumMap[keyof ActionTypeEnumMap],
    move: enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap],
    hyphenargsList: Array<enums_pb.HyphenargsEnumMap[keyof enums_pb.HyphenargsEnumMap]>,
  }
}

export interface ActionTypeEnumMap {
  MOVE: 0;
  SWITCH: 1;
}

export const ActionTypeEnum: ActionTypeEnumMap;

