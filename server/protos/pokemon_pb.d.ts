// package: pokemon
// file: pokemon.proto

import * as jspb from "google-protobuf";
import * as enums_pb from "./enums_pb";

export class Move extends jspb.Message {
  getMoveid(): enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap];
  setMoveid(value: enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap]): void;

  getPpused(): number;
  setPpused(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Move.AsObject;
  static toObject(includeInstance: boolean, msg: Move): Move.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Move, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Move;
  static deserializeBinaryFromReader(message: Move, reader: jspb.BinaryReader): Move;
}

export namespace Move {
  export type AsObject = {
    moveid: enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap],
    ppused: number,
  }
}

export class Pokemon extends jspb.Message {
  getSpecies(): enums_pb.SpeciesEnumMap[keyof enums_pb.SpeciesEnumMap];
  setSpecies(value: enums_pb.SpeciesEnumMap[keyof enums_pb.SpeciesEnumMap]): void;

  getItem(): enums_pb.ItemsEnumMap[keyof enums_pb.ItemsEnumMap];
  setItem(value: enums_pb.ItemsEnumMap[keyof enums_pb.ItemsEnumMap]): void;

  getAbility(): enums_pb.AbilitiesEnumMap[keyof enums_pb.AbilitiesEnumMap];
  setAbility(value: enums_pb.AbilitiesEnumMap[keyof enums_pb.AbilitiesEnumMap]): void;

  clearMovesetList(): void;
  getMovesetList(): Array<Move>;
  setMovesetList(value: Array<Move>): void;
  addMoveset(value?: Move, index?: number): Move;

  getHpratio(): number;
  setHpratio(value: number): void;

  getActive(): boolean;
  setActive(value: boolean): void;

  getFainted(): boolean;
  setFainted(value: boolean): void;

  getLevel(): number;
  setLevel(value: number): void;

  getGender(): enums_pb.GendersEnumMap[keyof enums_pb.GendersEnumMap];
  setGender(value: enums_pb.GendersEnumMap[keyof enums_pb.GendersEnumMap]): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Pokemon.AsObject;
  static toObject(includeInstance: boolean, msg: Pokemon): Pokemon.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Pokemon, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Pokemon;
  static deserializeBinaryFromReader(message: Pokemon, reader: jspb.BinaryReader): Pokemon;
}

export namespace Pokemon {
  export type AsObject = {
    species: enums_pb.SpeciesEnumMap[keyof enums_pb.SpeciesEnumMap],
    item: enums_pb.ItemsEnumMap[keyof enums_pb.ItemsEnumMap],
    ability: enums_pb.AbilitiesEnumMap[keyof enums_pb.AbilitiesEnumMap],
    movesetList: Array<Move.AsObject>,
    hpratio: number,
    active: boolean,
    fainted: boolean,
    level: number,
    gender: enums_pb.GendersEnumMap[keyof enums_pb.GendersEnumMap],
  }
}

