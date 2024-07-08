// package: pokemon
// file: pokemon.proto

import * as jspb from "google-protobuf";
import * as enums_pb from "./enums_pb";

export class Pokemon extends jspb.Message {
  getSpecies(): enums_pb.SpeciesEnumMap[keyof enums_pb.SpeciesEnumMap];
  setSpecies(value: enums_pb.SpeciesEnumMap[keyof enums_pb.SpeciesEnumMap]): void;

  getItem(): enums_pb.ItemsEnumMap[keyof enums_pb.ItemsEnumMap];
  setItem(value: enums_pb.ItemsEnumMap[keyof enums_pb.ItemsEnumMap]): void;

  getAbility(): enums_pb.AbilitiesEnumMap[keyof enums_pb.AbilitiesEnumMap];
  setAbility(value: enums_pb.AbilitiesEnumMap[keyof enums_pb.AbilitiesEnumMap]): void;

  getMove1id(): enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap];
  setMove1id(value: enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap]): void;

  getMove2id(): enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap];
  setMove2id(value: enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap]): void;

  getMove3id(): enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap];
  setMove3id(value: enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap]): void;

  getMove4id(): enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap];
  setMove4id(value: enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap]): void;

  getPp1used(): number;
  setPp1used(value: number): void;

  getPp2used(): number;
  setPp2used(value: number): void;

  getPp3used(): number;
  setPp3used(value: number): void;

  getPp4used(): number;
  setPp4used(value: number): void;

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
    move1id: enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap],
    move2id: enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap],
    move3id: enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap],
    move4id: enums_pb.MovesEnumMap[keyof enums_pb.MovesEnumMap],
    pp1used: number,
    pp2used: number,
    pp3used: number,
    pp4used: number,
    hpratio: number,
    active: boolean,
    fainted: boolean,
    level: number,
    gender: enums_pb.GendersEnumMap[keyof enums_pb.GendersEnumMap],
  }
}

