// package: history
// file: history.proto

import * as jspb from "google-protobuf";

export class History extends jspb.Message {
  getEdges(): Uint8Array | string;
  getEdges_asU8(): Uint8Array;
  getEdges_asB64(): string;
  setEdges(value: Uint8Array | string): void;

  getEntities(): Uint8Array | string;
  getEntities_asU8(): Uint8Array;
  getEntities_asB64(): string;
  setEntities(value: Uint8Array | string): void;

  getSideconditions(): Uint8Array | string;
  getSideconditions_asU8(): Uint8Array;
  getSideconditions_asB64(): string;
  setSideconditions(value: Uint8Array | string): void;

  getField(): Uint8Array | string;
  getField_asU8(): Uint8Array;
  getField_asB64(): string;
  setField(value: Uint8Array | string): void;

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
    edges: Uint8Array | string,
    entities: Uint8Array | string,
    sideconditions: Uint8Array | string,
    field: Uint8Array | string,
    length: number,
  }
}

