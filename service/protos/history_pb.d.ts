// package: history
// file: history.proto

import * as jspb from "google-protobuf";

export class History extends jspb.Message {
  getAbsoluteEdge(): Uint8Array | string;
  getAbsoluteEdge_asU8(): Uint8Array;
  getAbsoluteEdge_asB64(): string;
  setAbsoluteEdge(value: Uint8Array | string): void;

  getRelativeEdges(): Uint8Array | string;
  getRelativeEdges_asU8(): Uint8Array;
  getRelativeEdges_asB64(): string;
  setRelativeEdges(value: Uint8Array | string): void;

  getEntities(): Uint8Array | string;
  getEntities_asU8(): Uint8Array;
  getEntities_asB64(): string;
  setEntities(value: Uint8Array | string): void;

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
    absoluteEdge: Uint8Array | string,
    relativeEdges: Uint8Array | string,
    entities: Uint8Array | string,
    length: number,
  }
}

