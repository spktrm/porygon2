// package: history
// file: history.proto

import * as jspb from "google-protobuf";

export class History extends jspb.Message {
  getAbsoluteedge(): Uint8Array | string;
  getAbsoluteedge_asU8(): Uint8Array;
  getAbsoluteedge_asB64(): string;
  setAbsoluteedge(value: Uint8Array | string): void;

  getRelativeedges(): Uint8Array | string;
  getRelativeedges_asU8(): Uint8Array;
  getRelativeedges_asB64(): string;
  setRelativeedges(value: Uint8Array | string): void;

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
    absoluteedge: Uint8Array | string,
    relativeedges: Uint8Array | string,
    entities: Uint8Array | string,
    length: number,
  }
}

