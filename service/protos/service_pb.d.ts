// package: servicev2
// file: service.proto

import * as jspb from "google-protobuf";

export class ClientMessage extends jspb.Message {
  getPlayerId(): number;
  setPlayerId(value: number): void;

  getGameId(): number;
  setGameId(value: number): void;

  hasConnect(): boolean;
  clearConnect(): void;
  getConnect(): ConnectMessage | undefined;
  setConnect(value?: ConnectMessage): void;

  hasStep(): boolean;
  clearStep(): void;
  getStep(): StepMessage | undefined;
  setStep(value?: StepMessage): void;

  hasReset(): boolean;
  clearReset(): void;
  getReset(): ResetMessage | undefined;
  setReset(value?: ResetMessage): void;

  getMessageTypeCase(): ClientMessage.MessageTypeCase;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ClientMessage.AsObject;
  static toObject(includeInstance: boolean, msg: ClientMessage): ClientMessage.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ClientMessage, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ClientMessage;
  static deserializeBinaryFromReader(message: ClientMessage, reader: jspb.BinaryReader): ClientMessage;
}

export namespace ClientMessage {
  export type AsObject = {
    playerId: number,
    gameId: number,
    connect?: ConnectMessage.AsObject,
    step?: StepMessage.AsObject,
    reset?: ResetMessage.AsObject,
  }

  export enum MessageTypeCase {
    MESSAGE_TYPE_NOT_SET = 0,
    CONNECT = 3,
    STEP = 4,
    RESET = 5,
  }
}

export class Action extends jspb.Message {
  getRqid(): number;
  setRqid(value: number): void;

  getValue(): number;
  setValue(value: number): void;

  getText(): string;
  setText(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Action.AsObject;
  static toObject(includeInstance: boolean, msg: Action): Action.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Action, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Action;
  static deserializeBinaryFromReader(message: Action, reader: jspb.BinaryReader): Action;
}

export namespace Action {
  export type AsObject = {
    rqid: number,
    value: number,
    text: string,
  }
}

export class StepMessage extends jspb.Message {
  hasAction(): boolean;
  clearAction(): void;
  getAction(): Action | undefined;
  setAction(value?: Action): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): StepMessage.AsObject;
  static toObject(includeInstance: boolean, msg: StepMessage): StepMessage.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: StepMessage, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): StepMessage;
  static deserializeBinaryFromReader(message: StepMessage, reader: jspb.BinaryReader): StepMessage;
}

export namespace StepMessage {
  export type AsObject = {
    action?: Action.AsObject,
  }
}

export class ResetMessage extends jspb.Message {
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ResetMessage.AsObject;
  static toObject(includeInstance: boolean, msg: ResetMessage): ResetMessage.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ResetMessage, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ResetMessage;
  static deserializeBinaryFromReader(message: ResetMessage, reader: jspb.BinaryReader): ResetMessage;
}

export namespace ResetMessage {
  export type AsObject = {
  }
}

export class ConnectMessage extends jspb.Message {
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ConnectMessage.AsObject;
  static toObject(includeInstance: boolean, msg: ConnectMessage): ConnectMessage.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ConnectMessage, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ConnectMessage;
  static deserializeBinaryFromReader(message: ConnectMessage, reader: jspb.BinaryReader): ConnectMessage;
}

export namespace ConnectMessage {
  export type AsObject = {
  }
}

export class ServerMessage extends jspb.Message {
  hasGameState(): boolean;
  clearGameState(): void;
  getGameState(): GameState | undefined;
  setGameState(value?: GameState): void;

  hasError(): boolean;
  clearError(): void;
  getError(): ErrorMessage | undefined;
  setError(value?: ErrorMessage): void;

  getMessageTypeCase(): ServerMessage.MessageTypeCase;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ServerMessage.AsObject;
  static toObject(includeInstance: boolean, msg: ServerMessage): ServerMessage.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ServerMessage, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ServerMessage;
  static deserializeBinaryFromReader(message: ServerMessage, reader: jspb.BinaryReader): ServerMessage;
}

export namespace ServerMessage {
  export type AsObject = {
    gameState?: GameState.AsObject,
    error?: ErrorMessage.AsObject,
  }

  export enum MessageTypeCase {
    MESSAGE_TYPE_NOT_SET = 0,
    GAME_STATE = 1,
    ERROR = 2,
  }
}

export class GameState extends jspb.Message {
  getPlayerId(): number;
  setPlayerId(value: number): void;

  getRqid(): number;
  setRqid(value: number): void;

  getState(): Uint8Array | string;
  getState_asU8(): Uint8Array;
  getState_asB64(): string;
  setState(value: Uint8Array | string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): GameState.AsObject;
  static toObject(includeInstance: boolean, msg: GameState): GameState.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: GameState, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): GameState;
  static deserializeBinaryFromReader(message: GameState, reader: jspb.BinaryReader): GameState;
}

export namespace GameState {
  export type AsObject = {
    playerId: number,
    rqid: number,
    state: Uint8Array | string,
  }
}

export class ErrorMessage extends jspb.Message {
  getErrorMessage(): string;
  setErrorMessage(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ErrorMessage.AsObject;
  static toObject(includeInstance: boolean, msg: ErrorMessage): ErrorMessage.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ErrorMessage, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ErrorMessage;
  static deserializeBinaryFromReader(message: ErrorMessage, reader: jspb.BinaryReader): ErrorMessage;
}

export namespace ErrorMessage {
  export type AsObject = {
    errorMessage: string,
  }
}

