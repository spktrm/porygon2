// package: servicev2
// file: service.proto

import * as jspb from "google-protobuf";

export class ClientRequest extends jspb.Message {
  hasStep(): boolean;
  clearStep(): void;
  getStep(): StepRequest | undefined;
  setStep(value?: StepRequest): void;

  hasReset(): boolean;
  clearReset(): void;
  getReset(): ResetRequest | undefined;
  setReset(value?: ResetRequest): void;

  getMessageTypeCase(): ClientRequest.MessageTypeCase;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ClientRequest.AsObject;
  static toObject(includeInstance: boolean, msg: ClientRequest): ClientRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ClientRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ClientRequest;
  static deserializeBinaryFromReader(message: ClientRequest, reader: jspb.BinaryReader): ClientRequest;
}

export namespace ClientRequest {
  export type AsObject = {
    step?: StepRequest.AsObject,
    reset?: ResetRequest.AsObject,
  }

  export enum MessageTypeCase {
    MESSAGE_TYPE_NOT_SET = 0,
    STEP = 1,
    RESET = 2,
  }
}

export class StepRequest extends jspb.Message {
  getUsername(): string;
  setUsername(value: string): void;

  getAction(): number;
  setAction(value: number): void;

  getRqid(): number;
  setRqid(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): StepRequest.AsObject;
  static toObject(includeInstance: boolean, msg: StepRequest): StepRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: StepRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): StepRequest;
  static deserializeBinaryFromReader(message: StepRequest, reader: jspb.BinaryReader): StepRequest;
}

export namespace StepRequest {
  export type AsObject = {
    username: string,
    action: number,
    rqid: number,
  }
}

export class ResetRequest extends jspb.Message {
  getUsername(): string;
  setUsername(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ResetRequest.AsObject;
  static toObject(includeInstance: boolean, msg: ResetRequest): ResetRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ResetRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ResetRequest;
  static deserializeBinaryFromReader(message: ResetRequest, reader: jspb.BinaryReader): ResetRequest;
}

export namespace ResetRequest {
  export type AsObject = {
    username: string,
  }
}

export class EnvironmentState extends jspb.Message {
  getInfo(): Uint8Array | string;
  getInfo_asU8(): Uint8Array;
  getInfo_asB64(): string;
  setInfo(value: Uint8Array | string): void;

  getLegalActions(): Uint8Array | string;
  getLegalActions_asU8(): Uint8Array;
  getLegalActions_asB64(): string;
  setLegalActions(value: Uint8Array | string): void;

  getHistoryEntities(): Uint8Array | string;
  getHistoryEntities_asU8(): Uint8Array;
  getHistoryEntities_asB64(): string;
  setHistoryEntities(value: Uint8Array | string): void;

  getHistoryAbsoluteEdge(): Uint8Array | string;
  getHistoryAbsoluteEdge_asU8(): Uint8Array;
  getHistoryAbsoluteEdge_asB64(): string;
  setHistoryAbsoluteEdge(value: Uint8Array | string): void;

  getHistoryRelativeEdges(): Uint8Array | string;
  getHistoryRelativeEdges_asU8(): Uint8Array;
  getHistoryRelativeEdges_asB64(): string;
  setHistoryRelativeEdges(value: Uint8Array | string): void;

  getHistoryLength(): number;
  setHistoryLength(value: number): void;

  getMoveset(): Uint8Array | string;
  getMoveset_asU8(): Uint8Array;
  getMoveset_asB64(): string;
  setMoveset(value: Uint8Array | string): void;

  getPublicTeam(): Uint8Array | string;
  getPublicTeam_asU8(): Uint8Array;
  getPublicTeam_asB64(): string;
  setPublicTeam(value: Uint8Array | string): void;

  getPrivateTeam(): Uint8Array | string;
  getPrivateTeam_asU8(): Uint8Array;
  getPrivateTeam_asB64(): string;
  setPrivateTeam(value: Uint8Array | string): void;

  getCurrentContext(): Uint8Array | string;
  getCurrentContext_asU8(): Uint8Array;
  getCurrentContext_asB64(): string;
  setCurrentContext(value: Uint8Array | string): void;

  getRqid(): number;
  setRqid(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): EnvironmentState.AsObject;
  static toObject(includeInstance: boolean, msg: EnvironmentState): EnvironmentState.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: EnvironmentState, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): EnvironmentState;
  static deserializeBinaryFromReader(message: EnvironmentState, reader: jspb.BinaryReader): EnvironmentState;
}

export namespace EnvironmentState {
  export type AsObject = {
    info: Uint8Array | string,
    legalActions: Uint8Array | string,
    historyEntities: Uint8Array | string,
    historyAbsoluteEdge: Uint8Array | string,
    historyRelativeEdges: Uint8Array | string,
    historyLength: number,
    moveset: Uint8Array | string,
    publicTeam: Uint8Array | string,
    privateTeam: Uint8Array | string,
    currentContext: Uint8Array | string,
    rqid: number,
  }
}

export class EnvironmentResponse extends jspb.Message {
  getUsername(): string;
  setUsername(value: string): void;

  hasState(): boolean;
  clearState(): void;
  getState(): EnvironmentState | undefined;
  setState(value?: EnvironmentState): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): EnvironmentResponse.AsObject;
  static toObject(includeInstance: boolean, msg: EnvironmentResponse): EnvironmentResponse.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: EnvironmentResponse, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): EnvironmentResponse;
  static deserializeBinaryFromReader(message: EnvironmentResponse, reader: jspb.BinaryReader): EnvironmentResponse;
}

export namespace EnvironmentResponse {
  export type AsObject = {
    username: string,
    state?: EnvironmentState.AsObject,
  }
}

export class ErrorResponse extends jspb.Message {
  getTrace(): string;
  setTrace(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ErrorResponse.AsObject;
  static toObject(includeInstance: boolean, msg: ErrorResponse): ErrorResponse.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ErrorResponse, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ErrorResponse;
  static deserializeBinaryFromReader(message: ErrorResponse, reader: jspb.BinaryReader): ErrorResponse;
}

export namespace ErrorResponse {
  export type AsObject = {
    trace: string,
  }
}

export class WorkerRequest extends jspb.Message {
  getTaskId(): number;
  setTaskId(value: number): void;

  hasStepRequest(): boolean;
  clearStepRequest(): void;
  getStepRequest(): StepRequest | undefined;
  setStepRequest(value?: StepRequest): void;

  hasResetRequest(): boolean;
  clearResetRequest(): void;
  getResetRequest(): ResetRequest | undefined;
  setResetRequest(value?: ResetRequest): void;

  getRequestCase(): WorkerRequest.RequestCase;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): WorkerRequest.AsObject;
  static toObject(includeInstance: boolean, msg: WorkerRequest): WorkerRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: WorkerRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): WorkerRequest;
  static deserializeBinaryFromReader(message: WorkerRequest, reader: jspb.BinaryReader): WorkerRequest;
}

export namespace WorkerRequest {
  export type AsObject = {
    taskId: number,
    stepRequest?: StepRequest.AsObject,
    resetRequest?: ResetRequest.AsObject,
  }

  export enum RequestCase {
    REQUEST_NOT_SET = 0,
    STEP_REQUEST = 2,
    RESET_REQUEST = 3,
  }
}

export class WorkerResponse extends jspb.Message {
  getTaskId(): number;
  setTaskId(value: number): void;

  hasEnvironmentResponse(): boolean;
  clearEnvironmentResponse(): void;
  getEnvironmentResponse(): EnvironmentResponse | undefined;
  setEnvironmentResponse(value?: EnvironmentResponse): void;

  hasErrorResponse(): boolean;
  clearErrorResponse(): void;
  getErrorResponse(): ErrorResponse | undefined;
  setErrorResponse(value?: ErrorResponse): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): WorkerResponse.AsObject;
  static toObject(includeInstance: boolean, msg: WorkerResponse): WorkerResponse.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: WorkerResponse, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): WorkerResponse;
  static deserializeBinaryFromReader(message: WorkerResponse, reader: jspb.BinaryReader): WorkerResponse;
}

export namespace WorkerResponse {
  export type AsObject = {
    taskId: number,
    environmentResponse?: EnvironmentResponse.AsObject,
    errorResponse?: ErrorResponse.AsObject,
  }
}

