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

export class Action extends jspb.Message {
  getActionType(): number;
  setActionType(value: number): void;

  getMoveSlot(): number;
  setMoveSlot(value: number): void;

  getSwitchSlot(): number;
  setSwitchSlot(value: number): void;

  getWildcardSlot(): number;
  setWildcardSlot(value: number): void;

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
    actionType: number,
    moveSlot: number,
    switchSlot: number,
    wildcardSlot: number,
  }
}

export class StepRequest extends jspb.Message {
  getUsername(): string;
  setUsername(value: string): void;

  hasAction(): boolean;
  clearAction(): void;
  getAction(): Action | undefined;
  setAction(value?: Action): void;

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
    action?: Action.AsObject,
    rqid: number,
  }
}

export class ResetRequest extends jspb.Message {
  getUsername(): string;
  setUsername(value: string): void;

  clearSpeciesIndicesList(): void;
  getSpeciesIndicesList(): Array<number>;
  setSpeciesIndicesList(value: Array<number>): void;
  addSpeciesIndices(value: number, index?: number): number;

  clearPackedSetIndicesList(): void;
  getPackedSetIndicesList(): Array<number>;
  setPackedSetIndicesList(value: Array<number>): void;
  addPackedSetIndices(value: number, index?: number): number;

  getSmogonFormat(): string;
  setSmogonFormat(value: string): void;

  getCurrentCkpt(): number;
  setCurrentCkpt(value: number): void;

  getOpponentCkpt(): number;
  setOpponentCkpt(value: number): void;

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
    speciesIndicesList: Array<number>,
    packedSetIndicesList: Array<number>,
    smogonFormat: string,
    currentCkpt: number,
    opponentCkpt: number,
  }
}

export class EnvironmentState extends jspb.Message {
  getInfo(): Uint8Array | string;
  getInfo_asU8(): Uint8Array;
  getInfo_asB64(): string;
  setInfo(value: Uint8Array | string): void;

  getActionMask(): Uint8Array | string;
  getActionMask_asU8(): Uint8Array;
  getActionMask_asB64(): string;
  setActionMask(value: Uint8Array | string): void;

  getHistoryEntityPublic(): Uint8Array | string;
  getHistoryEntityPublic_asU8(): Uint8Array;
  getHistoryEntityPublic_asB64(): string;
  setHistoryEntityPublic(value: Uint8Array | string): void;

  getHistoryEntityRevealed(): Uint8Array | string;
  getHistoryEntityRevealed_asU8(): Uint8Array;
  getHistoryEntityRevealed_asB64(): string;
  setHistoryEntityRevealed(value: Uint8Array | string): void;

  getHistoryEntityEdges(): Uint8Array | string;
  getHistoryEntityEdges_asU8(): Uint8Array;
  getHistoryEntityEdges_asB64(): string;
  setHistoryEntityEdges(value: Uint8Array | string): void;

  getHistoryField(): Uint8Array | string;
  getHistoryField_asU8(): Uint8Array;
  getHistoryField_asB64(): string;
  setHistoryField(value: Uint8Array | string): void;

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

  getRevealedTeam(): Uint8Array | string;
  getRevealedTeam_asU8(): Uint8Array;
  getRevealedTeam_asB64(): string;
  setRevealedTeam(value: Uint8Array | string): void;

  getPrivateTeam(): Uint8Array | string;
  getPrivateTeam_asU8(): Uint8Array;
  getPrivateTeam_asB64(): string;
  setPrivateTeam(value: Uint8Array | string): void;

  getField(): Uint8Array | string;
  getField_asU8(): Uint8Array;
  getField_asB64(): string;
  setField(value: Uint8Array | string): void;

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
    actionMask: Uint8Array | string,
    historyEntityPublic: Uint8Array | string,
    historyEntityRevealed: Uint8Array | string,
    historyEntityEdges: Uint8Array | string,
    historyField: Uint8Array | string,
    historyLength: number,
    moveset: Uint8Array | string,
    publicTeam: Uint8Array | string,
    revealedTeam: Uint8Array | string,
    privateTeam: Uint8Array | string,
    field: Uint8Array | string,
    rqid: number,
  }
}

export class EnvironmentTrajectory extends jspb.Message {
  clearStatesList(): void;
  getStatesList(): Array<EnvironmentState>;
  setStatesList(value: Array<EnvironmentState>): void;
  addStates(value?: EnvironmentState, index?: number): EnvironmentState;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): EnvironmentTrajectory.AsObject;
  static toObject(includeInstance: boolean, msg: EnvironmentTrajectory): EnvironmentTrajectory.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: EnvironmentTrajectory, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): EnvironmentTrajectory;
  static deserializeBinaryFromReader(message: EnvironmentTrajectory, reader: jspb.BinaryReader): EnvironmentTrajectory;
}

export namespace EnvironmentTrajectory {
  export type AsObject = {
    statesList: Array<EnvironmentState.AsObject>,
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

  hasOpponentResetRequest(): boolean;
  clearOpponentResetRequest(): void;
  getOpponentResetRequest(): ResetRequest | undefined;
  setOpponentResetRequest(value?: ResetRequest): void;

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
    opponentResetRequest?: ResetRequest.AsObject,
  }
}

