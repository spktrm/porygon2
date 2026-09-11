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

export class ActionMask extends jspb.Message {
  getKind(): ActionRequestKindMap[keyof ActionRequestKindMap];
  setKind(value: ActionRequestKindMap[keyof ActionRequestKindMap]): void;

  getSwitchSlots(): number;
  setSwitchSlots(value: number): void;

  clearMoveTargetsList(): void;
  getMoveTargetsList(): Array<number>;
  setMoveTargetsList(value: Array<number>): void;
  addMoveTargets(value: number, index?: number): number;

  getOtherSrcs(): number;
  setOtherSrcs(value: number): void;

  getActiveSlot(): number;
  setActiveSlot(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ActionMask.AsObject;
  static toObject(includeInstance: boolean, msg: ActionMask): ActionMask.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ActionMask, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ActionMask;
  static deserializeBinaryFromReader(message: ActionMask, reader: jspb.BinaryReader): ActionMask;
}

export namespace ActionMask {
  export type AsObject = {
    kind: ActionRequestKindMap[keyof ActionRequestKindMap],
    switchSlots: number,
    moveTargetsList: Array<number>,
    otherSrcs: number,
    activeSlot: number,
  }
}

export class Action extends jspb.Message {
  getCell(): number;
  setCell(value: number): void;

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
    cell: number,
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

  getTeampreview(): boolean;
  setTeampreview(value: boolean): void;

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
    teampreview: boolean,
  }
}

export class ResetRequest extends jspb.Message {
  getUsername(): string;
  setUsername(value: string): void;

  getSmogonFormat(): string;
  setSmogonFormat(value: string): void;

  getGameId(): string;
  setGameId(value: string): void;

  clearPackedTeamsList(): void;
  getPackedTeamsList(): Array<number>;
  setPackedTeamsList(value: Array<number>): void;
  addPackedTeams(value: number, index?: number): number;

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
    smogonFormat: string,
    gameId: string,
    packedTeamsList: Array<number>,
  }
}

export class EnvironmentState extends jspb.Message {
  getInfo(): Uint8Array | string;
  getInfo_asU8(): Uint8Array;
  getInfo_asB64(): string;
  setInfo(value: Uint8Array | string): void;

  getHistoryEntityPublicCache(): Uint8Array | string;
  getHistoryEntityPublicCache_asU8(): Uint8Array;
  getHistoryEntityPublicCache_asB64(): string;
  setHistoryEntityPublicCache(value: Uint8Array | string): void;

  getHistoryEntityRevealedCache(): Uint8Array | string;
  getHistoryEntityRevealedCache_asU8(): Uint8Array;
  getHistoryEntityRevealedCache_asB64(): string;
  setHistoryEntityRevealedCache(value: Uint8Array | string): void;

  getHistoryEntityEdgeCache(): Uint8Array | string;
  getHistoryEntityEdgeCache_asU8(): Uint8Array;
  getHistoryEntityEdgeCache_asB64(): string;
  setHistoryEntityEdgeCache(value: Uint8Array | string): void;

  getHistoryField(): Uint8Array | string;
  getHistoryField_asU8(): Uint8Array;
  getHistoryField_asB64(): string;
  setHistoryField(value: Uint8Array | string): void;

  getHistoryLength(): number;
  setHistoryLength(value: number): void;

  getMyMoveset(): Uint8Array | string;
  getMyMoveset_asU8(): Uint8Array;
  getMyMoveset_asB64(): string;
  setMyMoveset(value: Uint8Array | string): void;

  getOppMoveset(): Uint8Array | string;
  getOppMoveset_asU8(): Uint8Array;
  getOppMoveset_asB64(): string;
  setOppMoveset(value: Uint8Array | string): void;

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

  getHistoryPackedLength(): number;
  setHistoryPackedLength(value: number): void;

  hasStructuredActionMask(): boolean;
  clearStructuredActionMask(): void;
  getStructuredActionMask(): ActionMask | undefined;
  setStructuredActionMask(value?: ActionMask): void;

  getOppPrivateTeam(): Uint8Array | string;
  getOppPrivateTeam_asU8(): Uint8Array;
  getOppPrivateTeam_asB64(): string;
  setOppPrivateTeam(value: Uint8Array | string): void;

  getHistoryRewriteCount(): number;
  setHistoryRewriteCount(value: number): void;

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
    historyEntityPublicCache: Uint8Array | string,
    historyEntityRevealedCache: Uint8Array | string,
    historyEntityEdgeCache: Uint8Array | string,
    historyField: Uint8Array | string,
    historyLength: number,
    myMoveset: Uint8Array | string,
    oppMoveset: Uint8Array | string,
    publicTeam: Uint8Array | string,
    revealedTeam: Uint8Array | string,
    privateTeam: Uint8Array | string,
    field: Uint8Array | string,
    rqid: number,
    historyPackedLength: number,
    structuredActionMask?: ActionMask.AsObject,
    oppPrivateTeam: Uint8Array | string,
    historyRewriteCount: number,
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

export class EnvironmentBatch extends jspb.Message {
  clearTrajectoriesList(): void;
  getTrajectoriesList(): Array<EnvironmentTrajectory>;
  setTrajectoriesList(value: Array<EnvironmentTrajectory>): void;
  addTrajectories(value?: EnvironmentTrajectory, index?: number): EnvironmentTrajectory;

  getMaxTrajectoryLength(): number;
  setMaxTrajectoryLength(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): EnvironmentBatch.AsObject;
  static toObject(includeInstance: boolean, msg: EnvironmentBatch): EnvironmentBatch.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: EnvironmentBatch, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): EnvironmentBatch;
  static deserializeBinaryFromReader(message: EnvironmentBatch, reader: jspb.BinaryReader): EnvironmentBatch;
}

export namespace EnvironmentBatch {
  export type AsObject = {
    trajectoriesList: Array<EnvironmentTrajectory.AsObject>,
    maxTrajectoryLength: number,
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

export interface ModalityEnumMap {
  MODALITY_ENUM___UNSPECIFIED: 0;
  MODALITY_ENUM__MOVE: 1;
  MODALITY_ENUM__SWITCH: 2;
  MODALITY_ENUM__WILDCARD: 3;
  MODALITY_ENUM__OTHER: 4;
}

export const ModalityEnum: ModalityEnumMap;

export interface MoveSlotMap {
  MOVE_SLOT___UNSPECIFIED: 0;
  MOVE_SLOT__ALLY_1_MOVE_1: 1;
  MOVE_SLOT__ALLY_1_MOVE_2: 2;
  MOVE_SLOT__ALLY_1_MOVE_3: 3;
  MOVE_SLOT__ALLY_1_MOVE_4: 4;
  MOVE_SLOT__ALLY_1_MOVE_1_WILDCARD: 5;
  MOVE_SLOT__ALLY_1_MOVE_2_WILDCARD: 6;
  MOVE_SLOT__ALLY_1_MOVE_3_WILDCARD: 7;
  MOVE_SLOT__ALLY_1_MOVE_4_WILDCARD: 8;
  MOVE_SLOT__ALLY_2_MOVE_1: 9;
  MOVE_SLOT__ALLY_2_MOVE_2: 10;
  MOVE_SLOT__ALLY_2_MOVE_3: 11;
  MOVE_SLOT__ALLY_2_MOVE_4: 12;
  MOVE_SLOT__ALLY_2_MOVE_1_WILDCARD: 13;
  MOVE_SLOT__ALLY_2_MOVE_2_WILDCARD: 14;
  MOVE_SLOT__ALLY_2_MOVE_3_WILDCARD: 15;
  MOVE_SLOT__ALLY_2_MOVE_4_WILDCARD: 16;
}

export const MoveSlot: MoveSlotMap;

export interface ReserveSlotMap {
  RESERVE_SLOT___UNSPECIFIED: 0;
  RESERVE_SLOT__RESERVE_1: 1;
  RESERVE_SLOT__RESERVE_2: 2;
  RESERVE_SLOT__RESERVE_3: 3;
  RESERVE_SLOT__RESERVE_4: 4;
  RESERVE_SLOT__RESERVE_5: 5;
  RESERVE_SLOT__RESERVE_6: 6;
}

export const ReserveSlot: ReserveSlotMap;

export interface TargetSlotMap {
  TARGET_SLOT___UNSPECIFIED: 0;
  TARGET_SLOT__DEFAULT: 1;
  TARGET_SLOT__ALLY_1: 2;
  TARGET_SLOT__ALLY_1_PASS: 3;
  TARGET_SLOT__ALLY_2: 4;
  TARGET_SLOT__ALLY_2_PASS: 5;
  TARGET_SLOT__ENEMY_1: 6;
  TARGET_SLOT__ENEMY_2: 7;
  TARGET_SLOT__AUTO: 8;
  TARGET_SLOT__ALL: 9;
  TARGET_SLOT__ALLY_SIDE: 10;
  TARGET_SLOT__FOE_SIDE: 11;
  TARGET_SLOT__ALLY_TEAM: 12;
  TARGET_SLOT__RANDOM_NORMAL: 13;
  TARGET_SLOT__ALL_ADJACENT: 14;
  TARGET_SLOT__ALL_ADJACENT_FOES: 15;
  TARGET_SLOT__ALLIES: 16;
}

export const TargetSlot: TargetSlotMap;

export interface ActionRequestKindMap {
  ACTION_REQUEST_KIND___UNSPECIFIED: 0;
  ACTION_REQUEST_KIND__MOVE: 1;
  ACTION_REQUEST_KIND__FORCE_SWITCH: 2;
  ACTION_REQUEST_KIND__TEAM_PREVIEW: 3;
  ACTION_REQUEST_KIND__WAIT: 4;
}

export const ActionRequestKind: ActionRequestKindMap;

