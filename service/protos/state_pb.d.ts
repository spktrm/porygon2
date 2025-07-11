// package: rlenv
// file: state.proto

import * as jspb from "google-protobuf";
import * as enums_pb from "./enums_pb";
import * as history_pb from "./history_pb";

export class Rewards extends jspb.Message {
  getWinReward(): number;
  setWinReward(value: number): void;

  getHpReward(): number;
  setHpReward(value: number): void;

  getFaintedReward(): number;
  setFaintedReward(value: number): void;

  getScaledFaintedReward(): number;
  setScaledFaintedReward(value: number): void;

  getScaledHpReward(): number;
  setScaledHpReward(value: number): void;

  getTerminalHpReward(): number;
  setTerminalHpReward(value: number): void;

  getTerminalFaintedReward(): number;
  setTerminalFaintedReward(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Rewards.AsObject;
  static toObject(includeInstance: boolean, msg: Rewards): Rewards.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Rewards, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Rewards;
  static deserializeBinaryFromReader(message: Rewards, reader: jspb.BinaryReader): Rewards;
}

export namespace Rewards {
  export type AsObject = {
    winReward: number,
    hpReward: number,
    faintedReward: number,
    scaledFaintedReward: number,
    scaledHpReward: number,
    terminalHpReward: number,
    terminalFaintedReward: number,
  }
}

export class Heuristics extends jspb.Message {
  getHeuristicAction(): number;
  setHeuristicAction(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Heuristics.AsObject;
  static toObject(includeInstance: boolean, msg: Heuristics): Heuristics.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Heuristics, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Heuristics;
  static deserializeBinaryFromReader(message: Heuristics, reader: jspb.BinaryReader): Heuristics;
}

export namespace Heuristics {
  export type AsObject = {
    heuristicAction: number,
  }
}

export class Info extends jspb.Message {
  getGameId(): number;
  setGameId(value: number): void;

  getDone(): boolean;
  setDone(value: boolean): void;

  getPlayerIndex(): boolean;
  setPlayerIndex(value: boolean): void;

  getTurn(): number;
  setTurn(value: number): void;

  getTs(): number;
  setTs(value: number): void;

  getDrawRatio(): number;
  setDrawRatio(value: number): void;

  getWorkerIndex(): number;
  setWorkerIndex(value: number): void;

  hasRewards(): boolean;
  clearRewards(): void;
  getRewards(): Rewards | undefined;
  setRewards(value?: Rewards): void;

  getSeed(): number;
  setSeed(value: number): void;

  getDraw(): boolean;
  setDraw(value: boolean): void;

  hasHeuristics(): boolean;
  clearHeuristics(): void;
  getHeuristics(): Heuristics | undefined;
  setHeuristics(value?: Heuristics): void;

  getRequestCount(): number;
  setRequestCount(value: number): void;

  getTimestamp(): number;
  setTimestamp(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Info.AsObject;
  static toObject(includeInstance: boolean, msg: Info): Info.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Info, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Info;
  static deserializeBinaryFromReader(message: Info, reader: jspb.BinaryReader): Info;
}

export namespace Info {
  export type AsObject = {
    gameId: number,
    done: boolean,
    playerIndex: boolean,
    turn: number,
    ts: number,
    drawRatio: number,
    workerIndex: number,
    rewards?: Rewards.AsObject,
    seed: number,
    draw: boolean,
    heuristics?: Heuristics.AsObject,
    requestCount: number,
    timestamp: number,
  }
}

export class State extends jspb.Message {
  hasInfo(): boolean;
  clearInfo(): void;
  getInfo(): Info | undefined;
  setInfo(value?: Info): void;

  getLegalActions(): Uint8Array | string;
  getLegalActions_asU8(): Uint8Array;
  getLegalActions_asB64(): string;
  setLegalActions(value: Uint8Array | string): void;

  hasHistory(): boolean;
  clearHistory(): void;
  getHistory(): history_pb.History | undefined;
  setHistory(value?: history_pb.History): void;

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

  getKey(): string;
  setKey(value: string): void;

  getCurrentContext(): Uint8Array | string;
  getCurrentContext_asU8(): Uint8Array;
  getCurrentContext_asB64(): string;
  setCurrentContext(value: Uint8Array | string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): State.AsObject;
  static toObject(includeInstance: boolean, msg: State): State.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: State, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): State;
  static deserializeBinaryFromReader(message: State, reader: jspb.BinaryReader): State;
}

export namespace State {
  export type AsObject = {
    info?: Info.AsObject,
    legalActions: Uint8Array | string,
    history?: history_pb.History.AsObject,
    moveset: Uint8Array | string,
    publicTeam: Uint8Array | string,
    privateTeam: Uint8Array | string,
    key: string,
    currentContext: Uint8Array | string,
  }
}

export class Trajectory extends jspb.Message {
  clearStatesList(): void;
  getStatesList(): Array<State>;
  setStatesList(value: Array<State>): void;
  addStates(value?: State, index?: number): State;

  clearActionsList(): void;
  getActionsList(): Array<number>;
  setActionsList(value: Array<number>): void;
  addActions(value: number, index?: number): number;

  clearRewardsList(): void;
  getRewardsList(): Array<number>;
  setRewardsList(value: Array<number>): void;
  addRewards(value: number, index?: number): number;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Trajectory.AsObject;
  static toObject(includeInstance: boolean, msg: Trajectory): Trajectory.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Trajectory, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Trajectory;
  static deserializeBinaryFromReader(message: Trajectory, reader: jspb.BinaryReader): Trajectory;
}

export namespace Trajectory {
  export type AsObject = {
    statesList: Array<State.AsObject>,
    actionsList: Array<number>,
    rewardsList: Array<number>,
  }
}

export class Dataset extends jspb.Message {
  clearTrajectoriesList(): void;
  getTrajectoriesList(): Array<Trajectory>;
  setTrajectoriesList(value: Array<Trajectory>): void;
  addTrajectories(value?: Trajectory, index?: number): Trajectory;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Dataset.AsObject;
  static toObject(includeInstance: boolean, msg: Dataset): Dataset.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Dataset, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Dataset;
  static deserializeBinaryFromReader(message: Dataset, reader: jspb.BinaryReader): Dataset;
}

export namespace Dataset {
  export type AsObject = {
    trajectoriesList: Array<Trajectory.AsObject>,
  }
}

