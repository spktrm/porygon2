// package: rlenv
// file: state.proto

import * as jspb from "google-protobuf";
import * as history_pb from "./history_pb";
import * as enums_pb from "./enums_pb";

export class Rewards extends jspb.Message {
  getWinreward(): number;
  setWinreward(value: number): void;

  getHpreward(): number;
  setHpreward(value: number): void;

  getFaintedreward(): number;
  setFaintedreward(value: number): void;

  getSwitchreward(): number;
  setSwitchreward(value: number): void;

  getLongevityreward(): number;
  setLongevityreward(value: number): void;

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
    winreward: number,
    hpreward: number,
    faintedreward: number,
    switchreward: number,
    longevityreward: number,
  }
}

export class Heuristics extends jspb.Message {
  getHeuristicaction(): number;
  setHeuristicaction(value: number): void;

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
    heuristicaction: number,
  }
}

export class Info extends jspb.Message {
  getGameid(): number;
  setGameid(value: number): void;

  getDone(): boolean;
  setDone(value: boolean): void;

  getPlayerindex(): boolean;
  setPlayerindex(value: boolean): void;

  getTurn(): number;
  setTurn(value: number): void;

  getTs(): number;
  setTs(value: number): void;

  getDrawratio(): number;
  setDrawratio(value: number): void;

  getWorkerindex(): number;
  setWorkerindex(value: number): void;

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
    gameid: number,
    done: boolean,
    playerindex: boolean,
    turn: number,
    ts: number,
    drawratio: number,
    workerindex: number,
    rewards?: Rewards.AsObject,
    seed: number,
    draw: boolean,
    heuristics?: Heuristics.AsObject,
  }
}

export class State extends jspb.Message {
  hasInfo(): boolean;
  clearInfo(): void;
  getInfo(): Info | undefined;
  setInfo(value?: Info): void;

  getLegalactions(): Uint8Array | string;
  getLegalactions_asU8(): Uint8Array;
  getLegalactions_asB64(): string;
  setLegalactions(value: Uint8Array | string): void;

  hasHistory(): boolean;
  clearHistory(): void;
  getHistory(): history_pb.History | undefined;
  setHistory(value?: history_pb.History): void;

  getMoveset(): Uint8Array | string;
  getMoveset_asU8(): Uint8Array;
  getMoveset_asB64(): string;
  setMoveset(value: Uint8Array | string): void;

  getTeam(): Uint8Array | string;
  getTeam_asU8(): Uint8Array;
  getTeam_asB64(): string;
  setTeam(value: Uint8Array | string): void;

  getKey(): string;
  setKey(value: string): void;

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
    legalactions: Uint8Array | string,
    history?: history_pb.History.AsObject,
    moveset: Uint8Array | string,
    team: Uint8Array | string,
    key: string,
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

