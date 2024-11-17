import { Action } from "../../protos/servicev2_pb";
import { Player } from "./player";

export type Rqid = number;
export type sendFnType = (player: Player) => Promise<Rqid | undefined>;
export type recvFnType = (rqid: Rqid) => Promise<Action | undefined>;
