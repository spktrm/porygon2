import { EventEmitter } from "events";
import { LegalActions } from "../../protos/state_pb";

export function chooseRandom(legalActions: LegalActions): number {
    const legalIndices = Object.values(legalActions.toObject()).flatMap(
        (v, i) => (v ? [i] : []),
    );
    return legalIndices[Math.floor(Math.random() * legalIndices.length)];
}
