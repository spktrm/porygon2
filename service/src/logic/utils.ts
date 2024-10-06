import { OneDBoolean } from "./arr";

export function legalActionBufferToVector(legalActions: Uint8Array) {
    const vector = new OneDBoolean(10, Uint8Array);
    vector.buffer.set(legalActions);
    return vector.toBinaryVector();
}

export function getLegalActionIndices(legalActions: number[]): number[] {
    return [...legalActions].flatMap((value, index) => (!!value ? index : []));
}

export function chooseRandom(legalActions: Uint8Array): number {
    const vector = legalActionBufferToVector(legalActions);
    const legalIndices = getLegalActionIndices(vector);
    return legalIndices[Math.floor(Math.random() * legalIndices.length)];
}
