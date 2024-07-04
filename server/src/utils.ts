import { EventEmitter } from "events";
import { LegalActions } from "../protos/state_pb";

export function chooseRandom(legalActions: LegalActions): number {
    const legalIndices = Object.values(legalActions.toObject()).flatMap(
        (v, i) => (v ? [i] : [])
    );
    return legalIndices[Math.floor(Math.random() * legalIndices.length)];
}
export class AsyncQueue<T> {
    private queue: T[];
    private eventEmitter: EventEmitter;

    constructor() {
        this.queue = [];
        this.eventEmitter = new EventEmitter();
    }

    enqueue(item: T): void {
        this.queue.push(item);
        this.eventEmitter.emit("itemAdded");
    }

    async dequeue(): Promise<T> {
        if (this.queue.length > 0) {
            return this.queue.shift()!;
        } else {
            return new Promise<T>((resolve) => {
                this.eventEmitter.once("itemAdded", () => {
                    resolve(this.queue.shift()!);
                });
            });
        }
    }

    size(): number {
        return this.queue.length;
    }
}
