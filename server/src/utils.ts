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
    private waitingDequeuePromises: ((value: T) => void)[];
    private isProcessing: boolean;

    constructor() {
        this.queue = [];
        this.waitingDequeuePromises = [];
        this.isProcessing = false;
    }

    enqueue(item: T): void {
        if (this.waitingDequeuePromises.length > 0) {
            const resolve = this.waitingDequeuePromises.shift();
            if (resolve) {
                resolve(item);
            }
        } else {
            this.queue.push(item);
        }
    }

    async dequeue(): Promise<T> {
        if (this.queue.length > 0) {
            return Promise.resolve(this.queue.shift()!);
        }

        return new Promise<T>((resolve) => {
            this.waitingDequeuePromises.push(resolve);
        });
    }

    size(): number {
        return this.queue.length;
    }

    clear(): void {
        this.queue = [];
        this.waitingDequeuePromises.forEach((resolve) => resolve(null as any));
        this.waitingDequeuePromises = [];
    }
}
