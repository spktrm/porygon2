/* eslint-disable @typescript-eslint/no-explicit-any */
export class TaskQueueSystem<T> {
    private results: Map<number, Promise<T>> = new Map();
    private resolvers: Map<number, (value: T) => void> = new Map();
    private pointer: number;

    constructor() {
        this.pointer = 0;
    }

    // Method to generate a unique key for each task
    private generateKey(): number {
        const currentKey = this.pointer;
        this.pointer += 1;
        return currentKey;
    }

    // Method to create a job and return a key
    public createJob(): number {
        const key = this.generateKey();
        this.results.set(
            key,
            new Promise<T>((resolve) => {
                this.resolvers.set(key, resolve);
            }),
        );
        return key;
    }

    // Method to submit a result for a given job key
    public submitResult(id: number, result: T): void {
        if (this.resolvers.has(id)) {
            this.resolvers.get(id)!(result);
            this.resolvers.delete(id);
        } else {
            throw new Error("Invalid job id");
        }
    }

    // Method to await the result of a completed job
    public async getResult(key: number): Promise<T> {
        const resultPromise = this.results.get(key);
        if (resultPromise) {
            const result = await resultPromise;
            this.results.delete(key);
            return result;
        } else {
            throw new Error("Invalid job id");
        }
    }

    public allDone(): boolean {
        return this.resolvers.size === 0 && this.results.size === 0;
    }

    reset() {
        this.pointer = 0;
    }
}

export type TypedArray =
    | Int8Array
    | Uint8Array
    | Uint8ClampedArray
    | Int16Array
    | Uint16Array
    | Int32Array
    | Uint32Array
    | Float32Array
    | Float64Array;

const typedArrayElementSizes: {
    [K in TypedArray["constructor"]["name"]]: number;
} = {
    Int8Array: 1,
    Uint8Array: 1,
    Uint8ClampedArray: 1,
    Int16Array: 2,
    Uint16Array: 2,
    Int32Array: 4,
    Uint32Array: 4,
    Float32Array: 4,
    Float64Array: 8,
};

export class OneDBoolean<T extends TypedArray = Uint8Array> {
    private readonly length: number;
    private readonly data: T;
    private readonly bitsPerElement: number;
    private readonly mask: number;
    sum: number;

    constructor(
        length: number,
        bufferConstructor: new (length: number) => T = Uint8Array as any,
    ) {
        this.length = length;
        const elementSize =
            typedArrayElementSizes[
                bufferConstructor.name as keyof typeof typedArrayElementSizes
            ];
        this.bitsPerElement = elementSize * 8;
        this.mask = this.bitsPerElement - 1;
        this.data = new bufferConstructor(
            Math.ceil(length / this.bitsPerElement),
        );
        this.sum = 0;
    }

    setBuffer(buffer: T): void {
        if (buffer.length < this.data.length) {
            throw new Error(
                `Buffer length ${buffer.length} is less than required length ${this.data.length}`,
            );
        }
        this.data.set(buffer.subarray(0, this.data.length));
    }

    private getElementAndBit(index: number): [number, number] {
        const element = (index / this.bitsPerElement) | 0;
        const bit = this.bitsPerElement - 1 - (index & this.mask); // Big-endian adjustment
        return [element, bit];
    }

    toggle(index: number): void {
        if (index < 0 || index >= this.length)
            throw new RangeError("Index out of bounds");
        const [element, bit] = this.getElementAndBit(index);
        if (bit === 0) {
            this.sum += 1;
        } else {
            this.sum -= 1;
        }
        this.data[element] ^= 1 << bit;
    }

    get(index: number): boolean {
        if (index < 0 || index >= this.length)
            throw new RangeError("Index out of bounds");
        const [element, bit] = this.getElementAndBit(index);
        return !!(this.data[element] & (1 << bit));
    }

    set(index: number, value: boolean): void {
        if (index < 0 || index >= this.length)
            throw new RangeError("Index out of bounds");
        const [element, bit] = this.getElementAndBit(index);
        if (value) {
            this.sum += 1;
            this.data[element] |= 1 << bit;
        } else {
            this.sum -= 1;
            this.data[element] &= ~(1 << bit);
        }
    }

    get buffer(): T {
        return this.data;
    }

    toBinaryVector(): number[] {
        const result: number[] = new Array(this.length);
        for (let i = 0; i < this.length; i++) {
            result[i] = this.get(i) ? 1 : 0;
        }
        return result;
    }
}

export function generateRandomString(length: number): string {
    const readableAsciiStart = 97; // 'a'
    const readableAsciiEnd = 122; // 'z'
    const range = readableAsciiEnd - readableAsciiStart + 1;

    const array = new Uint8Array(length);
    crypto.getRandomValues(array);

    // Map each byte to a readable ASCII character
    const chars = Array.from(array, (byte) =>
        String.fromCharCode(readableAsciiStart + (byte % range)),
    );

    return chars.join("");
}

export function isEvalUser(userName: string) {
    return userName.startsWith("eval");
}

export function isBaselineUser(userName: string) {
    return userName.startsWith("baseline");
}
export class AsyncQueue<T> {
    private queue: T[] = [];
    private maxSize: number;
    private waitingConsumers: ((value: T | PromiseLike<T>) => void)[] = [];

    constructor(maxSize: number = Infinity) {
        this.maxSize = maxSize;
    }

    /**
     * Adds an item to the queue synchronously. Throws an error if the queue is full.
     * @param item - The item to be added to the queue
     */
    put(item: T): void {
        if (this.queue.length >= this.maxSize) {
            throw new Error("Queue is full");
        }

        this.queue.push(item);

        // If there are waiting consumers, immediately provide them with the item
        if (this.waitingConsumers.length > 0) {
            const consumer = this.waitingConsumers.shift();
            if (consumer) consumer(this.queue.shift()!);
        }
    }

    /**
     * Removes and returns an item from the queue. If the queue is empty, it waits until an item is available.
     * @returns A promise that resolves to the item
     */
    async get(): Promise<T> {
        if (this.queue.length === 0) {
            return new Promise<T>((resolve) =>
                this.waitingConsumers.push(resolve),
            );
        }

        const item = this.queue.shift()!;
        return item;
    }

    /**
     * Returns the current size of the queue.
     */
    size(): number {
        return this.queue.length;
    }

    /**
     * Checks if the queue is empty.
     */
    isEmpty(): boolean {
        return this.queue.length === 0;
    }

    /**
     * Checks if the queue is full.
     */
    isFull(): boolean {
        return this.queue.length >= this.maxSize;
    }
}
