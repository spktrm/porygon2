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
            this.data[element] |= 1 << bit;
        } else {
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

export class NDimBoolean<T extends TypedArray = Uint8Array> {
    private readonly dimensions: number[];
    private readonly data: OneDBoolean<T>;
    private readonly strides: number[];
    private readonly totalElements: number;

    constructor(
        dimensions: number[],
        bufferConstructor?: new (length: number) => T,
    ) {
        this.dimensions = dimensions;
        this.totalElements = this.calculateTotalElements(dimensions);
        this.data = new OneDBoolean(this.totalElements, bufferConstructor);
        this.strides = this.calculateStrides(dimensions);
    }

    private calculateTotalElements(dimensions: number[]): number {
        let total = 1;
        for (let i = 0; i < dimensions.length; i++) {
            total *= dimensions[i];
        }
        return total;
    }

    private calculateStrides(dimensions: number[]): number[] {
        const strides = new Array(dimensions.length);
        strides[dimensions.length - 1] = 1;
        for (let i = dimensions.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * dimensions[i + 1];
        }
        return strides;
    }

    private validateIndices(indices: number[]): void {
        if (indices.length !== this.dimensions.length) {
            throw new Error(
                `Expected ${this.dimensions.length} indices, got ${indices.length}`,
            );
        }
        for (let i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= this.dimensions[i]) {
                throw new RangeError(`Index out of bounds at dimension ${i}`);
            }
        }
    }

    private flattenIndex(indices: number[]): number {
        this.validateIndices(indices);
        let flatIndex = 0;
        for (let i = 0; i < indices.length; i++) {
            flatIndex += indices[i] * this.strides[i];
        }
        return flatIndex;
    }

    toggle(...indices: number[]): void {
        const flatIndex = this.flattenIndex(indices);
        this.data.toggle(flatIndex);
    }

    get(...indices: number[]): boolean {
        const flatIndex = this.flattenIndex(indices);
        return this.data.get(flatIndex);
    }

    set(...args: [...number[], boolean]): void {
        const value = args[args.length - 1] as boolean;
        const indices = args.slice(0, -1) as number[];
        const flatIndex = this.flattenIndex(indices);
        this.data.set(flatIndex, value);
    }

    get buffer(): T {
        return this.data.buffer;
    }

    toBinaryVector(): number[] {
        return this.data.toBinaryVector();
    }

    toBinaryArray(): number[] | number[][] | number[][][] | any {
        const vector = this.toBinaryVector();
        return this.reshapeVector(vector, this.dimensions);
    }

    private reshapeVector(
        vector: number[],
        dims: number[],
    ): number[] | number[][] | number[][][] | any {
        if (dims.length === 1) {
            return vector;
        }
        const result: any[] = [];
        const subArrayLength = vector.length / dims[0];
        for (let i = 0; i < dims[0]; i++) {
            const start = i * subArrayLength;
            const subArray = vector.slice(start, start + subArrayLength);
            result.push(this.reshapeVector(subArray, dims.slice(1)));
        }
        return result;
    }
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
