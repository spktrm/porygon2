import { numActionFeatures } from "./data";

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
    readonly length: number;
    readonly width: number | undefined;
    readonly height: number | undefined;
    readonly data: T;
    readonly bitsPerElement: number;
    readonly mask: number;

    constructor(
        length: number,
        bufferConstructor: new (length: number) => T = Uint8Array as any,
        width: number | undefined = undefined,
    ) {
        this.length = length;
        this.width = width;
        if (width !== undefined) {
            if (length % width !== 0) {
                throw new Error(
                    "Length must be a multiple of width for 2D representation",
                );
            }
            this.height = Math.ceil(length / width);
        }

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

    setRowCol(row: number, col: number, value: boolean): void {
        if (!row || !col) {
            throw new Error(
                "row and col must be defined for 2D representation",
            );
        }
        const index = row * this.width! + col;
        this.set(index, value);
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

    sum(): number {
        let total = 0;
        for (let i = 0; i < this.length; i++) {
            if (this.get(i)) {
                total += 1;
            }
        }
        return total;
    }

    split(parts: number): OneDBoolean<T>[] {
        if (!Number.isInteger(parts) || parts <= 0) {
            throw new RangeError("parts must be a positive integer");
        }

        const result: OneDBoolean<T>[] = [];

        const baseSize = Math.floor(this.length / parts);
        const remainder = this.length % parts;

        // Use the same buffer constructor as the original
        const bufferConstructor = this.data.constructor as {
            new (length: number): T;
        };

        let offset = 0;

        for (let i = 0; i < parts; i++) {
            const segmentLength = baseSize + (i < remainder ? 1 : 0);

            const segment = new OneDBoolean<T>(
                segmentLength,
                bufferConstructor,
            );

            for (let j = 0; j < segmentLength; j++) {
                const value = this.get(offset + j);
                segment.set(j, value);
            }

            result.push(segment);
            offset += segmentLength;
        }

        return result;
    }

}

export function isEvalUser(userName: string) {
    return userName.startsWith("eval-heuristic");
}

export function isBaselineUser(userName: string) {
    return userName.startsWith("baseline");
}
