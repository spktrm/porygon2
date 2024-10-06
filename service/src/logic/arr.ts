type TypedArray =
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
