export class TwoDBoolArray {
    rows: number;
    cols: number;
    data: Uint8Array;

    constructor(rows: number, cols: number, buffer?: Uint8Array) {
        this.rows = rows;
        this.cols = cols;

        if (!buffer) {
            this.data = new Uint8Array(Math.ceil((rows * cols) / 8));
            this.data.fill(0);
        } else {
            this.data = buffer;
        }
    }

    set(row: number, col: number) {
        const ind = row * this.cols + col;
        const part = Math.floor(ind / 8);
        const rem = ind % 8;
        this.data[part] |= 1 << rem;
    }

    get buffer() {
        return this.data;
    }
}
