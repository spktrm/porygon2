export class TaskQueueSystem<T> {
    private results: Map<string, Promise<T>> = new Map();
    private resolvers: Map<string, (value: T) => void> = new Map();

    // Method to generate a unique key for each task
    private generateKey(): string {
        return Math.random().toString(36).substr(2, 9);
    }

    // Method to create a job and return a key
    public createJob(): string {
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
    public submitResult(key: string, result: T): void {
        if (this.resolvers.has(key)) {
            this.resolvers.get(key)!(result);
            this.resolvers.delete(key);
        } else {
            throw new Error("Invalid job key");
        }
    }

    // Method to await the result of a completed job
    public async getResult(key: string): Promise<T> {
        const resultPromise = this.results.get(key);
        if (resultPromise) {
            const result = await resultPromise;
            this.results.delete(key);
            return result;
        } else {
            throw new Error("Invalid job key");
        }
    }

    public allDone(): boolean {
        return this.resolvers.size === 0 && this.results.size === 0;
    }
}
