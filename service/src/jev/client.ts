export const OPENROUTER_DECISIONS_URL =
    "https://openrouter.ai/api/alpha/decisions";

const RETRY_DELAY_MS = 1000;
const RATE_LIMITED = 429;
const SERVER_ERROR_FLOOR = 500;

export interface JevChoiceResult {
    choice: string;
    probabilities: Record<string, number>;
    confidence: number;
    modelEcho: string;
    inputTokens: number;
    costUsd: number;
    latencyMs: number;
}

export interface JevClientOptions {
    apiKey: string;
    model: string;
    deadlineMs: number;
    endpoint?: string;
}

export class JevError extends Error {
    retryable: boolean;

    constructor(message: string, retryable: boolean) {
        super(message);
        this.retryable = retryable;
    }
}

function delay(millis: number) {
    return new Promise((resolve) => setTimeout(resolve, millis));
}

export class JevClient {
    calls = 0;
    inputTokens = 0;
    costUsd = 0;
    modelEcho: string | undefined;

    private readonly options: JevClientOptions;
    private readonly endpoint: string;

    constructor(options: JevClientOptions) {
        this.options = options;
        this.endpoint = options.endpoint ?? OPENROUTER_DECISIONS_URL;
    }

    /**
     * One retry, and only where a second attempt can differ: a 5xx, a rate
     * limit or a dropped connection. A 4xx is the request's own fault, and a
     * deadline abort is left alone so a slow jev is reported rather than
     * quietly given twice the clock.
     */
    async choose(
        state: unknown,
        instructions: string,
        criteria: Record<string, string>,
    ): Promise<JevChoiceResult> {
        try {
            return await this.attempt(state, instructions, criteria);
        } catch (error) {
            if (!(error instanceof JevError) || !error.retryable) {
                throw error;
            }
        }
        await delay(RETRY_DELAY_MS);
        return await this.attempt(state, instructions, criteria);
    }

    private async attempt(
        state: unknown,
        instructions: string,
        criteria: Record<string, string>,
    ): Promise<JevChoiceResult> {
        const controller = new AbortController();
        const timer = setTimeout(
            () => controller.abort(),
            this.options.deadlineMs,
        );
        const started = Date.now();
        let response: Response;
        try {
            response = await fetch(this.endpoint, {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${this.options.apiKey}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    model: this.options.model,
                    state,
                    questions: {
                        action: { type: "choice", instructions, criteria },
                    },
                }),
                signal: controller.signal,
            });
        } catch (error) {
            if (controller.signal.aborted) {
                throw new JevError(
                    `jev deadline of ${this.options.deadlineMs} ms exceeded`,
                    false,
                );
            }
            throw new JevError(`jev network error: ${String(error)}`, true);
        } finally {
            clearTimeout(timer);
        }
        if (!response.ok) {
            const retryable =
                response.status >= SERVER_ERROR_FLOOR ||
                response.status === RATE_LIMITED;
            throw new JevError(
                `jev returned ${response.status}: ${(await response.text()).slice(0, 300)}`,
                retryable,
            );
        }
        const body = await response.json();
        const answer = body?.answers?.action;
        if (
            answer === undefined ||
            typeof answer.choice !== "string" ||
            typeof answer.probabilities !== "object"
        ) {
            throw new JevError(
                `jev response carries no choice answer: ${JSON.stringify(body).slice(0, 300)}`,
                false,
            );
        }
        const inputTokens = Number(body.usage?.input_tokens ?? 0);
        const costUsd = Number(body.usage?.cost ?? 0);
        this.calls += 1;
        this.inputTokens += inputTokens;
        this.costUsd += costUsd;
        this.modelEcho = body.model;
        return {
            choice: answer.choice,
            probabilities: answer.probabilities,
            confidence: answer.confidence,
            modelEcho: body.model,
            inputTokens,
            costUsd,
            latencyMs: Date.now() - started,
        };
    }
}
