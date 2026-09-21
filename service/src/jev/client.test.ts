import http from "http";
import { AddressInfo } from "net";
import { afterAll, beforeAll, expect, test } from "vitest";
import { JevClient, JevError } from "./client";
import fixture from "./__fixtures__/choice_response.json";

const requestsByPath = new Map<string, number>();
let server: http.Server;
let baseUrl: string;

beforeAll(async () => {
    server = http.createServer((request, response) => {
        const url = request.url ?? "";
        const seen = (requestsByPath.get(url) ?? 0) + 1;
        requestsByPath.set(url, seen);
        request.resume();
        if (url === "/stall") {
            return;
        }
        if (url === "/unauthorised") {
            response.writeHead(401).end("no key");
            return;
        }
        if (url === "/flaky" && seen === 1) {
            response.writeHead(500).end("try again");
            return;
        }
        response
            .writeHead(200, { "Content-Type": "application/json" })
            .end(JSON.stringify(fixture));
    });
    await new Promise<void>((resolve) => server.listen(0, resolve));
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterAll(() => {
    server.closeAllConnections();
    server.close();
});

function clientFor(route: string, deadlineMs: number) {
    return new JevClient({
        apiKey: "test",
        model: "typesafe/jev-1.13",
        deadlineMs,
        endpoint: `${baseUrl}${route}`,
    });
}

const CRITERIA = { cell_0: "move 1", cell_1: "move 2", cell_4: "switch 2" };

test("a prompt server returns the typed result and accrues usage", async () => {
    const client = clientFor("/ok", 2000);
    const result = await client.choose({ turn: 1 }, "choose", CRITERIA);
    expect(result.choice).toBe(fixture.answers.action.choice);
    expect(result.probabilities).toEqual(fixture.answers.action.probabilities);
    expect(result.modelEcho).toBe(fixture.model);
    expect(client.inputTokens).toBe(fixture.usage.input_tokens);
    expect(client.costUsd).toBeCloseTo(fixture.usage.cost, 12);
    expect(client.calls).toBe(1);
});

test("the deadline aborts a stalled server and is not retried", async () => {
    const client = clientFor("/stall", 150);
    const started = Date.now();
    await expect(client.choose({}, "choose", CRITERIA)).rejects.toThrow(
        /deadline/,
    );
    expect(Date.now() - started).toBeLessThan(1000);
    expect(requestsByPath.get("/stall")).toBe(1);
});

test("a 500 is retried once and then succeeds", async () => {
    const client = clientFor("/flaky", 2000);
    const result = await client.choose({}, "choose", CRITERIA);
    expect(result.choice).toBe(fixture.answers.action.choice);
    expect(requestsByPath.get("/flaky")).toBe(2);
});

test("a 401 is never retried", async () => {
    const client = clientFor("/unauthorised", 2000);
    const failure = await client
        .choose({}, "choose", CRITERIA)
        .catch((error) => error);
    expect(failure).toBeInstanceOf(JevError);
    expect(failure.retryable).toBe(false);
    expect(requestsByPath.get("/unauthorised")).toBe(1);
});
