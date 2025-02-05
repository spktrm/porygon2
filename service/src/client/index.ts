/* eslint-disable @typescript-eslint/no-explicit-any */

import WebSocket from "ws";
import { inspect } from "util";
import { Player } from "../server/player";
import { ObjectReadWriteStream } from "@pkmn/streams";
import { recvFnType, sendFnType } from "../server/types";
import { Action, GameState } from "../../protos/service_pb";
import { TaskQueueSystem } from "../server/utils";
import { request } from "https";

const offline = `localhost:8000`;
const online = "sim3.psim.us";

function stringToUniqueInt(str: string): number {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
        hash = (hash * 33) ^ str.charCodeAt(i);
    }
    return hash >>> 0; // Ensure the hash is a positive integer
}

async function ActionFromResponse(
    modelOutput: Record<string, any>,
): Promise<Action> {
    const { action: actionIndex } = modelOutput;
    const action = new Action();
    action.setValue(actionIndex);
    return action;
}

class ClientStream extends ObjectReadWriteStream<string> {
    debug: boolean;
    roomId: string | undefined;
    player: Player;
    tasks: TaskQueueSystem<Action>;

    constructor(
        options: {
            debug?: boolean;
            roomId?: string;
            choose?: (action: string) => void;
        } = {},
    ) {
        super();
        this.debug = !!options.debug;
        this.roomId = options?.roomId;

        this.tasks = new TaskQueueSystem();

        const sendFn: sendFnType = async (player) => {
            const gameState = new GameState();
            const state = player.createState();
            gameState.setState(state.serializeBinary());
            const playerId = 0;
            let rqid = -1;
            if (!state.getInfo()!.getDone()) {
                rqid = this.tasks.createJob();
            }
            gameState.setRqid(rqid);
            gameState.setPlayerId(playerId);
            const response = await fetch("http://127.0.0.1:8001/predict", {
                method: "POST",
                body: state.serializeBinary(),
            });
            const modelOutput = await response.json();
            const action = await ActionFromResponse(modelOutput);
            action.setRqid(rqid);
            if (rqid >= 0) this.tasks.submitResult(rqid, action);
            return rqid;
        };

        const recvFn: recvFnType = async (rqid) => {
            return rqid >= 0 ? this.tasks.getResult(rqid) : undefined;
        };

        if (!options.roomId) {
            throw new Error("Options must have roomId");
        }

        this.player = new Player(
            0,
            stringToUniqueInt(options?.roomId),
            this,
            0,
            sendFn,
            recvFn,
            null,
            options.choose!,
        );
        this.player.start();
    }

    _write(message: string) {
        this.push(message);
    }
}

class Battle {
    roomId: string;
    private ws: WebSocket;
    stream: ClientStream;
    prevMessage: string | undefined;

    constructor(roomId: string, ws: WebSocket) {
        this.roomId = roomId;
        this.ws = ws;

        this.prevMessage = undefined;

        const stream = new ClientStream({
            roomId,
            choose: (message: string) => {
                this.ws.send(`${this.roomId}|/choose ${message}`);
                this.prevMessage = message;
            },
        });
        this.stream = stream;
    }

    public async receive(message: string): Promise<void> {
        if (
            this.prevMessage !== undefined &&
            message.includes(`class="message-throttle-notice"`)
        ) {
            this.send(this.prevMessage);
        }
        this.stream.write(message);
    }

    send(message: string): void {
        const toSend = `${this.roomId}|${message}`;
        this.ws.send(toSend);
        this.prevMessage = message;
    }
}

class BattleManager {
    battles: { [roomId: string]: Battle } = {};
    private ws: WebSocket;

    constructor(ws: WebSocket) {
        this.ws = ws;
    }

    public addBattle(roomId: string): [boolean, Battle] {
        let isNew = false;
        if (!this.battles[roomId]) {
            this.battles[roomId] = new Battle(roomId, this.ws);
            isNew = true;
        }
        return [isNew, this.battles[roomId]];
    }

    public getBattle(roomId: string): Battle | undefined {
        return this.battles[roomId];
    }

    public removeBattle(roomId: string): void {
        delete this.battles[roomId];
    }
}

class PokemonShowdownBot {
    private ws: WebSocket;
    private serverUrl: string = `ws://${offline}/showdown/websocket`;
    private username: string = "YourUsername";
    private password: string | undefined = undefined;
    private battleManager: BattleManager;

    constructor(username: string, password: string | undefined = undefined) {
        this.username = username;
        this.password = password;
        this.ws = new WebSocket(this.serverUrl);
        this.battleManager = new BattleManager(this.ws);
        this.setupListeners();
    }

    private setupListeners(): void {
        this.ws.on("open", () => {
            console.log("Connected to Pokémon Showdown.");
            this.send("|/cmd rooms");
        });

        this.ws.on("message", async (data) =>
            this.handleMessage(data.toString()),
        );

        this.ws.on("close", () => console.log("Disconnected from the server."));
    }

    private async handleMessage(message: string): Promise<void> {
        console.log("Received:", message);
        const lines = message.split("\n");
        for (const line of lines) {
            if (line.startsWith("|challstr|")) {
                this.login(line.slice("|challstr|".length));
            } else if (line.startsWith("|updatesearch|")) {
                this.handleUpdateSearch(line);
            } else if (line.startsWith("|pm|")) {
                this.handlePrivateMessage(line);
            } else if (line.startsWith(">")) {
                const roomId = line.slice(1).trim();
                const [isNew, battle] = this.battleManager.addBattle(roomId);
                if (isNew) {
                    if (this.serverUrl.includes(online)) {
                        this.send(`${roomId}|/timer on`);
                    }
                    // this.send(`${roomId}|glhf`);
                }
                await battle.receive(lines.slice(1).join("\n"));
                if (lines[1].startsWith("|request|")) {
                    try {
                        const data = JSON.parse(lines[1].split("|")[2]);
                        console.log(inspect(data, false, null, true));
                        // eslint-disable-next-line @typescript-eslint/no-unused-vars
                    } catch (err) {
                        /* empty */
                    }
                }
                return;
            } else if (line.startsWith("|win|") || line.startsWith("|tie|")) {
                this.send(`|/search gen3randombattle`);
            }
        }
    }

    private login(challstr: string): void {
        const loginDetails: { [k: string]: any } = {
            name: this.username,
            pass: this.password,
            challstr,
        };

        const requestOptions = {
            hostname: "play.pokemonshowdown.com",
            path: "/api/login",
            method: "POST",
            headers: {
                "Sec-Fetch-Mode": "cors",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        };

        const postData = Object.keys(loginDetails)
            .map((key) => {
                return (
                    encodeURIComponent(key) +
                    "=" +
                    encodeURIComponent(loginDetails[key])
                );
            })
            .join("&");

        const req = request(requestOptions, (res: any) => {
            let data = "";
            let payload: { [k: string]: any };
            res.on("data", (chunk: any) => (data += chunk));
            res.on("end", () => {
                if (data.charAt(0) === "]") {
                    payload = JSON.parse(data.slice(1));
                    if (payload.actionsuccess) {
                        console.log("Logged in successfully!");
                        this.send(
                            `|/trn ${this.username},0,${payload.assertion}`,
                        );
                    } else {
                        console.log("Login failed:", data);
                    }
                }
            });
        });

        req.on("error", (e: any) =>
            console.error(`Request error: ${e.message}`),
        );
        req.write(postData);
        req.end();
    }

    private handleUpdateSearch(line: string): void {
        const parts = line.split("|");
        const data = JSON.parse(parts[2]);
        if (
            data.searching.length === 0 &&
            Object.keys(data.games ?? {}).length < 4
        ) {
            this.searchForBattles();
        }
    }

    private searchForBattles(): void {
        this.send(`|/utm null`);
        this.send(`|/search gen3randombattle`);
    }

    private send(message: string): void {
        console.log("Sending:", message);
        this.ws.send(message);
    }

    private handlePrivateMessage(line: string): void {
        const parts = line.split("|");
        const sender = parts[2];
        const message = parts[4];
        console.log(`PM from ${sender}: ${message}`);
    }
}

new PokemonShowdownBot("asdf234fae", "asdf234fae");
