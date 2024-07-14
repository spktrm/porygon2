import WebSocket from "ws";
import { TaskQueueSystem } from "../utils";
import { Action } from "../../protos/action_pb";
import { StreamHandler } from "../logic/handler";
import { actionIndexMapping } from "../server/game";

const offline = "localhost";
const online = "sim.smogon.com";

const queueSystem = new TaskQueueSystem<Action>();

async function ActionFromResponse(response: Response): Promise<Action> {
    const { pi, v, action: actionIndex, prev_pi } = await response.json();
    console.log(pi);
    console.log(v);
    const action = new Action();
    action.setIndex(actionIndex);
    return action;
}

function stringToUniqueInt(str: string): number {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
        hash = (hash * 33) ^ str.charCodeAt(i);
    }
    return hash >>> 0; // Ensure the hash is a positive integer
}

class Battle {
    roomId: string;
    private ws: WebSocket;
    handler: StreamHandler;

    prevMessage: string | undefined;

    constructor(roomId: string, ws: WebSocket) {
        this.roomId = roomId;
        this.ws = ws;

        this.prevMessage = undefined;

        this.handler = new StreamHandler({
            gameId: stringToUniqueInt(roomId),
            isTraining: true,
            sendFn: async (state) => {
                const jobKey = queueSystem.createJob();
                state.setKey(jobKey);
                const response = await fetch("http://127.0.0.1:8080/predict", {
                    method: "POST",
                    body: state.serializeBinary(),
                });
                const action = await ActionFromResponse(response);
                action.setKey(jobKey);
                queueSystem.submitResult(jobKey, action);
                return jobKey;
            },
            recvFn: async (key: string) => {
                return await queueSystem.getResult(key);
            },
        });
    }

    public async receive(message: string): Promise<void> {
        if (
            this.prevMessage !== undefined &&
            message.includes(`class="message-throttle-notice"`)
        ) {
            this.send(this.prevMessage);
        }
        const action = await this.handler.ingestChunk(message);
        if (action !== undefined) {
            const actionIndex = action.getIndex();
            const actionString =
                actionIndexMapping[actionIndex] ?? "choose default";
            const toSend = `/${actionString}|${this.handler.rqid}`;
            console.log(toSend);
            this.send(toSend);
        }
    }

    private send(message: string): void {
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
    private serverUrl: string = `ws://${offline}:8000/showdown/websocket`;
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
        for (let line of lines) {
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
                }
                await battle.receive(lines.slice(1).join("\n"));
                return;
            } else if (line.startsWith("|win|") || line.startsWith("|tie|")) {
                this.send(`|/search gen3randombattle`);
            }
        }
    }

    private login(challstr: string): void {
        const loginDetails: { [k: string]: any } = {
            act: "login",
            name: this.username,
            pass: this.password,
            challstr,
        };

        const requestOptions = {
            hostname: "play.pokemonshowdown.com",
            path: "/action.php",
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

        const https = require("https");
        const req = https.request(requestOptions, (res: any) => {
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
