import * as https from "https";
import WebSocket from "ws";
import { Protocol } from "@pkmn/protocol";
import { Action, Actions, LoginDetails } from "@pkmn/login";
import { AnyObject, Teams, TeamValidator } from "@pkmn/sim";
import * as dotenv from "dotenv";

import { ObjectReadWriteStream } from "@pkmn/streams";

import * as path from "path";
import { TrainablePlayerAI } from "../server/runner";
import { Action as ProtoAction, StepRequest } from "../../protos/service_pb";
import { generateTeamFromArray } from "../server/state";

// Before anything reads process.env: RL_SERVER_URL below is one of its keys.
const dotenvResult = dotenv.config({
    path: path.resolve(__dirname, "../../../.env"),
});
if (dotenvResult.error) {
    console.error(
        "Error loading .env file. Please ensure it exists and is configured correctly.",
        dotenvResult.error,
    );
    process.exit(1);
}

const RL_SERVER_URL = process.env.RL_SERVER_URL || "http://localhost:8001";

const server = "ws://localhost:8000/showdown/websocket";
// Derived from the server rather than set beside it: the timer must never be
// off on a ladder we do not own (pokeagent, the main server), and on a local
// server the opponent is a person at the keyboard who should not be timed.
const timerRequired = !["localhost", "127.0.0.1", "[::1]"].includes(
    new URL(server).hostname,
);
const MAX_BATTLES = 5;
const smogonFormat = "gen9randombattle";

function cookieFetch(action: Action, cookie?: string): Promise<string> {
    const headers = cookie
        ? { Cookie: cookie, ...action.headers }
        : action.headers;

    return new Promise<string>((resolve, reject) => {
        let buf = "";

        const req = https.request(
            action.url,
            { method: action.method, headers },
            (res) => {
                if (res.statusCode !== 200) {
                    return reject(new Error(`HTTP ${res.statusCode}`));
                }
                res.on("data", (d: Buffer) => {
                    buf += d;
                });
                res.on("end", () => resolve(buf));
            },
        );

        req.on("error", reject);
        req.write(action.data);
        req.end();
    });
}

function processAssertion(details: { username: string }, assertion: string) {
    if (assertion.slice(0, 14).toLowerCase() === "<!doctype html") {
        // some sort of MitM proxy - ignore it
        const endIndex = assertion.indexOf(">");
        if (endIndex > 0) assertion = assertion.slice(endIndex + 1);
    }

    if (assertion.startsWith("\r")) assertion = assertion.slice(1);
    if (assertion.startsWith("\n")) assertion = assertion.slice(1);
    if (assertion.includes("<")) {
        throw new Error(
            "Something appears to be interfering with the connection",
        );
    } else if (assertion === ";") {
        throw new Error(`Authentication required for '${details.username}'`);
    } else if (assertion === ";;@gmail") {
        throw new Error(
            `Authentication from Google required for '${details.username}'`,
        );
    } else if (assertion.startsWith(";;")) {
        throw new Error(`Invalid name: ${assertion.substr(2)}`);
    } else if (assertion.includes("\n") || !assertion) {
        throw new Error(
            "Something appears to be interfering with the connection",
        );
    }

    return assertion;
}

function sanitizeRoomId(roomId: string): string {
    const parts = roomId.split("-");
    if (parts.length >= 4) {
        parts.pop();
        roomId = parts.join("-");
    }
    return roomId;
}

class Connection {
    private ws!: WebSocket;

    open(callback: (data: string) => void): void {
        this.ws = new WebSocket(server);

        this.ws.onmessage = ({ data }) => callback(data.toString());
        this.ws.onopen = () => {
            console.log(`Connected to ${this.ws.url}`);
        };
        this.ws.onclose = (e) => {
            const clean = e.wasClean ? " cleanly " : " ";
            const reason = e.reason ? `: ${e.reason}` : "";
            console.log(
                `Disconnected${clean}from ${this.ws.url} with ${e.code}${reason}`,
            );
        };
        this.ws.onerror = (e: WebSocket.ErrorEvent) => {
            const msg: string | undefined = e.message;
            if (msg === "TIMEOUT") return;
            console.error(`Connection error${msg ? `: ${msg}` : ""}`);
        };
    }

    close(onClosed?: () => void): void {
        if (onClosed) this.ws.once("close", onClosed);
        this.ws.close();
    }

    send(message: string): void {
        this.ws.send(message);
    }
}

function safeJSON(data?: string) {
    if (!data || data.length < 1) throw new Error("No data received");
    if (data[0] === "]") data = data.substr(1);
    return JSON.parse(data);
}

class ClientStream extends ObjectReadWriteStream<string> {
    constructor(options = {}) {
        super(options);
    }

    _write(message: string) {
        this.push(message);
    }
}

class Battle {
    private battleId: string;
    private conn: Connection;
    private username: string;
    private readonly team: string | undefined;
    stream: ObjectReadWriteStream<string>;
    player: TrainablePlayerAI;
    prevMessage: string | undefined;
    active: boolean;
    private timerRequested: boolean;
    private ended: boolean;

    constructor(roomId: string, conn: Connection, username: string) {
        this.battleId = roomId;
        this.conn = conn;
        this.active = true;
        this.username = username;
        this.prevMessage = undefined;
        this.timerRequested = false;
        this.ended = false;

        this.stream = new ClientStream();
        this.player = new TrainablePlayerAI(
            this.username,
            this.stream,
            {},
            false,
        );
        this.player.choose = (choice: string) => {
            this.conn.send(
                `${this.getBattleId()}|/choose ${choice}|${this.player.rqid}`,
            );
        };
        this.player.start();
    }

    updateBattleId(roomId: string) {
        this.battleId = roomId;
    }

    public async start() {
        while (true) {
            const state = await this.player.receiveEnvironmentState();
            if (!this.player.done) {
                const response = await fetch(`${RL_SERVER_URL}/step`, {
                    method: "POST",
                    body: state.serializeBinary(),
                });
                if (!response.ok) {
                    throw new Error(
                        `RL server /step returned ${response.status}: ${await response.text()}`,
                    );
                }

                const { cell } = await response.json();
                if (!Number.isInteger(cell)) {
                    throw new Error(`RL server /step returned cell ${cell}`);
                }
                const stepRequest = new StepRequest();

                const protoAction = new ProtoAction();
                protoAction.setCell(cell);

                stepRequest.setAction(protoAction);
                stepRequest.setRqid(state.getRqid());
                this.player.submitStepRequest(stepRequest);
            } else {
                break;
            }
        }
    }

    public async receive(message: string): Promise<void> {
        if (this.ended) return;

        // The sim's player stream carries no ">roomid" header line, and the
        // player's chunk-level checks (|error|) assume there is none.
        let chunk = message;
        if (chunk.startsWith(">")) {
            chunk = chunk.slice(chunk.indexOf("\n") + 1);
        }
        const cmds = chunk
            .split("\n")
            .map((line) => line.slice(1).split("|")[0]);

        // The first updatesearch names the room WITHOUT its private suffix and
        // before we have joined it, so a room message is the first point at
        // which the server accepts the command; and whoever turns the timer
        // off afterwards gets it turned back on.
        const timerMissing =
            !this.timerRequested || cmds.includes("inactiveoff");
        if (timerRequired && timerMissing) {
            this.timerRequested = true;
            this.conn.send(`${this.battleId}|/timer on`);
        }

        this.stream.write(chunk);

        // The sim ends the player stream with the battle; a Showdown room
        // does not, and the player only leaves its loop on the next chunk or
        // the end of the stream.
        if (cmds.includes("win") || cmds.includes("tie")) {
            this.ended = true;
            this.stream.pushEnd();
        }
    }

    public getBattleId(): string {
        return this.battleId;
    }

    public forfeit() {
        this.conn.send(`${this.battleId}|/forfeit`);
    }

    public leave() {
        this.conn.send(`${this.battleId}|/leave`);
    }
}

interface SearchState {
    searching: string[];
    games: { [k: string]: string } | null;
}

class BattleStorage {
    private battles: { [k: string]: Battle };

    constructor() {
        this.battles = {};
    }

    addBattle(roomId: string, battle: Battle) {
        const battleId = sanitizeRoomId(roomId);
        this.battles[battleId] = battle;
    }

    getBattle(roomId: string): Battle | undefined {
        const battleId = sanitizeRoomId(roomId);
        const battle = this.battles[battleId];
        if (battle && battle.getBattleId() !== roomId) {
            battle.updateBattleId(roomId);
        }
        return battle;
    }

    removeBattle(battleId: string) {
        delete this.battles[battleId];
    }
}

class User {
    private username?: string;
    private searchState?: SearchState;
    private battles: BattleStorage;
    private teams: string[];
    private searchUpdated: boolean;
    private numBattles: number;
    private currentFormat?: string;

    constructor(private readonly connection: Connection) {
        this.searchState = undefined;
        this.battles = new BattleStorage();
        this.teams = [];
        this.searchUpdated = false;
        this.numBattles = 0;
        this.currentFormat = undefined;
    }

    get isLoggedIn(): boolean {
        return this.username !== undefined;
    }

    createNewBattle(roomId: string) {
        const battle = new Battle(roomId, this.connection, this.username!);
        this.battles.addBattle(roomId, battle);
        battle
            .start()
            .catch((err) => {
                console.error(`Battle ${battle.getBattleId()} failed:`, err);
                battle.forfeit();
            })
            .then(() => {
                battle.leave();
                if (this.numBattles >= MAX_BATTLES) {
                    console.log(
                        "Reached maximum number of battles, logging out.",
                    );
                    this.logout()
                        .catch((err) => console.error("Logout failed:", err))
                        .then(() => {
                            this.connection.close(() => process.exit(0));
                        });
                }
            });
    }

    async receiveBattleData(roomId: string, data: string): Promise<void> {
        if (!this.battles.getBattle(roomId)) {
            this.createNewBattle(roomId);
        }
        await this.battles.getBattle(roomId)?.receive(data);
    }

    async login(details: LoginDetails): Promise<void> {
        const action = {
            method: "POST",
            url: "https://play.pokemonshowdown.com/action.php",
            data: new URLSearchParams({
                act: "login",
                name: details.username,
                pass: details.password ?? "",
                challstr: details.challstr,
            }).toString(),
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
            },
            onResponse: (data?: string) => {
                const response = safeJSON(data) as AnyObject;
                if (response.curuser?.loggedin) {
                    return `|/trn ${details.username},0,${processAssertion(
                        details,
                        response.assertion,
                    )}`;
                } else {
                    console.error(
                        `Login failed: ${response.error || "Unknown error"}`,
                    );
                }
            },
        } as unknown as Action;
        cookieFetch(action).then((response) => {
            const cmd = action.onResponse(response);
            if (cmd) this.send(cmd);
        });
    }

    async upkeep(details: LoginDetails, cookie?: string): Promise<void> {
        const action = Actions.upkeep(details);
        cookieFetch(action, cookie).then((response) => {
            const cmd = action.onResponse(response);
            if (cmd) this.send(cmd);
        });
    }

    async logout(): Promise<void> {
        if (!this.username) return;
        const action = Actions.logout({ username: this.username });
        this.username = undefined;
        const response = await cookieFetch(action);
        const cmd = action.onResponse(response);
        if (cmd) this.send(cmd);
    }

    async send(message: string): Promise<void> {
        console.log(`Sending: ${message}`);
        this.connection.send(message);
    }

    async cancelSearch(): Promise<void> {
        this.send(`|/cancelsearch`);
    }

    async search(format: string): Promise<void> {
        if (!format.includes("random")) {
            while (true) {
                const response = await fetch(`${RL_SERVER_URL}/reset`, {
                    method: "POST",
                    body: JSON.stringify({ format }),
                });
                const modelOutput = await response.json();
                const team = generateTeamFromArray(modelOutput.packed_team)!;
                const validator = new TeamValidator(format);
                const errors = validator.validateTeam(Teams.unpack(team));
                if (errors === null) {
                    this.teams.push(team);
                    await this.send(`|/utm ${team}`);
                    break;
                } else {
                    console.log(
                        `Team validation for ${team} failed: ${errors}`,
                    );
                }
            }
        }
        await this.send(`|/search ${format}`);
        this.currentFormat = format;
    }

    async updateSearch(
        searchState: SearchState | undefined = undefined,
    ): Promise<void> {
        if (searchState !== undefined) {
            this.searchState = searchState;
        } else {
            searchState = this.searchState;
        }
        if (searchState === undefined) return;

        const { searching, games } = searchState;
        // numBattles counts battles STARTED, so the last battle's trailing
        // games:null must not queue one more that the exit then abandons.
        if (
            searching.length === 0 &&
            games === null &&
            !this.searchUpdated &&
            this.numBattles < MAX_BATTLES
        ) {
            this.search(smogonFormat);
            this.searchUpdated = true;
            return;
        }
        if (games !== null) {
            if (searching.length > 0) {
                this.cancelSearch();
            } else {
                this.searchUpdated = false;
            }
            for (const gameId in games) {
                if (this.battles.getBattle(gameId)) continue;
                this.createNewBattle(gameId);
                this.numBattles++;
            }
        }
    }

    setUsername(name: string): void {
        this.username = name;
    }
}

async function waitForServer(waitTimeout: number = 1000) {
    while (true) {
        try {
            const response = await fetch(`${RL_SERVER_URL}/ping`, {
                method: "GET",
            });
            const pong = await response.text();
            if (pong === "pong") {
                break;
            }
        } catch {
            // Not listening yet.
        }
        console.log("Waiting for RL server to be ready...");
        await new Promise((resolve) => setTimeout(resolve, waitTimeout));
    }
}

waitForServer().then(() => {
    const connection = new Connection();
    const user = new User(connection);

    connection.open((data) => {
        console.log(data);
        if (data.startsWith(">")) {
            const roomId = data.split("\n", 1)[0].slice(1);
            user.receiveBattleData(roomId, data);
            return;
        }

        let searchState = undefined;
        for (const { args } of Protocol.parse(data)) {
            switch (args[0]) {
                case "challstr": {
                    const challstr = args[1];
                    const {
                        SHOWDOWN_USERNAME: username,
                        SHOWDOWN_PASSWORD: password,
                    } = process.env;
                    if (!username || !password) {
                        console.error(
                            "Please set SHOWDOWN_USERNAME and SHOWDOWN_PASSWORD in your .env file.",
                        );
                        connection.close();
                        process.exit(1);
                    }
                    user.login({
                        challstr,
                        username,
                        password,
                    });
                    break;
                }

                case "updateuser": {
                    const username = args[1].trim();
                    const namedStatus = args[2].trim();
                    if (namedStatus === "1") {
                        user.setUsername(username);
                        console.log(`Logged in as '${username}'`);
                    }
                    break;
                }

                case "updatesearch": {
                    searchState = JSON.parse(args[1]);
                    user.updateSearch(searchState);
                    break;
                }

                case "popup": {
                    if (args[1].startsWith("Your team was rejected")) {
                        user.updateSearch();
                        break;
                    }
                }
            }
        }
    });
});
