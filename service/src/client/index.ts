import * as https from "https";
import WebSocket from "ws";
import { Protocol } from "@pkmn/protocol";
import { Action, Actions, LoginDetails } from "@pkmn/login";
import { AnyObject, Teams, TeamValidator } from "@pkmn/sim";
import * as dotenv from "dotenv";

import { ObjectReadWriteStream } from "@pkmn/streams";

import * as path from "path";
import { TrainablePlayerAI } from "../server/runner";
import {
    Action as MultiDiscreteAction,
    StepRequest,
} from "../../protos/service_pb";
import { generateTeamFromIndices } from "../server/state";

const RL_SERVER_URL = process.env.RL_SERVER_URL || "http://localhost:8001";

// const server = "ws://localhost:8000/showdown/websocket";
// const server = "wss://sim3.psim.us/showdown/websocket";
const server = "wss://pokeagentshowdown.com/showdown/websocket";
const MAX_BATTLES = 50; // Maximum number of battles to run in sequence
const smogonFormat = "gen1ou";

function cookieFetch(action: Action, cookie?: string): Promise<string> {
    const headers = cookie
        ? { "Set-Cookie": cookie, ...action.headers }
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

    close(): void {
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

    constructor(
        roomId: string,
        conn: Connection,
        username: string,
        team?: string,
    ) {
        this.battleId = roomId;
        this.conn = conn;
        this.active = true;
        this.username = username;
        this.prevMessage = undefined;
        this.team = team;

        this.conn.send(`${this.battleId}|/timer on`);
        // this.ws.send(`${this.battleId}|${welcomeMessage}`);
        const unpackedTeam = team === undefined ? team : Teams.unpack(team);
        if (unpackedTeam === null) {
            throw new Error("Invalid team provided");
        }
        this.stream = new ClientStream();
        this.player = new TrainablePlayerAI(
            this.username,
            this.stream,
            {},
            false,
            unpackedTeam,
        );
        this.player.choose = (choice: string) => {
            this.conn.send(
                `${this.battleId}|/choose ${choice}|${this.player.rqid}`,
            );
        };
        this.player.start();
    }

    public async start(rateLimit: number = 0) {
        while (true) {
            const state = await this.player.receiveEnvironmentState();
            if (!this.player.done) {
                const response = await fetch(`${RL_SERVER_URL}/step`, {
                    method: "POST",
                    body: state.serializeBinary(),
                });
                // await new Promise((resolve) => setTimeout(resolve, rateLimit));

                const stepResponse = await response.json();
                const stepRequest = new StepRequest();

                const action = new MultiDiscreteAction();
                action.setActionType(stepResponse.action_type);
                action.setMoveSlot(stepResponse.move_slot);
                action.setWildcardSlot(stepResponse.wildcard_slot);
                action.setSwitchSlot(stepResponse.switch_slot);

                stepRequest.setAction(action);
                stepRequest.setRqid(state.getRqid());
                this.player.submitStepRequest(stepRequest);
            } else {
                break;
            }
        }
    }

    public async receive(message: string): Promise<void> {
        this.stream.write(message);
    }

    public getBattleId(): string {
        return this.battleId;
    }

    public leave() {
        this.conn.send(`${this.battleId}|/leave`);
    }
}

interface SearchState {
    searching: string[];
    games: { [k: string]: string } | null;
}

class User {
    private username?: string;
    private searchState?: SearchState;
    private battles: { [k: string]: Battle };
    private teams: string[];
    private searchUpdated: boolean;
    private numBattles: number;

    constructor(private readonly connection: Connection) {
        this.searchState = undefined;
        this.battles = {};
        this.teams = [];
        this.searchUpdated = false;
        this.numBattles = 0;
    }

    get isLoggedIn(): boolean {
        return this.username !== undefined;
    }

    async receiveBattleData(roomId: string, data: string): Promise<void> {
        if (!this.battles[roomId]) {
            const battle = new Battle(roomId, this.connection, this.username!);
            this.battles[roomId] = battle;
            battle.start().then(() => {
                battle.leave();
            });
        }
        await this.battles[roomId].receive(data);
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
        cookieFetch(action).then((response) => {
            const cmd = action.onResponse(response);
            if (cmd) this.send(cmd);
        });
        this.username = undefined;
    }

    async send(message: string): Promise<void> {
        console.log(`Sending: ${message}`);
        this.connection.send(message);
    }

    async cancelSearch(): Promise<void> {
        this.send(`|/cancelsearch`);
    }

    async search(format: string): Promise<void> {
        if (!format.endsWith("randombattle")) {
            while (true) {
                const response = await fetch(`${RL_SERVER_URL}/reset`, {
                    method: "POST",
                    body: JSON.stringify({ format }),
                });
                const modelOutput = await response.json();
                const team = generateTeamFromIndices(
                    format.replace("ou", "_ou_all_formats"),
                    modelOutput.species_indices,
                    modelOutput.packed_set_indices,
                )!;
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
        if (searching.length === 0 && games === null && !this.searchUpdated) {
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
                if (this.battles[gameId]) continue; // Already in a battle
                const team = this.teams.shift();
                if (!team) {
                    console.error("No team available for battle");
                    continue;
                }
                const battle = new Battle(
                    gameId,
                    this.connection,
                    this.username!,
                    team,
                );
                this.battles[gameId] = battle;
                battle.start().then(() => {
                    battle.leave();
                    if (this.numBattles >= MAX_BATTLES) {
                        console.log(
                            "Reached maximum number of battles, logging out.",
                        );
                        this.logout().then(() => {
                            this.connection.close();
                            process.exit(0);
                        });
                    }
                });
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
                // Ensure this URL is correct for your setup
                method: "GET",
            });
            const pong = await response.text();
            if (pong === "pong") {
                break;
            }
        } catch (error) {
            console.log("Waiting for RL server to be ready...");
            await new Promise((resolve) => setTimeout(resolve, waitTimeout));
            continue;
        }
    }
}

waitForServer().then(() => {
    const connection = new Connection();
    const user = new User(connection);

    const result = dotenv.config({
        path: path.resolve(__dirname, "../../../.env"), // Adjust path if your .env is elsewhere
    });
    if (result.error) {
        console.error(
            "Error loading .env file. Please ensure it exists and is configured correctly.",
            result.error,
        );
        // throw result.error; // Or handle more gracefully
        process.exit(1);
    }

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
