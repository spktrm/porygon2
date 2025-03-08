import WebSocket from "ws";
import axios from "axios";

import { Player } from "../server/player";
import { ObjectReadWriteStream } from "@pkmn/streams";
import { recvFnType, sendFnType } from "../server/types";
import { Action, GameState } from "../../protos/service_pb";
import { TaskQueueSystem } from "../server/utils";

async function ActionFromResponse(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
            try {
                const stateBinary = state.serializeBinary();
                gameState.setState(stateBinary);
            } catch (e) {
                console.log(state.toObject());
                console.log(e);
            }
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
            Math.floor(2 ** 16 * Math.random()),
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

const welcomeMessage =
    "You are fighting an AI in development - trained with reinforcement learning. If you have any questions, please message spktrm#6133 on discord. glfh :)";

/**
 * Class representing a Pokémon Showdown battle
 */
class Battle {
    private battleId: string;
    private ws: WebSocket;
    stream: ClientStream;
    prevMessage: string | undefined;
    active: boolean;

    constructor(roomId: string, ws: WebSocket) {
        this.battleId = roomId;
        this.ws = ws;
        this.active = true;

        this.prevMessage = undefined;

        this.ws.send(`${this.battleId}|/timer on`);
        this.ws.send(`${this.battleId}|${welcomeMessage}`);
        this.stream = new ClientStream({
            roomId,
            choose: (message: string) => {
                this.ws.send(`${this.battleId}|/choose ${message}`);
                this.prevMessage = message;
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
        this.stream.write(message);
    }

    send(message: string): void {
        const toSend = `${this.battleId}|/choose ${message}`;
        this.ws.send(toSend);
        this.prevMessage = message;
    }
    /**
     * Check if the battle is active
     * @returns Whether the battle is active
     */
    public isActive(): boolean {
        return !this.stream.player.done;
    }

    /**
     * Get the battle ID
     * @returns The battle ID
     */
    public getBattleId(): string {
        return this.battleId;
    }
}

interface BotConfig {
    username: string;
    password: string;
    format: string;
    team?: string;
    maxConcurrentBattles: number; // Changed from maxSearches to maxConcurrentBattles
    serverUrl: string;
    secure: boolean;
}

interface BotState {
    activeBattles: number; // Track current active battles
    rooms: Set<string>;
    challenges: Map<string, string>; // challenger name -> format
    battles: Map<string, Battle>;
}

export class ShowdownBot {
    private config: BotConfig;
    private state: BotState;
    private ws: WebSocket | null = null;
    private challstr: string = "";
    private connected: boolean = false;
    private loggedIn: boolean = false;

    constructor(config: BotConfig) {
        this.config = config;
        this.state = {
            activeBattles: 0,
            rooms: new Set(),
            challenges: new Map(),
            battles: new Map(),
        };
    }

    public start(): void {
        this.connect();
    }

    private connect(): void {
        const protocol = this.config.secure ? "wss" : "ws";
        const serverUrl = `${protocol}://${this.config.serverUrl}/showdown/websocket`;

        console.log(`Connecting to ${serverUrl}...`);

        this.ws = new WebSocket(serverUrl);

        this.ws.on("open", () => {
            console.log("Connection established.");
            this.connected = true;
        });

        this.ws.on("message", (data) => {
            this.handleMessage(data);
        });

        this.ws.on("close", () => {
            console.log(
                "Connection closed. Attempting to reconnect in 10 seconds...",
            );
            this.connected = false;
            this.loggedIn = false;

            setTimeout(() => {
                this.connect();
            }, 10000);
        });

        this.ws.on("error", (error) => {
            console.error("WebSocket error:", error.message);
        });
    }

    private handleMessage(data: WebSocket.RawData): void {
        const message = data.toString();
        const lines = message.split("\n");

        if (message.startsWith(">")) {
            const roomId = lines[0].slice(1);
            if (!this.state.battles.has(roomId)) {
                this.state.battles.set(roomId, new Battle(roomId, this.ws!));
                this.state.activeBattles++; // Increment active battles when a new battle starts
            }

            const battle = this.state.battles.get(roomId);
            if (!battle) {
                throw new Error("Battle not found.");
            }

            battle.receive(message);

            const battleIsWon = message.includes("|win|");
            const battleIsTie = message.includes("|tie|");
            if (battleIsWon || battleIsTie) {
                // Clean up battle
                if (this.state.battles.has(roomId)) {
                    this.state.activeBattles--; // Decrement active battles when a battle ends
                }

                if (this.state.rooms.has(roomId)) {
                    this.state.rooms.delete(roomId);
                }

                // Search for another battle if we have room for more
                this.checkAndSearchForBattles(); // Check if we can search for more battles
            }
        } else {
            for (const line of lines) {
                if (!line || line === "") continue;

                if (line.startsWith("|")) {
                    this.parseLine(line);
                }
            }
        }
    }

    private parseLine(line: string): void {
        const parts = line.slice(1).split("|");
        const messageType = parts[0];
        console.log(line);

        switch (messageType) {
            case "challstr":
                this.challstr = parts.slice(1).join("|");
                this.login();
                break;

            case "updateuser": {
                const username = parts[1].trim();
                const namedStatus = parts[2];

                if (
                    username.toLowerCase() ===
                        this.config.username.toLowerCase() &&
                    namedStatus === "1"
                ) {
                    console.log(`Successfully logged in as ${username}`);
                    this.loggedIn = true;

                    // Start searching for battles if configured
                    this.checkAndSearchForBattles(); // Modified to check first
                }
                break;
            }

            case "pm": {
                const sender = parts[1].trim();
                const receiver = parts[2].trim();
                const message = parts.slice(3).join("|");

                // Handle challenge requests
                if (
                    message.startsWith("/challenge") &&
                    receiver.toLowerCase() ===
                        this.config.username.toLowerCase()
                ) {
                    const challengeParts = message.split("|");
                    const format = challengeParts[1];

                    // Store the challenge
                    this.state.challenges.set(sender, format);

                    // Accept the challenge if we haven't reached the concurrent limit
                    this.checkAndAcceptChallenge(sender); // Modified to check first
                }
                break;
            }

            case "updatechallenges":
                try {
                    const challengesData = JSON.parse(parts[1]);

                    // Process incoming challenges
                    if (challengesData.challengesFrom) {
                        for (const challenger in challengesData.challengesFrom) {
                            const format =
                                challengesData.challengesFrom[challenger];

                            // Store the challenge
                            this.state.challenges.set(challenger, format);

                            // Accept the challenge if we haven't reached the concurrent limit
                            this.checkAndAcceptChallenge(challenger); // Modified to check first
                        }
                    }
                } catch (error) {
                    console.error("Error parsing challenges:", error);
                }
                break;

            case "updatesearch": {
                this.checkAndSearchForBattles();
                break;
            }

            case "init": {
                // Track room init
                if (parts[1] === "battle") {
                    const roomId = parts[2] || "";
                    this.state.rooms.add(roomId);
                }
                break;
            }

            case "deinit": {
                // Track room init
                const roomId = parts[2] || "";
                if (this.state.battles.has(roomId)) {
                    this.state.battles.delete(roomId);
                }
                break;
            }
        }
    }

    private async login(): Promise<void> {
        if (!this.challstr) {
            console.error("No challstr received. Cannot login.");
            return;
        }

        try {
            const response = await axios.post(
                "https://play.pokemonshowdown.com/action.php",
                {
                    act: "login",
                    name: this.config.username,
                    pass: this.config.password,
                    challstr: this.challstr,
                },
                {
                    headers: {
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Sec-Fetch-Mode": "cors",
                    },
                },
            );

            const responseData = response.data;

            if (
                typeof responseData === "string" &&
                responseData.startsWith("]")
            ) {
                const json = JSON.parse(responseData.slice(1));

                if (json.assertion) {
                    this.send(
                        `|/trn ${this.config.username},0,${json.assertion}`,
                    );
                } else {
                    console.error("No assertion received. Login failed.");
                }
            } else {
                console.error("Invalid login response:", responseData);
            }
        } catch (error) {
            console.error(
                "Login error:",
                error instanceof Error ? error.message : error,
            );
        }
    }

    private checkAndSearchForBattles(): void {
        if (!this.loggedIn) {
            console.log("Not logged in. Cannot search for battles.");
            return;
        }

        // Check if we can start more battles
        const totalActiveBattles = this.state.activeBattles;

        if (totalActiveBattles >= this.config.maxConcurrentBattles) {
            console.log(
                `Maximum number of concurrent battles reached (${totalActiveBattles}/${this.config.maxConcurrentBattles}).`,
            );
            return;
        }

        // Set the team if provided
        if (this.config.team) {
            this.send(`|/utm ${this.config.team}`);
        }

        // Search for a battle
        this.send(`|/search ${this.config.format}`);

        console.log(
            `Searching for battle. Current active: ${this.state.activeBattles}, max: ${this.config.maxConcurrentBattles}`,
        );
    }

    private checkAndAcceptChallenge(challenger: string): void {
        if (!this.loggedIn) {
            console.log("Not logged in. Cannot accept challenges.");
            return;
        }

        const format = this.state.challenges.get(challenger);

        if (!format) {
            console.error(`No challenge found from ${challenger}`);
            return;
        }

        if (format !== this.config.format) {
            console.log(
                `Rejecting challenge from ${challenger} in format ${format} (not ${this.config.format})`,
            );
            this.send(`|/reject ${challenger}`);
            this.state.challenges.delete(challenger);
            return;
        }

        // Check if we can start more battles
        const totalActiveBattles = this.state.activeBattles;

        if (totalActiveBattles >= this.config.maxConcurrentBattles) {
            console.log(
                `Maximum number of concurrent challenges reached (${totalActiveBattles}/${this.config.maxConcurrentBattles}). Rejecting challenge from ${challenger}.`,
            );
            this.send(`|/reject ${challenger}`);
            this.state.challenges.delete(challenger);
            return;
        }

        // Set the team if provided
        if (this.config.team) {
            this.send(`|/utm ${this.config.team}`);
        }

        // Accept the challenge
        this.send(`|/accept ${challenger}`);

        console.log(
            `Accepted challenge from ${challenger} in format: ${format}. Current active battles: ${this.state.activeBattles}/${this.config.maxConcurrentBattles}`,
        );

        // Remove the challenge
        this.state.challenges.delete(challenger);
    }

    private send(message: string): void {
        if (!this.connected || !this.ws) {
            console.error("Not connected. Cannot send message.");
            return;
        }

        this.ws.send(message);
    }
}

// Example usage:
if (require.main === module) {
    const config: BotConfig = {
        username: "asdf234fae",
        password: "asdf234fae",
        format: "gen3randombattle",
        maxConcurrentBattles: 5, // Changed from maxSearches to maxConcurrentBattles
        serverUrl: "localhost:8000", // Change to 'sim.smogon.com' for main server
        secure: false, // Set to true for main server
        // serverUrl: "sim3.psim.us", // Use 'localhost' for local server
        // secure: true, // Use true for wss:// (usually with play.pokemonshowdown.com)
    };

    const bot = new ShowdownBot(config);
    bot.start();
}
