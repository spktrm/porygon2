import WebSocket from "ws";
import axios from "axios";

import { Player } from "../server/player";
import { ObjectReadWriteStream } from "@pkmn/streams";
import { recvFnType, sendFnType } from "../server/types";
import { Action, GameState } from "../../protos/service_pb";
import { TaskQueueSystem } from "../server/utils";

/**
 * Configuration interface for the Showdown Bot
 */
interface BotConfig {
    username: string;
    password?: string;
    battleFormat: string;
    maxConcurrentBattles: number;
    server: {
        url: string;
        secure: boolean;
    };
}

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
        const stream = new ClientStream({
            roomId,
            choose: (message: string) => {
                this.ws.send(`${this.battleId}|/choose ${message}`);
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

/**
 * Class representing a Pokémon Showdown bot
 */
class ShowdownBot {
    private config: BotConfig;
    private ws: WebSocket | null = null;
    private loginAttempted: boolean = false;
    private authenticated: boolean = false;
    private challstr: string = "";
    private battles: Map<string, Battle> = new Map();
    private searching: boolean = false;

    /**
     * Create a new Showdown bot
     * @param config The bot configuration
     */
    constructor(config: BotConfig) {
        this.config = config;
    }

    /**
     * Start the bot
     */
    public start() {
        console.log("Starting Pokémon Showdown Bot...");
        this.connect();
        this.runMaintenanceTasks();
    }

    /**
     * Connect to the Pokémon Showdown server
     */
    private connect() {
        const protocol = this.config.server.secure ? "wss" : "ws";
        const serverUrl = `${protocol}://${this.config.server.url}/showdown/websocket`;

        console.log(`Connecting to ${serverUrl}...`);

        this.ws = new WebSocket(serverUrl);

        this.ws.on("open", () => {
            console.log("Connected to Pokémon Showdown server!");
        });

        this.ws.on("message", (data: WebSocket.Data) => {
            this.handleMessage(data.toString());
        });

        this.ws.on("close", () => {
            console.log("Disconnected from Pokémon Showdown server.");
            setTimeout(() => this.connect(), 5000);
        });

        this.ws.on("error", (error) => {
            console.error("WebSocket error:", error);
        });
    }

    /**
     * Handle a message from the server
     * @param message The message to handle
     */
    private handleMessage(message: string) {
        const lines = message.split("\n");

        // Extract the room ID if present
        let roomId = "";
        if (lines[0].startsWith(">")) {
            roomId = lines[0].slice(1);
            lines.shift();
        }

        // Handle global messages or messages in a specific room
        if (roomId === "") {
            // Global messages
            this.handleGlobalMessages(lines.join("\n"));
        } else if (roomId.startsWith("battle-")) {
            // Battle messages
            const battle = this.battles.get(roomId);
            if (battle) {
                try {
                    console.log(message);
                    battle.receive(message);
                } catch (err) {
                    console.log(err);
                }
            }
        }
    }

    /**
     * Handle global messages
     * @param message The message to handle
     */
    private handleGlobalMessages(message: string) {
        const lines = message.split("\n");

        for (const line of lines) {
            if (!line) continue;

            console.log(`Global: ${line}`); // Log all global messages for debugging

            if (line.startsWith("|challstr|")) {
                // Authentication challenge
                this.challstr = line.slice(10);
                console.log("Received authentication challenge.");
                this.login();
            } else if (line.startsWith("|updateuser|")) {
                // User update
                const parts = line.slice(12).split("|");
                const username = parts[0].trim();
                const named = parts[1] === "1";

                if (
                    username.toLowerCase() ===
                    this.config.username.toLowerCase()
                ) {
                    console.log(`Logged in as ${username} (named: ${named})`);
                    this.authenticated = true;

                    // Wait a moment before searching to ensure everything is initialized
                    setTimeout(() => this.searchForBattles(), 1000);
                }
            } else if (line.startsWith("|updatesearch|")) {
                // Search update
                this.handleSearchUpdate(line.slice(14));
            } else if (line.startsWith("|popup|")) {
                // Popup message (usually errors)
                console.log(`Popup: ${line.slice(7)}`);
            } else if (line.startsWith("|formats|")) {
                console.log("Received formats list");
                // You could parse and validate the battle format here
            } else if (line.startsWith("|deinit|")) {
                // Room was closed
                const roomId = line.slice(8);
                console.log(`Room closed: ${roomId}`);
                if (this.battles.has(roomId)) {
                    this.battles.delete(roomId);
                    // Try to search for a new battle
                    setTimeout(() => this.searchForBattles(), 1000);
                }
            }
        }
    }

    /**
     * Handle search updates
     * @param searchJSON The search JSON
     */
    private handleSearchUpdate(searchJSON: string) {
        try {
            console.log(`Search update: ${searchJSON}`);
            const search = JSON.parse(searchJSON);

            // Update searching status
            const wasSearching = this.searching;
            if (search.searching) {
                this.searching = search.searching.includes(
                    this.config.battleFormat,
                );
                console.log(`Searching status updated: ${this.searching}`);
            } else {
                this.searching = false;
            }

            // If we were searching but aren't anymore and don't have a new game,
            // something might have gone wrong, try searching again
            if (
                wasSearching &&
                !this.searching &&
                (!search.games || Object.keys(search.games).length === 0)
            ) {
                console.log(
                    "Search ended without starting a battle, will try again shortly",
                );
                setTimeout(() => this.searchForBattles(), 5000);
            }

            // Check for new battles
            if (search.games) {
                console.log(`Current games: ${JSON.stringify(search.games)}`);
                Object.keys(search.games).forEach((roomId) => {
                    if (
                        roomId.startsWith("battle-") &&
                        !this.battles.has(roomId)
                    ) {
                        console.log(`New battle started: ${roomId}`);
                        // Join the battle room
                        this.sendMessage("", `/join ${roomId}`);
                        // Create a new battle instance
                        this.battles.set(roomId, new Battle(roomId, this.ws!));
                    }
                });
            }

            // Start searching for more battles if needed
            this.searchForBattles();
        } catch (error) {
            console.error("Error parsing search update:", error);
        }
    }

    /**
     * Search for battles if we're not already searching
     * and we have room for more battles
     */
    private searchForBattles() {
        if (!this.authenticated) {
            console.log("Not authenticated yet, cannot search for battles");
            return;
        }

        const activeBattleCount = Array.from(this.battles.values()).filter(
            (battle) => battle.isActive(),
        ).length;
        console.log(
            `Active battles: ${activeBattleCount}/${this.config.maxConcurrentBattles}, Searching: ${this.searching}`,
        );

        if (
            !this.searching &&
            activeBattleCount < this.config.maxConcurrentBattles
        ) {
            console.log(`Searching for ${this.config.battleFormat} battle...`);

            // First, ensure we have a valid team (or null for random battles)
            const needsTeam = !this.config.battleFormat.includes("random");
            if (needsTeam) {
                // For formats that need a team, send an empty team for now
                // In a real implementation, you would provide a valid team
                this.sendMessage("", `/utm null`);
            }

            // Then start the search
            this.sendMessage("", `/search ${this.config.battleFormat}`);
            this.searching = true;
        }
    }

    /**
     * Clean up inactive battles and search for new ones if needed
     */
    private cleanupBattles() {
        // Remove inactive battles
        for (const [roomId, battle] of this.battles.entries()) {
            if (!battle.isActive()) {
                console.log(`Removing inactive battle: ${roomId}`);
                this.battles.delete(roomId);
                // Leave the battle room
                this.sendMessage("", `/leave ${roomId}`);
            }
        }

        // Search for new battles if needed
        this.searchForBattles();
    }

    /**
     * Login to the Pokémon Showdown server
     */
    private async login() {
        if (this.loginAttempted || this.authenticated) return;
        this.loginAttempted = true;

        try {
            const username = this.config.username;
            const password = this.config.password;

            let assertion;

            if (password) {
                // Login with a registered account
                const loginUrl = "https://play.pokemonshowdown.com/action.php";
                const loginData = new URLSearchParams();
                loginData.append("act", "login");
                loginData.append("name", username);
                loginData.append("pass", password);
                loginData.append("challstr", this.challstr);

                const response = await axios.post(loginUrl, loginData, {
                    headers: {
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Sec-Fetch-Mode": "cors",
                    },
                });

                const responseData = JSON.parse(
                    (response.data as string).slice(1),
                );
                assertion = responseData.assertion;
            } else {
                // Login as a guest
                const loginUrl = `https://play.pokemonshowdown.com/action.php?act=getassertion&userid=${encodeURIComponent(
                    username,
                )}&challstr=${encodeURIComponent(this.challstr)}`;
                const response = await axios.get(loginUrl);
                assertion = response.data;

                if (assertion === ";") {
                    throw new Error(
                        `Username "${username}" is registered. Please provide a password or use a different username.`,
                    );
                }

                if (
                    typeof assertion === "string" &&
                    assertion.startsWith(";;")
                ) {
                    throw new Error(
                        `Login error: ${(assertion as string).slice(2)}`,
                    );
                }
            }

            // Send the login command
            this.sendMessage("", `/trn ${username},0,${assertion}`);
        } catch (error) {
            console.error("Login error:", error);
            this.loginAttempted = false;
        }
    }

    /**
     * Send a message to the server
     * @param roomId The room ID (empty for global)
     * @param message The message to send
     */
    private sendMessage(roomId: string, message: string) {
        if (!this.ws) return;
        this.ws.send(`${roomId}|${message}`);
    }

    /**
     * Run the bot's maintenance tasks
     */
    private runMaintenanceTasks() {
        // Clean up inactive battles every 30 seconds
        setInterval(() => this.cleanupBattles(), 30000);

        // Force search check every 10 seconds in case we miss events
        setInterval(() => this.searchForBattles(), 10000);
    }
}

// Example usage
const defaultConfig: BotConfig = {
    username: "asdf234fae",
    password: "asdf234fae", // Leave undefined for guest login, provide for registered account
    battleFormat: "gen3randombattle", // Format to play
    maxConcurrentBattles: 5, // Maximum number of concurrent battles
    server: {
        url: "sim3.psim.us", // Use 'localhost' for local server
        secure: true, // Use true for wss:// (usually with play.pokemonshowdown.com)
    },
    // server: {
    //     url: "localhost:8000",
    //     secure: false, // Use true for wss:// (usually with play.pokemonshowdown.com)
    // },
};

// Start the bot with the default configuration
const bot = new ShowdownBot(defaultConfig);
bot.start();

// Export for use in other files
export { ShowdownBot, BotConfig, Battle };
