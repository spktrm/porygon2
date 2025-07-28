import WebSocket from "ws";
import axios from "axios";
import * as dotenv from "dotenv";
import * as path from "path";

import { ObjectReadWriteStream } from "@pkmn/streams";
import { TrainablePlayerAI } from "../server/runner";
import { StepRequest } from "../../protos/service_pb";

// class ClientStream extends ObjectReadWriteStream<string> {
//     debug: boolean;
//     roomId: string | undefined;
//     player: Player;
//     tasks: TaskQueueSystem<Action>;
//     modelOutput: Record<string, unknown>;
//     userName: string;

//     constructor(
//         options: {
//             debug?: boolean;
//             roomId?: string;
//             config?: BotConfig;
//             choose?: (action: string) => void;
//         } = {},
//     ) {
//         super();
//         this.debug = !!options.debug;
//         this.roomId = options?.roomId;
//         if (options?.config?.username === undefined) {
//             throw new Error("bad");
//         }
//         this.userName = options?.config?.username;
//         this.modelOutput = {};

//         this.tasks = new TaskQueueSystem();

//         const sendFn: sendFnType = async (player) => {
//             const gameState = new GameState();
//             const state = player.createState();
//             try {
//                 const stateBinary = state.serializeBinary();
//                 gameState.setState(stateBinary);
//             } catch (e) {
//                 console.log(state.toObject());
//                 console.log(e);
//             }
//             const playerId = 0;
//             let rqid = -1;
//             if (!state.getInfo()!.getDone()) {
//                 rqid = this.tasks.createJob();
//             }
//             gameState.setRqid(rqid);
//             gameState.setPlayerId(playerId);
//             const response = await fetch("http://127.0.0.1:8001/predict", {
//                 // Ensure this URL is correct for your setup
//                 method: "POST",
//                 body: state.serializeBinary(),
//             });
//             const modelOutput = await response.json();
//             this.modelOutput = modelOutput;
//             const action = await ActionFromResponse(modelOutput);
//             action.setRqid(rqid);
//             if (rqid >= 0) this.tasks.submitResult(rqid, action);
//             return rqid;
//         };

//         const recvFn: recvFnType = async (rqid) => {
//             return rqid >= 0 ? this.tasks.getResult(rqid) : undefined;
//         };

//         if (!options.roomId) {
//             throw new Error("Options must have roomId");
//         }

//         this.player = new Player(
//             0,
//             Math.floor(2 ** 16 * Math.random()),
//             this,
//             0,
//             sendFn,
//             recvFn,
//             null,
//             this.userName,
//             options.choose!,
//         );
//         this.player.start();
//     }

//     _write(message: string) {
//         this.push(message);
//     }

//     getThinkingMesssage(): string {
//         const request = this.player.getRequest() as AnyObject;
//         const pi = this.modelOutput.pi as number[] | undefined;
//         const action = this.modelOutput.action as number | undefined;
//         const value = this.modelOutput.v as number | undefined;

//         if (
//             request !== undefined &&
//             this.modelOutput &&
//             pi !== undefined &&
//             action !== undefined &&
//             value !== undefined
//         ) {
//             const messages = [];

//             if (value > 0.66) {
//                 const winningMessages = [
//                     "I'm almost certain to win this battle!",
//                     "I'm very likely to win this battle!",
//                     "Victory is within my grasp!",
//                     "I feel confident about winning!",
//                 ];
//                 messages.push(
//                     winningMessages[
//                         Math.floor(Math.random() * winningMessages.length)
//                     ],
//                 );
//             } else if (value > 0.33) {
//                 const favorableMessages = [
//                     "Things are looking good for me.",
//                     "I'm in a favorable position.",
//                     "I have the advantage right now.",
//                     "I'm ahead in this battle.",
//                 ];
//                 messages.push(
//                     favorableMessages[
//                         Math.floor(Math.random() * favorableMessages.length)
//                     ],
//                 );
//             } else if (value > 0) {
//                 const slightAdvantageMessages = [
//                     "I have a slight advantage.",
//                     "I'm doing okay so far.",
//                     "It's looking a bit better for me.",
//                     "I'm a little ahead.",
//                 ];
//                 messages.push(
//                     slightAdvantageMessages[
//                         Math.floor(
//                             Math.random() * slightAdvantageMessages.length,
//                         )
//                     ],
//                 );
//             } else if (value > -0.33) {
//                 const evenMessages = [
//                     "This battle is pretty even.",
//                     "It's a close match.",
//                     "Neither side has a clear advantage.",
//                     "It's anyone's game right now.",
//                 ];
//                 messages.push(
//                     evenMessages[
//                         Math.floor(Math.random() * evenMessages.length)
//                     ],
//                 );
//             } else if (value > -0.66) {
//                 const unfavorableMessages = [
//                     "I'm in a tough spot.",
//                     "Things aren't looking great for me.",
//                     "I'm at a disadvantage right now.",
//                     "I need to turn things around.",
//                 ];
//                 messages.push(
//                     unfavorableMessages[
//                         Math.floor(Math.random() * unfavorableMessages.length)
//                     ],
//                 );
//             } else if (value > -1) {
//                 const losingMessages = [
//                     "I'm very likely to lose this battle.",
//                     "Defeat seems imminent.",
//                     "I'm almost certain to lose.",
//                     "It's not looking good for me at all.",
//                 ];
//                 messages.push(
//                     losingMessages[
//                         Math.floor(Math.random() * losingMessages.length)
//                     ],
//                 );
//             }

//             const active = (request?.active ?? [])[0] as
//                 | Protocol.MoveRequest["active"][0]
//                 | null;
//             const activeMoves = active && active.moves ? active.moves : [];
//             const switches = (request?.side?.pokemon ??
//                 []) as Protocol.Request.SideInfo["pokemon"];

//             console.log(active?.moves.map((move) => move.id));
//             console.log(switches.map((poke) => poke.speciesForme));

//             const moveLabels = [
//                 ...activeMoves.map((action) => `use ${action.id}`),
//                 ...new Array(4 - activeMoves.length),
//                 ...switches.map((action) => `switch to ${action.speciesForme}`),
//                 ...new Array(6 - switches.length),
//             ];

//             const entropy =
//                 -pi.reduce(
//                     (acc, prob) => acc - prob * Math.log(prob + 1e-10),
//                     0,
//                 ) / Math.log(pi.length);
//             if (entropy > 0.8) {
//                 const notSureMessages = [
//                     "I'm not sure what to do...",
//                     "I need more time to think...",
//                     "I'm still deciding...",
//                 ];
//                 messages.push(
//                     notSureMessages[
//                         Math.floor(Math.random() * notSureMessages.length)
//                     ],
//                 );
//                 return messages
//                     .flatMap((m) => (Math.random() < 0.5 ? [] : [m]))
//                     .join("... ");
//             }

//             const topKIndices = pi
//                 .map((prob: number, index: number) => ({
//                     index,
//                     prob,
//                 }))
//                 .sort((a, b) => b.prob - a.prob)
//                 .slice(0, 3)
//                 .map((item) => item.index);

//             for (const kIndex of topKIndices) {
//                 const moveLabel = moveLabels[kIndex];
//                 if (moveLabel) {
//                     const probability = pi[kIndex];

//                     if (probability > 0.75) {
//                         const probMessages = [
//                             "I'm almost certain to",
//                             "I'm very likely to",
//                             "I will probably",
//                             "I will likely",
//                             "I will definitely",
//                             "I'm planning to",
//                             "I'm set to",
//                             "I'm going to",
//                             "I'm sure to",
//                             "I'm confident to",
//                             "I'm ready to",
//                             "I'm about to",
//                             "I'm inclined to",
//                             "I'm determined to",
//                             "I'm prepared to",
//                         ];
//                         const randomIndex = Math.floor(
//                             Math.random() * probMessages.length,
//                         );
//                         messages.push(
//                             `${probMessages[randomIndex]} ${moveLabel}.`,
//                         );
//                     } else if (probability > 0.5) {
//                         const mightMessages = [
//                             "It's possible to",
//                             "There's a chance to",
//                             "I could possibly",
//                             "Maybe I'll",
//                             "I might",
//                             "I may decide to",
//                             "I could go for",
//                             "I might try to",
//                         ];
//                         const randomIndex = Math.floor(
//                             Math.random() * mightMessages.length,
//                         );
//                         messages.push(
//                             `${mightMessages[randomIndex]} ${moveLabel}.`,
//                         );
//                     } else if (probability > 0.25) {
//                         const couldMessages = [
//                             "I might",
//                             "There's a chance to",
//                             "It's possible to",
//                             "I could",
//                             "Maybe I'll",
//                             "I may",
//                             "I suppose I could",
//                             "I might consider",
//                             "I could end up",
//                             "Perhaps I'll",
//                             "I could see myself",
//                             "I might opt to",
//                         ];
//                         const randomIndex = Math.floor(
//                             Math.random() * couldMessages.length,
//                         );
//                         messages.push(
//                             `${couldMessages[randomIndex]} ${moveLabel}.`,
//                         );
//                     } else if (probability > 0.01) {
//                         const unlikelyMessages = [
//                             "I probably won't",
//                             "It's unlikely I'll",
//                             "I doubt I'll",
//                             "I don't think I'll",
//                             "I might not",
//                             "I probably won't end up",
//                             "It's doubtful that I'll",
//                             "I don't expect to",
//                             "I likely won't",
//                             "I could avoid",
//                             "I might skip",
//                             "I may not",
//                         ];
//                         const randomIndex = Math.floor(
//                             Math.random() * unlikelyMessages.length,
//                         );
//                         messages.push(
//                             `${unlikelyMessages[randomIndex]} ${moveLabel}.`,
//                         );
//                     }
//                 }
//             }

//             if (messages.length === 0) {
//                 messages.push(`I will definitely ${moveLabels[action]}`);
//                 return messages
//                     .flatMap((m) => (Math.random() < 0.5 ? [] : [m]))
//                     .join("... ");
//             } else {
//                 return messages
//                     .flatMap((m) => (Math.random() < 0.5 ? [] : [m]))
//                     .join("... ");
//             }
//         }
//         const generalMessages = [
//             "Thinking...",
//             "Processing...",
//             "Analyzing the situation...",
//         ];
//         const randomIndex = Math.floor(Math.random() * generalMessages.length);
//         return generalMessages[randomIndex];
//     }
// }

const welcomeMessage =
    "You are fighting an AI in development - trained with reinforcement learning. If you have any questions, please message spktrm#6133 on discord. glfh :)";

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
    private ws: WebSocket;
    private config: BotConfig;
    stream: ObjectReadWriteStream<string>;
    player: TrainablePlayerAI;
    prevMessage: string | undefined;
    active: boolean;

    constructor(roomId: string, ws: WebSocket, config: BotConfig) {
        this.battleId = roomId;
        this.ws = ws;
        this.active = true;
        this.config = config;
        this.prevMessage = undefined;

        // this.ws.send(`${this.battleId}|/timer on`);
        this.ws.send(`${this.battleId}|${welcomeMessage}`);
        this.stream = new ClientStream();
        this.player = new TrainablePlayerAI(
            this.config.username,
            this.stream,
            {},
            true,
        );
        this.player.choose = (choice: string) => {
            this.ws.send(
                `${this.battleId}|/choose ${choice}|${this.player.rqid}`,
            );
        };
        this.player.start();
    }

    public async start(rateLimit: number = 1000) {
        while (!this.player.done) {
            const state = await this.player.receiveEnvironmentState();
            const response = await fetch("http://127.0.0.1:8001/predict", {
                // Ensure this URL is correct for your setup
                method: "POST",
                body: state.serializeBinary(),
            });
            await new Promise((resolve) => setTimeout(resolve, rateLimit));
            const modelOutput = await response.json();
            const stepRequest = new StepRequest();
            stepRequest.setAction(modelOutput.action);
            stepRequest.setRqid(state.getRqid());
            this.player.submitStepRequest(stepRequest);
        }
    }

    public async receive(message: string): Promise<void> {
        this.stream.write(message);
    }

    public getBattleId(): string {
        return this.battleId;
    }

    public leave() {
        this.ws.send(`${this.battleId}|/leave`);
    }
}

interface BotConfig {
    username: string;
    password: string;
    format: string;
    team?: string;
    maxConcurrentBattles: number;
    serverUrl: string;
    secure: boolean;
}

interface BotState {
    activeBattles: number;
    isSearchingLadder: boolean; // Added to track ladder search status
    rooms: Set<string>;
    challenges: Map<string, string>;
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
            isSearchingLadder: false, // Initialize here
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
            this.state.isSearchingLadder = false; // Reset search state on disconnect

            setTimeout(() => {
                this.connect();
            }, 10000);
        });

        this.ws.on("error", (error) => {
            console.error("WebSocket error:", error.message);
            this.state.isSearchingLadder = false; // Reset search state on error
        });
    }

    private handleMessage(data: WebSocket.RawData): void {
        const message = data.toString();
        const lines = message.split("\n");

        if (message.startsWith(">")) {
            // Battle room messages
            const roomId = lines[0].slice(1);
            if (!this.state.battles.has(roomId)) {
                // This is a new battle starting
                const battle = new Battle(roomId, this.ws!, this.config);
                this.state.battles.set(roomId, battle);
                battle.start();
                this.state.activeBattles++;
                this.state.isSearchingLadder = false; // A battle started, so we are no longer just "searching"
                console.log(
                    `Battle ${roomId} started. Active battles: ${this.state.activeBattles}. No longer searching ladder.`,
                );
            }

            const battle = this.state.battles.get(roomId);
            if (!battle) {
                // Should not happen if the above logic is correct
                console.error(
                    `Error: Battle instance for ${roomId} not found after creation.`,
                );
                return;
            }

            battle.receive(message);

            const battleIsWon = message.includes("|win|");
            const battleIsTie = message.includes("|tie|");
            if (battleIsWon || battleIsTie) {
                if (this.state.battles.has(roomId)) {
                    this.state.activeBattles--;
                    // No need to delete from this.state.battles here, |deinit| handles it.
                    // Or, if |deinit| is not reliable for this, clean up here:
                    // this.state.battles.delete(roomId);
                    console.log(
                        `Battle ${roomId} ended. Active battles: ${this.state.activeBattles}.`,
                    );
                }

                const battle = this.state.battles.get(roomId);
                if (battle) {
                    battle.leave();
                }
                if (this.state.rooms.has(roomId)) {
                    this.state.rooms.delete(roomId);
                }

                this.state.isSearchingLadder = false; // Reset search state as a battle slot might be free
                this.checkAndSearchForBattles();
            }
        } else {
            // General server messages
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
        // console.log(line); // Optional: keep for debugging

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
                    this.state.isSearchingLadder = false; // Reset search state on login
                    this.checkAndSearchForBattles();
                }
                break;
            }

            case "pm": {
                const sender = parts[1].trim();
                const receiver = parts[2].trim();
                const messageContent = parts.slice(3).join("|");

                if (
                    messageContent.startsWith("/challenge") &&
                    receiver.toLowerCase() ===
                        this.config.username.toLowerCase()
                ) {
                    const challengeParts = messageContent.split("|");
                    const format = challengeParts[1]; // Ensure this parsing is robust

                    this.state.challenges.set(sender, format);
                    this.checkAndAcceptChallenge(sender);
                }
                break;
            }

            case "updatechallenges":
                try {
                    const challengesData = JSON.parse(parts.slice(1).join("|")); // Ensure full JSON is parsed

                    if (challengesData.challengesFrom) {
                        for (const challenger in challengesData.challengesFrom) {
                            const format =
                                challengesData.challengesFrom[challenger];
                            this.state.challenges.set(challenger, format);
                            this.checkAndAcceptChallenge(challenger);
                        }
                    }
                } catch (error) {
                    console.error(
                        "Error parsing challenges JSON:",
                        error,
                        "Raw data:",
                        parts.slice(1).join("|"),
                    );
                }
                break;

            case "updatesearch": {
                try {
                    const searchData = JSON.parse(parts.slice(1).join("|")); // Ensure full JSON is parsed
                    if (
                        this.state.isSearchingLadder &&
                        searchData.searching === false
                    ) {
                        console.log(
                            "Server indicated searching has stopped (no match found or cancelled). Resetting isSearchingLadder flag.",
                        );
                        this.state.isSearchingLadder = false;
                    }
                } catch (e) {
                    console.error(
                        "Error parsing updatesearch JSON:",
                        e,
                        "Raw data:",
                        parts.slice(1).join("|"),
                    );
                }
                // Always call checkAndSearchForBattles; it will decide based on the new state.
                this.checkAndSearchForBattles();
                break;
            }

            case "init": {
                if (parts[1] === "battle") {
                    const roomId = parts[2] || ""; // parts[2] should be the room ID e.g. "battle-gen3randombattle-123"
                    console.log(`Room initialized: ${roomId}`);
                    this.state.rooms.add(roomId);
                    // Note: Battle instance creation and activeBattles increment is now in handleMessage for ">roomid"
                }
                break;
            }

            case "deinit": {
                // eslint-disable-next-line @typescript-eslint/no-unused-vars
                const roomTypeOrId = parts[1]; // Sometimes this is just 'battle', other times it's the full room ID.
                // If it's just 'battle', then parts[2] is the room ID.
                // Let's assume for now it's the room ID or not relevant if just 'battle'.
                let roomId = "";
                if (parts.length > 2 && parts[1] === "battle") {
                    // If message is like |deinit|battle|battle-id
                    roomId = parts[2];
                } else if (parts.length > 1 && parts[1].startsWith("battle-")) {
                    // If message is like |deinit|battle-id
                    roomId = parts[1];
                }

                if (roomId && this.state.battles.has(roomId)) {
                    console.log(
                        `Room deinitialized: ${roomId}. Removing from active battles map.`,
                    );
                    this.state.battles.delete(roomId);
                    // activeBattles should have been decremented by win/loss logic,
                    // but if a battle ends for other reasons (disconnect, forfeit without win/loss message),
                    // ensure activeBattles is correct.
                    // However, the primary decrement is tied to win/loss. If deinit occurs without win/loss,
                    // activeBattles might be off. Consider if activeBattles should be this.state.battles.size.
                    // For no
                    // w, relying on win/loss to decrement activeBattles.
                }
                if (roomId && this.state.rooms.has(roomId)) {
                    this.state.rooms.delete(roomId);
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
                new URLSearchParams({
                    // Use URLSearchParams for form data
                    act: "login",
                    name: this.config.username,
                    pass: this.config.password,
                    challstr: this.challstr,
                }).toString(),
                {
                    headers: {
                        "Content-Type": "application/x-www-form-urlencoded",
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
                    console.error(
                        "No assertion received in login response. Login might have failed.",
                        json,
                    );
                    this.loggedIn = false;
                }
            } else {
                console.error("Invalid login response format:", responseData);
                this.loggedIn = false;
            }
        } catch (error) {
            console.error(
                "Login HTTP request error:",
                error instanceof Error ? error.message : error,
            );
            this.loggedIn = false;
        }
    }

    private checkAndSearchForBattles(): void {
        if (!this.loggedIn) {
            console.log("Not logged in. Cannot search for battles.");
            return;
        }

        if (this.state.activeBattles >= this.config.maxConcurrentBattles) {
            console.log(
                `Maximum number of active battles reached (${this.state.activeBattles}/${this.config.maxConcurrentBattles}). Not searching.`,
            );
            return;
        }

        if (this.state.isSearchingLadder) {
            console.log(
                "Already searching for a ladder battle. Not sending another search request.",
            );
            return;
        }

        // At this point, activeBattles < maxConcurrentBattles AND not currently searching.
        console.log(
            `Attempting to search for new battle. Active: ${this.state.activeBattles}, Max: ${this.config.maxConcurrentBattles}, Searching: ${this.state.isSearchingLadder}`,
        );

        if (this.config.team) {
            this.send(`|/utm ${this.config.team}`);
        }

        this.send(`|/search ${this.config.format}`);
        this.state.isSearchingLadder = true; // Set flag after sending search
        console.log(
            `Sent search request for ${this.config.format}. Flag isSearchingLadder set to true.`,
        );
    }

    private checkAndAcceptChallenge(challenger: string): void {
        if (!this.loggedIn) {
            console.log("Not logged in. Cannot accept challenges.");
            return;
        }

        const format = this.state.challenges.get(challenger);
        if (!format) {
            console.error(
                `No challenge found from ${challenger} in state.challenges. This shouldn't happen.`,
            );
            return;
        }

        if (format !== this.config.format) {
            console.log(
                `Rejecting challenge from ${challenger} due to mismatched format: ${format} (expected ${this.config.format})`,
            );
            this.send(`|/reject ${challenger}`);
            this.state.challenges.delete(challenger);
            return;
        }

        // Calculate potential battles if this challenge is accepted
        // A ladder search counts as a potential battle if active.
        const potentialBattleSlotsOccupied =
            this.state.activeBattles + (this.state.isSearchingLadder ? 1 : 0);

        if (potentialBattleSlotsOccupied >= this.config.maxConcurrentBattles) {
            console.log(
                `Cannot accept challenge from ${challenger}. Would exceed max concurrent battles. Active: ${this.state.activeBattles}, SearchingLadder: ${this.state.isSearchingLadder}, Max: ${this.config.maxConcurrentBattles}. Rejecting.`,
            );
            this.send(`|/reject ${challenger}`);
            // Do not delete the challenge yet, it might become acceptable if a slot frees up soon.
            // Or, decide to delete if you don't want to re-evaluate it:
            this.state.challenges.delete(challenger);
            return;
        }

        console.log(
            `Accepting challenge from ${challenger} for format ${format}. Active: ${this.state.activeBattles}, SearchingLadder: ${this.state.isSearchingLadder}, Max: ${this.config.maxConcurrentBattles}.`,
        );

        if (this.config.team) {
            this.send(`|/utm ${this.config.team}`);
        }

        this.send(`|/accept ${challenger}`);
        // Note: isSearchingLadder is NOT changed here. Accepting a challenge doesn't stop a ladder search.
        // activeBattles will increment when the battle room for this challenge starts.
        this.state.challenges.delete(challenger); // Accepted, so remove from pending challenges
    }

    private send(message: string): void {
        if (!this.connected || !this.ws) {
            console.error("Not connected. Cannot send message:", message);
            return;
        }
        this.ws.send(message);
    }
}

// Example usage (ensure your .env file is correctly set up at the specified path)
if (require.main === module) {
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

    const config: BotConfig = {
        username: process.env.SHOWDOWN_USERNAME!,
        password: process.env.SHOWDOWN_PASSWORD!,
        format: "gen3randombattle",
        maxConcurrentBattles: 1, // Changed from maxSearches to maxConcurrentBattles
        serverUrl: "localhost:8000", // Change to 'sim.smogon.com' for main server
        secure: false, // Set to true for main server
        // serverUrl: "sim3.psim.us", // Use 'localhost' for local server
        // secure: true, // Use true for wss:// (usually with play.pokemonshowdown.com)
    };

    if (!config.username || !config.password) {
        console.error(
            "Username or password not found in environment variables. Please check your .env file.",
        );
        process.exit(1);
    }

    console.log("Starting bot with configuration:", {
        ...config,
        password: "[REDACTED]", // Don't log password
    });

    const bot = new ShowdownBot(config);
    bot.start();
}
