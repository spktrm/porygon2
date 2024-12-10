import { ObjectReadWriteStream } from "@pkmn/streams";
import { Player } from "../server/player";
import { readFileSync } from "fs";

class OfflineStream extends ObjectReadWriteStream<string> {
    constructor() {
        super();
    }

    _write(message: string) {
        this.push(message);
    }
}

async function main() {
    const replay = JSON.parse(
        readFileSync(
            "../replays/data/gen3randombattle/gen3randombattle-11181.json",
        ).toString(),
    );
    console.log(replay);
    const chunks = replay.log.split("|upkeep\n");
    const inputLog: string[] = replay.inputlog.split("\n");

    for (const i of [0, 1]) {
        const states = [];
        const playerInputLog = inputLog.filter((x) =>
            x.startsWith(`>p${i + 1}`),
        );

        const stream = new OfflineStream();
        for (const chunk of chunks) {
            stream.write(chunk);
        }
        stream.pushEnd();

        async function send(player: Player) {
            const state = player.createState();
            states.push(state);
            // console.log(playerInputLog[states.length - 1]);
            return -1;
        }

        async function recv() {
            return undefined;
        }

        const player = new Player(
            0,
            0,
            stream,
            0,
            send,
            recv,
            null,
            undefined,
            true,
            i,
        );
        await player.start();

        console.log(states.length);
        console.log(playerInputLog.length);
    }
}

main();
