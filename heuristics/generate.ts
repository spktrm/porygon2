import { writeFileSync } from "fs";

import { TeamGenerators } from "@pkmn/randoms";

const formatId = "gen3randombattle";
const generator = TeamGenerators.getTeamGenerator(formatId);

function stringToUniqueInt(str: string): number {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
        hash = (hash * 33) ^ str.charCodeAt(i);
    }
    return hash >>> 0; // Ensure the hash is a positive integer
}

function main() {
    const allSetsMap = new Map();

    for (let i = 0; i < 100000; i++) {
        const team = generator.getTeam();

        for (const { ability, gender, item, level, name, moves } of team) {
            const uniqueSet = {
                name,
                ability,
                moves: moves.sort(),
                item,
                gender,
                level,
            };
            allSetsMap.set(
                stringToUniqueInt(JSON.stringify(uniqueSet)),
                uniqueSet,
            );
        }

        console.log(allSetsMap.size);
    }

    const allSets = Array.from(allSetsMap.values());
    allSets.sort((a, b) => a.name.localeCompare(b.name));
    writeFileSync("allSets.json", JSON.stringify(allSets));
}

main();
