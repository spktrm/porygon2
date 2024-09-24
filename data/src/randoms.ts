import { TeamGenerators } from "@pkmn/randoms";
import fs from "fs";

// List of all relevant randombattle formats
const formats = [
    "gen1randombattle",
    "gen2randombattle",
    "gen3randombattle",
    "gen4randombattle",
    "gen5randombattle",
    "gen6randombattle",
    "gen7randombattle",
    "gen8randombattle",
    "gen9randombattle",
];

const maxIterations = 500;

// Initialize an object to hold the results for each format
const results: Record<string, any> = {};

for (const format of formats) {
    const generator = TeamGenerators.getTeamGenerator(format);

    const speciesSet = new Set<string>();
    const movesSet = new Set<string>();
    const itemsSet = new Set<string>();
    const abilitiesSet = new Set<string>();

    let noChangeCounter = 0;
    let previousSpeciesSize = 0;
    let previousMovesSize = 0;
    let previousItemsSize = 0;
    let previousAbilitiesSize = 0;

    while (true) {
        const team = generator.getTeam();

        for (const member of team) {
            speciesSet.add(member.species);
            for (const move of member.moves) {
                movesSet.add(move);
            }
            itemsSet.add(member.item);
            abilitiesSet.add(member.ability);
        }

        // Check if the sizes have changed
        if (
            speciesSet.size === previousSpeciesSize &&
            movesSet.size === previousMovesSize &&
            itemsSet.size === previousItemsSize &&
            abilitiesSet.size === previousAbilitiesSize
        ) {
            noChangeCounter++;
        } else {
            noChangeCounter = 0; // Reset counter if sizes have changed
        }

        // Update previous sizes
        previousSpeciesSize = speciesSet.size;
        previousMovesSize = movesSet.size;
        previousItemsSize = itemsSet.size;
        previousAbilitiesSize = abilitiesSet.size;

        // Exit loop if no changes for maxIterations iterations
        if (noChangeCounter >= maxIterations) {
            console.log(
                `No changes in ${maxIterations} iterations for ${format}. Exiting loop.`,
            );
            break;
        }
    }

    // Store the results for the current format
    results[format] = {
        species: Array.from(speciesSet),
        moves: Array.from(movesSet),
        items: Array.from(itemsSet),
        abilities: Array.from(abilitiesSet),
    };
}

// Save the results to a JSON file
fs.writeFileSync(
    "../data/data/randombattle_data.json",
    JSON.stringify(results, null, 2),
);

console.log("Data saved to randombattle_data.json.");
