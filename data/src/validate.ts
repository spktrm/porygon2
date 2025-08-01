import { PokemonSet, toID } from "@pkmn/dex";
import { Teams, TeamValidator } from "@pkmn/sim";
import * as fs from "fs";
import * as path from "path";

function processFile(unpackedSets: PokemonSet<string>[], format: string) {
    const validator = new TeamValidator(format);

    const uniqueSpecies = new Set<string>();
    unpackedSets.forEach((set) => {
        if (set.species && validator.validateTeam([set]) === null) {
            uniqueSpecies.add(toID(set.species));
        }
    });

    const validTeams = unpackedSets.filter((unpacked) => {
        const species = validator.dex.species.get(unpacked.species);
        if (species.nfe && unpacked.item.toLowerCase() !== "eviolite") {
            const nextEvolutions = [...species.evos];
            while (nextEvolutions.length > 0) {
                const next = nextEvolutions.shift();
                const nextTierEvos = validator.dex.species.get(next).evos;
                nextEvolutions.push(...nextTierEvos);
                if (uniqueSpecies.has(toID(next))) {
                    console.warn(
                        `Skipping ${unpacked.species} in ${format} because it has a next evolution: ${next}`,
                    );
                    return false;
                }
            }
        }

        const errors = validator.validateTeam([unpacked]);
        if (errors !== null) {
            console.error(
                `Invalid team in ${format}: ${unpacked} — Errors: ${errors}`,
            );
        }
        return errors === null;
    });

    console.log(`Valid teams for ${format}: ${validTeams.length}`);
    fs.writeFileSync(
        `../data/data/${format}_packed.json`,
        JSON.stringify(
            validTeams.map((unpacked) => Teams.pack([unpacked])),
            null,
            2,
        ),
    );
}

function main() {
    const dataDir = path.resolve(__dirname, "../data");
    const packedFiles = fs
        .readdirSync(dataDir)
        .filter((f) => f.includes("packed") && f.endsWith(".json"));

    if (packedFiles.length === 0) {
        console.warn("No packed JSON files found in ../data");
        return;
    }

    for (let i = 1; i < 10; i++) {
        const genIPackedFiles = packedFiles.filter((f) =>
            f.startsWith(`gen${i}`),
        );
        const formats = genIPackedFiles.map((f) =>
            path.basename(f, "_packed.json"),
        );
        const genIPackedSets = genIPackedFiles
            .map((f) => {
                try {
                    return JSON.parse(
                        fs.readFileSync(path.join(dataDir, f), "utf-8"),
                    );
                } catch (error) {
                    console.error(`Error reading ${f}: ${error}`);
                    return [];
                }
            })
            .flatMap((f) => f as string[]);

        const unpackedSets = new Set<string>();
        genIPackedSets.flatMap((packed) => {
            const unpackedTeam = Teams.unpack([packed].join("]"));
            if (unpackedTeam !== null && unpackedTeam.length > 0) {
                const onlySet = unpackedTeam[0];
                onlySet.moves.sort();
                unpackedSets.add(JSON.stringify(onlySet));
            }
        });

        for (const format of formats) {
            console.log(`Processing ${format}...`);
            processFile(
                Array.from(unpackedSets).map((x) => JSON.parse(x)),
                format,
            );
        }
    }
}

main();
