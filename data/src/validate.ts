import { PokemonSet, toID } from "@pkmn/dex";
import { Teams, TeamValidator } from "@pkmn/sim";
import * as fs from "fs";
import * as path from "path";

const FORMATS = ["ubers", "ou", "uu", "ru", "nu", "pu", "zu"];

function processSet(packed: string, validator: TeamValidator) {
    const unpacked = Teams.unpack([packed].join("]"));
    if (unpacked === null || unpacked.length === 0) {
        return false;
    }
    const errors = validator.validateTeam(unpacked);
    return errors === null;
}

function main() {
    const dataDir = path.resolve(__dirname, "../data");
    const mainJson = JSON.parse(
        fs.readFileSync(path.join(dataDir, `data.json`), "utf-8"),
    );

    for (let i = 1; i < 10; i++) {
        const readPath = path.join(dataDir, `gen${i}/packed_sets.json`);

        const packedSets: string[] = JSON.parse(
            fs.readFileSync(readPath, "utf-8"),
        );

        const validators: TeamValidator[] = [];
        FORMATS.map((format) => {
            const trueFormat = `gen${i}${format}`;
            try {
                validators.push(new TeamValidator(trueFormat));
            } catch (error) {}
        });

        const unpackedSets = new Set<string>();
        packedSets.flatMap((packed) => {
            const unpackedTeam = Teams.unpack([packed].join("]"));
            if (unpackedTeam !== null && unpackedTeam.length > 0) {
                const onlySet = unpackedTeam[0];
                onlySet.moves.sort();
                unpackedSets.add(JSON.stringify(onlySet));
            }
        });

        const output = new Map<string, Record<string, boolean>>();
        for (const packedSet of [...packedSets].sort()) {
            const row = new Map<string, boolean>();
            let anyValid = false;
            for (const validator of validators) {
                const valid = processSet(packedSet, validator);
                anyValid ||= valid;
                row.set(validator.format.id, valid);
            }
            if (anyValid) {
                output.set(packedSet, Object.fromEntries(row));
            }
        }

        for (const format of FORMATS) {
            const mapping = new Map<string, string[]>();
            const outputObj = Object.fromEntries(output);
            const formatSets = Object.entries(outputObj).flatMap(
                ([packedSet, formats]) => {
                    const formatValid = formats[`gen${i}${format}`] ?? false;
                    return formatValid ? [packedSet] : [];
                },
            );
            for (const [species, index] of Object.entries(
                mainJson["species"],
            )) {
                const sets = formatSets.filter((packedSet) => {
                    const splits = packedSet.split("|", 3);
                    return splits[0] === species || splits[1] === species;
                });
                mapping.set(species, sets);
            }
            const finalOutput = Object.fromEntries(mapping);

            const writePath = path.join(
                dataDir,
                `gen${i}/validated_packed_${format}_sets.json`,
            );
            console.log(
                Object.values(finalOutput)
                    .map((sets) => {
                        return sets.length;
                    })
                    .reduce((a, b) => a + b, 0),
                "valid sets in",
                writePath,
            );
            fs.writeFileSync(writePath, JSON.stringify(finalOutput));
        }
    }
}

main();
