import { PokemonSet, toID } from "@pkmn/dex";
import { Teams, TeamValidator } from "@pkmn/sim";
import * as fs from "fs";
import * as path from "path";

const FORMATS = [
    "ubers",
    "ou",
    "uu",
    "uubl",
    "ru",
    "rubl",
    "nu",
    "nubl",
    "pu",
    "publ",
    "zu",
    "zubl",
];

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
    const packedFiles = fs
        .readdirSync(dataDir)
        .filter((f) => f.includes("packed") && f.endsWith(".json"));

    if (packedFiles.length === 0) {
        console.warn("No packed JSON files found in ../data");
        return;
    }

    for (const [idx, fpath] of packedFiles.entries()) {
        const packedSets: string[] = JSON.parse(
            fs.readFileSync(path.join(dataDir, fpath), "utf-8"),
        );

        const validators: TeamValidator[] = [];
        FORMATS.map((format) => {
            const trueFormat = `gen${idx + 1}${format}`;
            try {
                validators.push(new TeamValidator(trueFormat));
            } catch (error) {
                console.log(
                    `Error creating validator for ${trueFormat}:`,
                    error,
                );
            }
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

        fs.writeFileSync(
            path.join(dataDir, `validated_${fpath}`),
            JSON.stringify(Object.fromEntries(output)),
        );
    }
}

main();
