import { Teams, TeamValidator } from "@pkmn/sim";
import * as fs from "fs";

const ALL_FORMATS = ["ubers", "ou", "uu", "ru", "nu", "pu", "zu"];
const MAX_PER_SPECIES = 1024;
const DEBUG = !!process.env.DEBUG;

function validateOnce(
    validator: TeamValidator,
    cache: Map<string, boolean>,
    packedSet: string,
): boolean {
    const cached = cache.get(packedSet);
    if (cached !== undefined) return cached;

    // No need for array/join: a single packed set is a valid packed team with 1 member
    const team = Teams.unpack(packedSet);
    const result = validator.validateTeam(team);
    const ok = result == null;
    cache.set(packedSet, ok);

    if (!ok && DEBUG) {
        // Keep this quiet by default—logging per-failure is costly
        // console.log(result);
    }
    return ok;
}

function processFile(
    generation: number,
    smogonFormat: string,
    suffix: "all_formats" | "only_format",
    validator: TeamValidator,
    cache: Map<string, boolean>,
) {
    const filePath = `../data/data/gen${generation}/${smogonFormat}_${suffix}.json`;
    const jsonData: Record<string, string[]> = JSON.parse(
        fs.readFileSync(filePath, "utf-8"),
    );

    for (const [species, packedSets] of Object.entries(jsonData)) {
        if (!Array.isArray(packedSets) || packedSets.length === 0) continue;

        // We want the *last* 1024 valid sets → scan from the end and stop as soon as we have enough
        const validatedReversed: string[] = [];
        for (
            let i = packedSets.length - 1;
            i >= 0 && validatedReversed.length < MAX_PER_SPECIES;
            i--
        ) {
            const ps = packedSets[i];
            if (validateOnce(validator, cache, ps)) {
                validatedReversed.push(ps);
            }
        }

        // Restore chronological order of those "last" sets
        const validated = validatedReversed.reverse();
        jsonData[species] = validated;

        // Optional lightweight progress log per species (much cheaper than per-set logs)
        console.log(
            generation,
            smogonFormat,
            suffix,
            species,
            validated.length,
        );
    }

    fs.writeFileSync(filePath, JSON.stringify(jsonData));
}

function main() {
    for (let generation = 9; generation > 0; generation--) {
        for (const smogonFormat of [...ALL_FORMATS].reverse()) {
            const validator = new TeamValidator(
                `gen${generation}${smogonFormat}`,
            );

            // Cache validation results across BOTH files ("all_formats" & "only_format")
            // for this (gen, format) so repeated sets don't get re-validated.
            const cache = new Map<string, boolean>();

            for (const suffix of ["all_formats", "only_format"] as const) {
                try {
                    processFile(
                        generation,
                        smogonFormat,
                        suffix,
                        validator,
                        cache,
                    );
                } catch (e) {
                    console.log(
                        `Error processing gen${generation} ${smogonFormat} ${suffix}:`,
                        e,
                    );
                }
            }
        }
    }
}

main();
