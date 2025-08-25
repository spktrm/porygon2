import { toID } from "@pkmn/dex";
import { Teams, TeamValidator } from "@pkmn/sim";
import * as fs from "fs";
import * as path from "path";

const SETS_URL_BASE =
    "https://raw.githubusercontent.com/pkmn/smogon/refs/heads/main/data/sets/";

type SetsJson = Record<string, Record<string, Record<string, SetType>>>;

type MoveSlot = string | string[];

type Evs = {
    hp?: number;
    atk?: number;
    def?: number;
    spa?: number;
    spd?: number;
    spc?: number;
    spe?: number;
};

type SetType = {
    species: string;
    ability: string | string[];
    evs?: Evs | Evs[];
    ivs?: Evs | Evs[];
    item: string | string[];
    moves: (string | string[])[];
    nature: string | string[];
    happiness: number | string[];
    hiddenPowerType: string | string[];
    teratypes?: string | string[];
};

async function getGenerationSets(gen: number): Promise<SetsJson> {
    if (gen < 1 || gen > 10) {
        throw new Error("Generation must be between 1 and 10.");
    }
    const response = await fetch(SETS_URL_BASE + `gen${gen}.json`);
    if (!response.ok) {
        throw new Error(`Failed to fetch generation ${gen} sets.`);
    }
    return response.json();
}

function combosOf4Unique(slots: MoveSlot[]): string[][] {
    const options: string[][] = slots.map((s) =>
        Array.isArray(s) ? Array.from(new Set(s)) : [s],
    );

    if (options.length < 4) return [];

    const results: string[][] = [];
    const seen = new Set<string>();

    function chooseSlotIndices(start: number, picked: number[]) {
        if (picked.length === 4) {
            const arrs = picked.map((i) => options[i]);
            const used = new Set<string>();
            const acc: string[] = [];

            function product(i: number) {
                if (i === arrs.length) {
                    const canon = [...acc].slice().sort().join("\u0000");
                    if (!seen.has(canon)) {
                        seen.add(canon);
                        results.push(acc.slice()); // keep slot order; change to sorted() if preferred
                    }
                    return;
                }
                for (const v of arrs[i]) {
                    if (used.has(v)) continue; // enforce uniqueness within the combo
                    used.add(v);
                    acc.push(v);
                    product(i + 1);
                    acc.pop();
                    used.delete(v);
                }
            }

            product(0);
            return;
        }

        for (let i = start; i <= options.length - (4 - picked.length); i++) {
            chooseSlotIndices(i + 1, [...picked, i]);
        }
    }

    chooseSlotIndices(0, []);
    return results;
}

function formatValueSpread(spread: Evs, defaultValue: number) {
    return [
        spread.hp ?? defaultValue,
        spread.atk ?? defaultValue,
        spread.def ?? defaultValue,
        spread.spa ?? spread.spc ?? defaultValue,
        spread.spd ?? spread.spc ?? defaultValue,
        spread.spe ?? defaultValue,
    ].join(",");
}

function generatePackedSets(
    species: string,
    set: SetType,
    generation: number,
): string[] {
    // NICKNAME|SPECIES|ITEM|ABILITY|MOVES|NATURE|EVS|GENDER|IVS|SHINY|LEVEL|HAPPINESS,POKEBALL,HIDDENPOWERTYPE,GIGANTAMAX,DYNAMAXLEVEL,TERATYPE

    const {
        item,
        ability,
        moves,
        nature,
        happiness,
        hiddenPowerType,
        teratypes,
    } = set;

    let evs = set.evs ?? {};
    let ivs = set.ivs ?? {};

    if (generation === 1) {
        evs = {
            hp: 252,
            atk: 252,
            def: 252,
            spc: 252,
            spe: 252,
        };
    } else if (generation === 2) {
        evs = {
            hp: 252,
            atk: 252,
            def: 252,
            spa: 252,
            spd: 252,
            spe: 252,
        };
    }

    const packedSets: string[] = [];

    const uniqueMovesets = Array.from(
        new Set(
            combosOf4Unique(
                moves.map((x) => (Array.isArray(x) ? x.map(toID) : toID(x))),
            ).map((moveset) => moveset.sort()),
        ),
    );

    for (const uniqueMoveset of uniqueMovesets) {
        for (const uniqueEvSpreads of Array.isArray(evs) ? evs : [evs]) {
            for (const uniqueIvSpreads of Array.isArray(ivs) ? ivs : [ivs]) {
                for (const uniqueAbility of Array.isArray(ability)
                    ? ability
                    : [ability]) {
                    for (const uniqueItem of Array.isArray(item)
                        ? item
                        : [item]) {
                        for (const uniqueNature of Array.isArray(nature)
                            ? nature
                            : [nature]) {
                            for (const uniqueTeratype of Array.isArray(
                                teratypes,
                            )
                                ? teratypes
                                : [teratypes]) {
                                packedSets.push(
                                    [
                                        "",
                                        toID(species) ?? "",
                                        toID(uniqueItem) ?? "",
                                        toID(uniqueAbility) ?? "",
                                        uniqueMoveset.join(","),
                                        toID(uniqueNature) ?? "",
                                        formatValueSpread(uniqueEvSpreads, 0),
                                        "",
                                        formatValueSpread(uniqueIvSpreads, 31),
                                        "",
                                        "",
                                        [
                                            happiness ?? "",
                                            "",
                                            hiddenPowerType ?? "",
                                            "",
                                            "",
                                            uniqueTeratype ?? "",
                                        ].join(","),
                                    ].join("|"),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    return packedSets;
}

function isSetValid(packed: string, validator: TeamValidator) {
    const unpacked = Teams.unpack([packed].join("]"));
    if (unpacked === null || unpacked.length === 0) {
        return false;
    }
    const errors = validator.validateTeam(unpacked);
    if (errors !== null) {
        // console.log(errors);
    }
    return errors === null;
}

function generateSets(args: {
    sets: SetsJson;
    filterFormat?: string;
    validator: TeamValidator;
}): Map<string, string[]> {
    const { sets, filterFormat, validator } = args;
    const packedSets: Map<string, string[]> = new Map();

    const allSpecies = validator.dex.species.all().map((x) => x.id);
    allSpecies.sort();

    for (const species of [
        "_UNSPECIFIED",
        "_NULL",
        "_PAD",
        "_UNK",
        ...allSpecies,
    ]) {
        packedSets.set(species, []);
    }

    const speciesKeys = Object.keys(sets);
    const speciesBsts = speciesKeys
        .map((species) => validator.dex.species.get(species).bst)
        .reduce((a, b) => a + b, 0);
    const bstAvg = speciesBsts / speciesKeys.length;
    const bstStd = Math.sqrt(
        speciesKeys
            .map((species) =>
                Math.pow(validator.dex.species.get(species).bst - bstAvg, 2),
            )
            .reduce((a, b) => a + b, 0) / speciesKeys.length,
    );

    for (const [species, speciesSets] of Object.entries(sets)) {
        const dexEntry = validator.dex.species.get(species);
        if (
            filterFormat === undefined &&
            dexEntry.bst < bstAvg - bstStd / 2 &&
            dexEntry.nfe
        ) {
            continue;
        }

        const speciesPackedSets = new Set<string>();
        for (const setsFormat of [
            "ubers",
            "ou",
            "uu",
            "ru",
            "nu",
            "pu",
            "monotype",
        ]) {
            const namedSets = speciesSets[setsFormat] ?? {};
            if (!filterFormat || filterFormat === setsFormat) {
                for (const [
                    setName,
                    speciesFormatSetSuperposition,
                ] of Object.entries(namedSets)) {
                    let ability = speciesFormatSetSuperposition.ability;
                    if (ability === undefined) {
                        ability = dexEntry.abilities[0];
                    }
                    for (const generatedSet of generatePackedSets(
                        species,
                        { ...speciesFormatSetSuperposition, ability },
                        validator.gen,
                    )) {
                        if (isSetValid(generatedSet, validator))
                            speciesPackedSets.add(generatedSet);
                    }
                }
            }
        }
        if (speciesPackedSets.size > 0) {
            const dedupedPackedSets = Array.from(speciesPackedSets);
            dedupedPackedSets.sort();
            packedSets.set(toID(species), dedupedPackedSets);
        }
    }

    return packedSets;
}

async function main() {
    const dataDir = path.resolve(__dirname, "../data");

    for (let i = 9; i > 0; i--) {
        const sets = await getGenerationSets(i);

        const validator = new TeamValidator(`gen${i}ou`);
        const allSets = generateSets({ sets, filterFormat: "ou", validator });
        fs.writeFileSync(
            path.join(dataDir, `gen${i}/validated_packed_ou_sets.json`),
            JSON.stringify(Object.fromEntries(allSets)),
        );
        console.log(`Generated validated packed sets for gen${i}`);

        const allFormatSets = generateSets({ sets, validator });
        fs.writeFileSync(
            path.join(dataDir, `gen${i}/validated_packed_all_ou_sets.json`),
            JSON.stringify(Object.fromEntries(allFormatSets)),
        );
        console.log(`Generated validated packed sets for gen${i}`);
    }
}

main();
