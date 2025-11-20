import * as fs from "fs";

import {
    AbilitiesEnumMap,
    BattlemajorargsEnum,
    BattlemajorargsEnumMap,
    BattleminorargsEnum,
    BattleminorargsEnumMap,
    BoostsEnum,
    BoostsEnumMap,
    ConditionEnumMap,
    EffectEnumMap,
    EffecttypesEnumMap,
    GendernameEnumMap,
    ItemeffecttypesEnumMap,
    ItemsEnumMap,
    LastitemeffecttypesEnumMap,
    MovesEnumMap,
    NaturesEnumMap,
    PseudoweatherEnum,
    PseudoweatherEnumMap,
    SideconditionEnum,
    SideconditionEnumMap,
    SpeciesEnumMap,
    StatusEnumMap,
    TypechartEnum,
    TypechartEnumMap,
    VolatilestatusEnum,
    VolatilestatusEnumMap,
    WeatherEnumMap,
} from "../../protos/enums_pb";
import { OneDBoolean } from "./utils";
import {
    MovesetFeature,
    InfoFeature,
    EntityEdgeFeature,
    FieldFeature,
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
} from "../../protos/features_pb";
import { ActionEnum, WildCardEnum } from "../../protos/service_pb";

export type EnumMappings =
    | SpeciesEnumMap
    | ItemsEnumMap
    | StatusEnumMap
    | TypechartEnumMap
    | ItemeffecttypesEnumMap
    | LastitemeffecttypesEnumMap
    | MovesEnumMap
    | AbilitiesEnumMap
    | BoostsEnumMap
    | VolatilestatusEnumMap
    | SideconditionEnumMap
    | ConditionEnumMap
    | WeatherEnumMap
    | PseudoweatherEnumMap
    | GendernameEnumMap
    | BattlemajorargsEnumMap
    | BattleminorargsEnumMap
    | EffectEnumMap
    | EffecttypesEnumMap
    | NaturesEnumMap;

export type MoveIndex = 0 | 1 | 2 | 3;
export type BenchIndex = MoveIndex | 4 | 5;

export const numBoosts = Object.keys(BoostsEnum).length;
export const numTypes = Object.keys(TypechartEnum).length;
export const numVolatiles = Object.keys(VolatilestatusEnum).length;
export const numSideConditions = Object.keys(SideconditionEnum).length;
export const numBattleMinorArgs = Object.keys(BattleminorargsEnum).length;
export const numBattleMajorArgs = Object.keys(BattlemajorargsEnum).length;
export const numPseudoweathers = Object.keys(PseudoweatherEnum).length;
export const numActionMaskFeatures = Object.keys(ActionEnum).length;
export const numWildcardMaskFeatures = Object.keys(WildCardEnum).length;

export const actionIndexMapping = {
    0: "move 1",
    1: "move 2",
    2: "move 3",
    3: "move 4",
    4: "switch 1",
    5: "switch 2",
    6: "switch 3",
    7: "switch 4",
    8: "switch 5",
    9: "switch 6",
};

export const sideIdMapping: {
    [k in "p1" | "p2"]: 0 | 1;
} = {
    p1: 0,
    p2: 1,
};

export const numPrivateEntityNodeFeatures = Object.keys(
    EntityPrivateNodeFeature,
).length;
export const numPublicEntityNodeFeatures = Object.keys(
    EntityPublicNodeFeature,
).length;
export const numRevealedEntityNodeFeatures = Object.keys(
    EntityRevealedNodeFeature,
).length;
export const numEntityEdgeFeatures = Object.keys(EntityEdgeFeature).length;
export const numFieldFeatures = Object.keys(FieldFeature).length;
export const numInfoFeatures = Object.keys(InfoFeature).length;
export const numMoveFeatures = Object.keys(MovesetFeature).length;
export const numMovesetFeatures = 10 * numMoveFeatures;

export const NUM_HISTORY = 128 * 3;

export const AllValidActions = new OneDBoolean(10, Uint8Array);
for (let actionIndex = 0; actionIndex < 10; actionIndex++) {
    AllValidActions.set(actionIndex, true);
}

// Define the path to the JSON file
const filePath = "../data/data/data.json";

// Read the file synchronously
const fileContent = fs.readFileSync(filePath, "utf-8");

function transformJson(
    json: Record<string, Record<string, number>>,
): Record<string, Record<number, string>> {
    const transformed: Record<string, Record<number, string>> = {};

    for (const [key, value] of Object.entries(json)) {
        if (typeof value === "object" && value !== null) {
            // Recursive transformation for nested objects
            transformed[key] = Object.fromEntries(
                Object.entries(value).map(([innerKey, innerValue]) => [
                    innerValue as number,
                    innerKey,
                ]),
            );
        } else {
            transformed[key] = value; // Keep non-object entries as-is
        }
    }

    return transformed;
}

// Parse the JSON content
export const jsonDatum = transformJson(JSON.parse(fileContent));

export const sets: { [k: string]: Record<string, string[]> } = {};

for (const i of [1, 9]) {
    for (const smogonFormat of ["ubers", "ou", "uu", "ru", "nu", "pu", "zu"]) {
        for (const suffix of ["all_formats", "only_format"] as const) {
            try {
                const data = fs.readFileSync(
                    `../data/data/gen${i}/${smogonFormat}_${suffix}.json`,
                    "utf-8",
                );
                sets[`gen${i}_${smogonFormat}_${suffix}`] = JSON.parse(data);
            } catch (err) {
                continue;
            }
        }
    }
}

export function lookUpSetsList(
    smogonFormat: string,
    species: string,
): string[] {
    const maybeSets = sets[smogonFormat];
    if (maybeSets === undefined) {
        throw new Error(`No sets found for format: ${smogonFormat}`);
    }
    const speciesSets = maybeSets[species];
    if (speciesSets === undefined) {
        throw new Error(`No sets found for species: ${species}`);
    }
    return speciesSets;
}
