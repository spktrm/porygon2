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
    EntityNodeFeature,
    EntityEdgeFeature,
    FieldFeature,
    ActionMaskFeature,
} from "../../protos/features_pb";
import { Teams, TeamValidator } from "@pkmn/sim";

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
export const numActionMaskFeatures = Object.keys(ActionMaskFeature).length;

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

export const numEntityNodeFeatures = Object.keys(EntityNodeFeature).length;
export const numEntityEdgeFeatures = Object.keys(EntityEdgeFeature).length;
export const numFieldFeatures = Object.keys(FieldFeature).length;
export const numInfoFeatures = Object.keys(InfoFeature).length;
export const numMoveFeatures = Object.keys(MovesetFeature).length;
export const numMovesetFeatures = 10 * numMoveFeatures;

export const NUM_HISTORY = 384;

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

function loadSets(format: string) {
    const validator = new TeamValidator(format);
    const setsToChoose: string[] = JSON.parse(
        fs.readFileSync(`../data/data/${format}_packed.json`, "utf-8"),
    );
    const validSets = setsToChoose.filter((x) => {
        const unpackedSet = Teams.unpack([x].join("]"));
        const errors = validator.validateTeam(unpackedSet);
        if (errors !== null) {
            console.error(
                `Invalid team for format ${format}: ${x} - Errors: ${errors}`,
            );
        }

        return errors === null;
    });
    return validSets;
}

const sets: { [k: string]: string[] } = {
    gen3ou: loadSets("gen3ou"),
};

export function lookUpSets(format: string): string[] {
    return sets[format];
}
