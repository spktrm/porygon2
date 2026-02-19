import * as fs from "fs";

import {
    AbilitiesEnum,
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
    GendernameEnum,
    GendernameEnumMap,
    ItemeffecttypesEnumMap,
    ItemsEnum,
    ItemsEnumMap,
    LastitemeffecttypesEnumMap,
    MovesEnum,
    MovesEnumMap,
    NaturesEnum,
    NaturesEnumMap,
    PseudoweatherEnum,
    PseudoweatherEnumMap,
    SideconditionEnum,
    SideconditionEnumMap,
    SpeciesEnum,
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
    PackedSetFeature,
} from "../../protos/features_pb";
import { ActionEnum } from "../../protos/service_pb";

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

export const ITOS = {
    species: Object.fromEntries(
        Object.entries(SpeciesEnum).map(([k, v]) => [
            v,
            k.split("__")[1].toLowerCase(),
        ]),
    ),
    items: Object.fromEntries(
        Object.entries(ItemsEnum).map(([k, v]) => [
            v,
            k.split("__")[1].toLowerCase(),
        ]),
    ),
    abilities: Object.fromEntries(
        Object.entries(AbilitiesEnum).map(([k, v]) => [
            v,
            k.split("__")[1].toLowerCase(),
        ]),
    ),
    moves: Object.fromEntries(
        Object.entries(MovesEnum).map(([k, v]) => [
            v,
            k.split("__")[1].toLowerCase(),
        ]),
    ),
    natures: Object.fromEntries(
        Object.entries(NaturesEnum).map(([k, v]) => [
            v,
            k.split("__")[1].toLowerCase(),
        ]),
    ),
    typechart: Object.fromEntries(
        Object.entries(TypechartEnum).map(([k, v]) => [
            v,
            k.split("__")[1].toLowerCase(),
        ]),
    ),
    genders: Object.fromEntries(
        Object.entries(GendernameEnum).map(([k, v]) => [
            v,
            k.split("__")[1].toLowerCase(),
        ]),
    ),
};

export type MoveIndex = 0 | 1 | 2 | 3;
export type BenchIndex = MoveIndex | 4 | 5;

export const numBoosts = Object.keys(BoostsEnum).length;
export const numTypes = Object.keys(TypechartEnum).length;
export const numVolatiles = Object.keys(VolatilestatusEnum).length;
export const numSideConditions = Object.keys(SideconditionEnum).length;
export const numBattleMinorArgs = Object.keys(BattleminorargsEnum).length;
export const numBattleMajorArgs = Object.keys(BattlemajorargsEnum).length;
export const numPseudoweathers = Object.keys(PseudoweatherEnum).length;
export const numActionFeatures = Object.keys(ActionEnum).length;

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
export const numPackedSetFeatures = Object.keys(PackedSetFeature).length;

export const NUM_HISTORY = 512;

export const WILDCARDS = ["dynamax", "mega", "zmove", "terastallize", "ultra"];

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
