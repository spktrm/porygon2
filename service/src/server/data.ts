import * as fs from "fs";
import * as path from "path";

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

type ITOSType = {
    species: Record<number, string>;
    items: Record<number, string>;
    abilities: Record<number, string>;
    moves: Record<number, string>;
    natures: Record<number, string>;
    typechart: Record<number, string>;
    genders: Record<number, string>;
};

let _ITOS: ITOSType | null = null;

export const ITOS = new Proxy({} as ITOSType, {
    get(_target, prop: string) {
        if (!_ITOS) {
            _ITOS = {
                species: Object.fromEntries(
                    Object.entries(SpeciesEnum).map(([k, v]) => [v, k.split("__")[1].toLowerCase()]),
                ),
                items: Object.fromEntries(
                    Object.entries(ItemsEnum).map(([k, v]) => [v, k.split("__")[1].toLowerCase()]),
                ),
                abilities: Object.fromEntries(
                    Object.entries(AbilitiesEnum).map(([k, v]) => [v, k.split("__")[1].toLowerCase()]),
                ),
                moves: Object.fromEntries(
                    Object.entries(MovesEnum).map(([k, v]) => [v, k.split("__")[1].toLowerCase()]),
                ),
                natures: Object.fromEntries(
                    Object.entries(NaturesEnum).map(([k, v]) => [v, k.split("__")[1].toLowerCase()]),
                ),
                typechart: Object.fromEntries(
                    Object.entries(TypechartEnum).map(([k, v]) => [v, k.split("__")[1].toLowerCase()]),
                ),
                genders: Object.fromEntries(
                    Object.entries(GendernameEnum).map(([k, v]) => [v, k.split("__")[1].toLowerCase()]),
                ),
            };
        }
        return (_ITOS as any)[prop];
    },
});

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

const constantsContent = fs.readFileSync("../constants/data.json", "utf-8");
const constantsData = JSON.parse(constantsContent);

export const NUM_HISTORY: number = constantsData.NUM_HISTORY;
export const MAX_RATIO_TOKEN: number = constantsData.MAX_RATIO_TOKEN;

export const WILDCARDS = ["dynamax", "mega", "zmove", "terastallize", "ultra"];

function transformJson(
    json: Record<string, Record<string, number>>,
): Record<string, Record<number, string>> {
    const transformed: Record<string, Record<number, string>> = {};

    for (const [key, value] of Object.entries(json)) {
        if (typeof value === "object" && value !== null) {
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

let _jsonDatum: Record<string, Record<number, string>> | null = null;

export function getJsonDatum(): Record<string, Record<number, string>> {
    if (!_jsonDatum) {
        const filePath = "../data/data/data.json";
        const fileContent = fs.readFileSync(filePath, "utf-8");
        _jsonDatum = transformJson(JSON.parse(fileContent));
    }
    return _jsonDatum;
}

/** @deprecated Use getJsonDatum() for lazy loading. Kept for backward compatibility. */
export const jsonDatum = new Proxy({} as Record<string, Record<number, string>>, {
    get(_target, prop: string) {
        return getJsonDatum()[prop];
    },
    ownKeys() {
        return Reflect.ownKeys(getJsonDatum());
    },
    getOwnPropertyDescriptor(_target, prop) {
        const datum = getJsonDatum();
        if (prop in datum) {
            return { configurable: true, enumerable: true, value: (datum as any)[prop] };
        }
        return undefined;
    },
    has(_target, prop) {
        return prop in getJsonDatum();
    },
});

let _sampleTeams: { [format: string]: string[] } | null = null;

export function getSampleTeams(): { [format: string]: string[] } {
    if (!_sampleTeams) {
        // Try __dirname first (works for both compiled dist/ and ts-node src/)
        let teamsPath = path.resolve(__dirname, "sampleTeams.json");
        if (!fs.existsSync(teamsPath)) {
            // Fallback: look in the source directory
            teamsPath = path.resolve(__dirname, "../../src/server/sampleTeams.json");
        }
        _sampleTeams = JSON.parse(fs.readFileSync(teamsPath, "utf-8"));
    }
    return _sampleTeams as { [format: string]: string[] };
}

/** @deprecated Use getSampleTeams() for lazy loading. Kept for backward compatibility. */
export const sampleTeams = new Proxy({} as { [format: string]: string[] }, {
    get(_target, prop: string) {
        return getSampleTeams()[prop];
    },
    ownKeys() {
        return Reflect.ownKeys(getSampleTeams());
    },
    getOwnPropertyDescriptor(_target, prop) {
        const teams = getSampleTeams();
        if (prop in teams) {
            return { configurable: true, enumerable: true, value: (teams as any)[prop] };
        }
        return undefined;
    },
    has(_target, prop) {
        return prop in getSampleTeams();
    },
});
