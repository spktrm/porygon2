import { toID } from "@pkmn/data";
import {
    AbilitiesEnum,
    ActionsEnum,
    BattlemajorargsEnum,
    BattleminorargsEnum,
    BoostsEnum,
    ConditionEnum,
    EffectEnum,
    EffecttypesEnum,
    GendernameEnum,
    ItemeffecttypesEnum,
    ItemsEnum,
    LastitemeffecttypesEnum,
    MovesEnum,
    PseudoweatherEnum,
    SideconditionEnum,
    SpeciesEnum,
    StatusEnum,
    VolatilestatusEnum,
    WeatherEnum,
} from "../../protos/enums_pb";
import {
    FeatureAdditionalInformation,
    FeatureEntity,
    FeatureMoveset,
    FeatureTurnContext,
    FeatureWeather,
} from "../../protos/features_pb";
import { OneDBoolean } from "./arr";

export type EnumMappings =
    | typeof SpeciesEnum
    | typeof ItemsEnum
    | typeof StatusEnum
    | typeof ActionsEnum
    | typeof ItemeffecttypesEnum
    | typeof LastitemeffecttypesEnum
    | typeof MovesEnum
    | typeof AbilitiesEnum
    | typeof BoostsEnum
    | typeof VolatilestatusEnum
    | typeof SideconditionEnum
    | typeof ConditionEnum
    | typeof WeatherEnum
    | typeof PseudoweatherEnum
    | typeof GendernameEnum
    | typeof BattlemajorargsEnum
    | typeof BattleminorargsEnum
    | typeof EffectEnum;

function GenerateEnumKeyMapping<T extends EnumMappings>(
    mapping: T,
): { [k: string]: keyof T } {
    return Object.fromEntries(
        Object.keys(mapping).map((key) => {
            const preKey = key.split("_").slice(1).join("_") ?? "";
            return [preKey.toLowerCase(), key as keyof T];
        }),
    );
}

export type MoveIndex = 0 | 1 | 2 | 3;
export type BenchIndex = MoveIndex | 4 | 5;

export const MappingLookup = {
    Species: SpeciesEnum,
    Items: ItemsEnum,
    Status: StatusEnum,
    ItemEffect: ItemeffecttypesEnum,
    LastItemEffect: LastitemeffecttypesEnum,
    Move: MovesEnum,
    Gender: GendernameEnum,
    Ability: AbilitiesEnum,
    Boost: BoostsEnum,
    Volatilestatus: VolatilestatusEnum,
    Sidecondition: SideconditionEnum,
    Condition: ConditionEnum,
    Weather: WeatherEnum,
    PseudoWeather: PseudoweatherEnum,
    BattleMinorArg: BattleminorargsEnum,
    BattleMajorArg: BattlemajorargsEnum,
    Actions: ActionsEnum,
    Effect: EffectEnum,
};

export const numBoosts = Object.keys(BoostsEnum).length;
export const numVolatiles = Object.keys(VolatilestatusEnum).length;
export const numSideConditions = Object.keys(SideconditionEnum).length;
export const numAdditionalInformations = Object.keys(
    FeatureAdditionalInformation,
).length;
export const numBattleMinorArgs = Object.keys(BattleminorargsEnum).length;
export const numBattleMajorArgs = Object.keys(BattlemajorargsEnum).length;
export const numPseudoweathers = Object.keys(PseudoweatherEnum).length;

export type Mappings = keyof typeof MappingLookup;

type EnumKeyMappingType = {
    [K in Mappings]: ReturnType<typeof GenerateEnumKeyMapping<EnumMappings>>;
};

export const EnumKeyMapping: EnumKeyMappingType = Object.fromEntries(
    Object.entries(MappingLookup).map(([key, value]) => {
        return [key, GenerateEnumKeyMapping(value)];
    }),
) as EnumKeyMappingType;

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

export const MAX_TS = 65;

export const sideIdMapping: {
    [k in "p1" | "p2"]: 0 | 1;
} = {
    p1: 0,
    p2: 1,
};

export interface SideObject {
    team: Uint8Array;
    boosts: Uint8Array;
    sideConditions: Uint8Array;
    volatileStatus: Uint8Array;
    additionalInformation: Uint8Array;
}

export interface FieldObject {
    weather: Uint8Array;
    turnContext: Uint8Array;
}

export type HistoryStep = [SideObject, SideObject, FieldObject];

export const numPokemonFields = Object.keys(FeatureEntity).length;
export const numTurnContextFields = Object.keys(FeatureTurnContext).length;
export const numWeatherFields = Object.keys(FeatureWeather).length;
export const numMoveFields = Object.keys(FeatureMoveset).length;
export const numMovesetFields = 2 * 10 * numMoveFields;

export const NUM_HISTORY = 8;

export const AllValidActions = new OneDBoolean(10, Uint8Array);
for (let actionIndex = 0; actionIndex < 10; actionIndex++) {
    AllValidActions.set(actionIndex, true);
}
