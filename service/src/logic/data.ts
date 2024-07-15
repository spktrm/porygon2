import {
    AbilitiesEnum,
    BoostsEnum,
    GendersEnum,
    HyphenargsEnum,
    ItemeffectEnum,
    ItemsEnum,
    MovesEnum,
    PseudoweatherEnum,
    SideconditionsEnum,
    SpeciesEnum,
    StatusesEnum,
    VolatilestatusEnum,
    WeathersEnum,
} from "../../protos/enums_pb";
import {
    FeatureEntity,
    FeatureMoveset,
    FeatureTurnContext,
    FeatureWeather,
} from "../../protos/features_pb";
import { LegalActions } from "../../protos/state_pb";

export type EnumMappings =
    | typeof SpeciesEnum
    | typeof ItemsEnum
    | typeof StatusesEnum
    | typeof ItemeffectEnum
    | typeof MovesEnum
    | typeof AbilitiesEnum
    | typeof BoostsEnum
    | typeof VolatilestatusEnum
    | typeof SideconditionsEnum
    | typeof WeathersEnum
    | typeof PseudoweatherEnum
    | typeof GendersEnum
    | typeof HyphenargsEnum;

function GenerateEnumKeyMapping<T extends EnumMappings>(
    mapping: T,
): { [k: string]: keyof T } {
    return Object.fromEntries(
        Object.keys(mapping).map((key) => {
            const preKey = key.split("_").pop() ?? "";
            return [preKey.toLowerCase(), key as keyof T];
        }),
    );
}

export type MoveIndex = 0 | 1 | 2 | 3;
export type BenchIndex = MoveIndex | 4 | 5;

export const MappingLookup = {
    Species: SpeciesEnum,
    Items: ItemsEnum,
    Statuses: StatusesEnum,
    ItemEffects: ItemeffectEnum,
    Moves: MovesEnum,
    Genders: GendersEnum,
    Abilities: AbilitiesEnum,
    Boosts: BoostsEnum,
    Volatilestatus: VolatilestatusEnum,
    Sideconditions: SideconditionsEnum,
    Weathers: WeathersEnum,
    PseudoWeathers: PseudoweatherEnum,
    Hyphenargs: HyphenargsEnum,
};

export const numBoosts = Object.keys(BoostsEnum).length;
export const numVolatiles = Object.keys(VolatilestatusEnum).length;
export const numSideConditions = Object.keys(SideconditionsEnum).length;
export const numHyphenArgs = Object.keys(HyphenargsEnum).length;
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

export const MAX_TS = 100;

export const sideIdMapping: {
    [k in "p1" | "p2"]: 0 | 1;
} = {
    p1: 0,
    p2: 1,
};

export interface SideObject {
    active: Uint8Array;
    boosts: Uint8Array;
    sideConditions: Uint8Array;
    volatileStatus: Uint8Array;
    hyphenArgs: Uint8Array;
}

export interface FieldObject {
    weather: Uint8Array;
    pseudoweather: Uint8Array;
    turnContext: Uint8Array;
}

export type HistoryStep = [SideObject, SideObject, FieldObject];

export const numPokemonFields = Object.keys(FeatureEntity).length;
export const numTurnContextFields = Object.keys(FeatureTurnContext).length;
export const numWeatherFields = Object.keys(FeatureWeather).length;
export const numMoveFields = Object.keys(FeatureMoveset).length;
export const numMovesetFields = 4 * numMoveFields;

export const AllValidActions = new LegalActions();
AllValidActions.setMove1(true);
AllValidActions.setMove2(true);
AllValidActions.setMove3(true);
AllValidActions.setMove4(true);
AllValidActions.setSwitch1(true);
AllValidActions.setSwitch2(true);
AllValidActions.setSwitch3(true);
AllValidActions.setSwitch4(true);
AllValidActions.setSwitch5(true);
AllValidActions.setSwitch6(true);
