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
} from "../protos/enums_pb";

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
    mapping: T
): { [k: string]: keyof T } {
    return Object.fromEntries(
        Object.keys(mapping).map((key) => {
            const preKey = key.split("_").pop() ?? "";
            return [preKey.toLowerCase(), key as keyof T];
        })
    );
}

export type MoveIndex = 0 | 1 | 2 | 3;

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

export type Mappings = keyof typeof MappingLookup;

type EnumKeyMappingType = {
    [K in Mappings]: ReturnType<typeof GenerateEnumKeyMapping<EnumMappings>>;
};

export const EnumKeyMapping: EnumKeyMappingType = Object.fromEntries(
    Object.entries(MappingLookup).map(([key, value]) => {
        return [key, GenerateEnumKeyMapping(value)];
    })
) as EnumKeyMappingType;
