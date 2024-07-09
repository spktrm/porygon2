import {
    AbilitiesEnum,
    AbilitiesEnumMap,
    BoostsEnum,
    BoostsEnumMap,
    HyphenargsEnum,
    HyphenargsEnumMap,
    ItemeffectEnum,
    ItemeffectEnumMap,
    ItemsEnum,
    ItemsEnumMap,
    MovesEnum,
    MovesEnumMap,
    SideconditionsEnum,
    SideconditionsEnumMap,
    SpeciesEnum,
    SpeciesEnumMap,
    VolatilestatusEnum,
    VolatilestatusEnumMap,
    WeathersEnum,
    WeathersEnumMap,
} from "../protos/enums_pb";

function GenerateEnumKeyMapping<
    T extends
        | SpeciesEnumMap
        | ItemsEnumMap
        | ItemeffectEnumMap
        | MovesEnumMap
        | AbilitiesEnumMap
        | BoostsEnumMap
        | VolatilestatusEnumMap
        | SideconditionsEnumMap
        | WeathersEnumMap
        | HyphenargsEnumMap
>(mapping: T): { [k: string]: keyof T } {
    return Object.fromEntries(
        Object.keys(mapping).map((key) => {
            const preKey = key.split("_").pop() ?? "";
            return [preKey.toLowerCase(), key as keyof T];
        })
    );
}

export const MappingLookup = {
    Species: SpeciesEnum,
    Items: ItemsEnum,
    ItemEffects: ItemeffectEnum,
    Moves: MovesEnum,
    Abilities: AbilitiesEnum,
    Boosts: BoostsEnum,
    Volatilestatus: VolatilestatusEnum,
    Sideconditions: SideconditionsEnum,
    Weathers: WeathersEnum,
    Hyphenargs: HyphenargsEnum,
} as const;

export type Mappings = keyof typeof MappingLookup;

export const EnumKeyMapping = {
    Species: GenerateEnumKeyMapping(SpeciesEnum),
    Items: GenerateEnumKeyMapping(ItemsEnum),
    ItemEffects: GenerateEnumKeyMapping(ItemeffectEnum),
    Moves: GenerateEnumKeyMapping(MovesEnum),
    Abilities: GenerateEnumKeyMapping(AbilitiesEnum),
    Boosts: GenerateEnumKeyMapping(BoostsEnum),
    Volatilestatus: GenerateEnumKeyMapping(VolatilestatusEnum),
    Sideconditions: GenerateEnumKeyMapping(SideconditionsEnum),
    Weathers: GenerateEnumKeyMapping(WeathersEnum),
    Hyphenargs: GenerateEnumKeyMapping(HyphenargsEnum),
};
