import { AnyObject, PokemonSet, Teams, TeamValidator, toID } from "@pkmn/sim";
import {
    Args,
    BattleInitArgName,
    BattleMajorArgName,
    BattleMinorArgName,
    BattleProgressArgName,
    KWArgs,
    PokemonIdent,
    Protocol,
} from "@pkmn/protocol";
import {
    AbilitiesEnum,
    BattlemajorargsEnum,
    BattleminorargsEnum,
    EffectEnum,
    EffecttypesEnum,
    GendernameEnum,
    ItemeffecttypesEnum,
    ItemsEnum,
    MovesEnum,
    NaturesEnum,
    SideconditionEnum,
    SpeciesEnum,
    StatusEnum,
    TypechartEnum,
    VolatilestatusEnum,
    WeatherEnum,
} from "../../protos/enums_pb";
import {
    EnumMappings,
    MoveIndex,
    NUM_HISTORY,
    jsonDatum,
    lookUpSetsList,
    numActionMaskFeatures,
    numEntityEdgeFeatures,
    numFieldFeatures,
    numInfoFeatures,
    numMoveFeatures,
    numPrivateEntityNodeFeatures,
    numPublicEntityNodeFeatures,
    numRevealedEntityNodeFeatures,
    sets,
} from "./data";
import { Battle, NA, Pokemon, Side } from "@pkmn/client";
import { Ability, Item, BoostID } from "@pkmn/dex-types";
import { ID, MoveTarget, SideID } from "@pkmn/types";
import { Condition, Effect } from "@pkmn/data";
import { OneDBoolean, TypedArray } from "./utils";
import {
    ActionMaskFeature,
    ActionType,
    EntityEdgeFeature,
    EntityEdgeFeatureMap,
    EntityPrivateNodeFeature,
    EntityPublicNodeFeature,
    EntityRevealedNodeFeature,
    FieldFeature,
    FieldFeatureMap,
    InfoFeature,
    MovesetFeature,
    MovesetHasPP,
} from "../../protos/features_pb";
import { TrainablePlayerAI } from "./runner";
import { EnvironmentState } from "../../protos/service_pb";
import { Move } from "@pkmn/dex";

type RemovePipes<T extends string> = T extends `|${infer U}|` ? U : T;
type MajorArgNames =
    | RemovePipes<BattleMajorArgName>
    | RemovePipes<BattleProgressArgName>
    | RemovePipes<BattleInitArgName>;
type MinorArgNames = RemovePipes<BattleMinorArgName>;

const MAX_RATIO_TOKEN = 16384;

const sampleTeams: { [format: string]: string[] } = {
    gen1ou: [
        "Alakazam|||NoAbility|Psychic,SeismicToss,ThunderWave,Recover|||||||]Exeggutor|||Chlorophyll|Psychic,SleepPowder,MegaDrain,Explosion|||||||]Gengar|||none|Hypnosis,Psychic,Thunderbolt,Explosion|||||||]Chansey|||NaturalCure|IceBeam,Thunderbolt,ThunderWave,SoftBoiled|||||||]Snorlax|||Immunity|BodySlam,Earthquake,SelfDestruct,Rest|||||||]Tauros|||Intimidate|BodySlam,Earthquake,Blizzard,HyperBeam|||||||\n",
        "Alakazam|||NoAbility|Psychic,SeismicToss,ThunderWave,Recover|||||||]Exeggutor|||NoAbility|Psychic,SleepPowder,MegaDrain,Explosion|||||||]Starmie|||NoAbility|Psychic,Blizzard,ThunderWave,Recover|||||||]Chansey|||NoAbility|IceBeam,Thunderbolt,ThunderWave,SoftBoiled|||F||||]Snorlax|||NoAbility|BodySlam,SelfDestruct,Reflect,Rest|||||||]Tauros|||NoAbility|BodySlam,Earthquake,Blizzard,HyperBeam|||M||||\n",
        "Alakazam|||NoAbility|Psychic,SeismicToss,ThunderWave,Recover|||||||]Exeggutor|||NoAbility|Psychic,SleepPowder,StunSpore,Rest|||||||]Rhydon|||NoAbility|Earthquake,BodySlam,Substitute,Rest|||||||]Cloyster|||NoAbility|Blizzard,Clamp,Rest,Explosion|||||||]Chansey|||NoAbility|Sing,SeismicToss,ThunderWave,SoftBoiled|||F||||]Tauros|||NoAbility|BodySlam,Earthquake,Rest,HyperBeam|||M||||\n",
        "Alakazam|||Synchronize|Psychic,SeismicToss,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Exeggutor|||Chlorophyll|SleepPowder,Psychic,DoubleEdge,Explosion|||||||]Zapdos|||Pressure|Thunderbolt,DrillPeck,Agility,ThunderWave|||||||]Chansey|||NaturalCure|SeismicToss,Reflect,ThunderWave,SoftBoiled||252,,252,252,252,252||,2,,,,|||]Snorlax|||Immunity|BodySlam,Earthquake,HyperBeam,SelfDestruct|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Earthquake|||||||\n",
        "Alakazam|||Synchronize|Psychic,SeismicToss,ThunderWave,Recover|||||||]Exeggutor|||Chlorophyll|SleepPowder,Psychic,MegaDrain,Explosion|||||||]Slowbro|||none|Amnesia,Surf,ThunderWave,Rest|||||||]Chansey|||NaturalCure|SeismicToss,Reflect,ThunderWave,SoftBoiled|||F||||]Snorlax|||Immunity|BodySlam,Reflect,Earthquake,Rest|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Thunderbolt|||M||||\n",
        "Alakazam|||Synchronize|Psychic,SeismicToss,ThunderWave,Recover|||||||]Exeggutor|||Chlorophyll|SleepPowder,Psychic,MegaDrain,Explosion|||||||]Zapdos|||Pressure|Thunderbolt,DrillPeck,ThunderWave,Agility|||||||]Chansey|||NaturalCure|Reflect,SeismicToss,ThunderWave,SoftBoiled|||||||]Snorlax|||Immunity|BodySlam,Earthquake,HyperBeam,SelfDestruct|||||||]Tauros|||Intimidate|BodySlam,Earthquake,Blizzard,HyperBeam|||||||\n",
        "Alakazam|||Synchronize|Psychic,SeismicToss,ThunderWave,Recover|||||||]Exeggutor|||Chlorophyll|SleepPowder,Psychic,StunSpore,Explosion|||||||]Starmie|||Illuminate|Surf,Thunderbolt,ThunderWave,Recover|||||||]Chansey|||NaturalCure|Sing,SeismicToss,ThunderWave,SoftBoiled|||F||||]Snorlax|||Immunity|BodySlam,Reflect,Earthquake,Rest|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Earthquake|||M||||\n",
        "Alakazam|||Synchronize|Psychic,SeismicToss,ThunderWave,Recover|||||||]Starmie|||Illuminate|Surf,Thunderbolt,ThunderWave,Recover|||||||]Rhydon|||none|Earthquake,BodySlam,Substitute,RockSlide|||||||]Chansey|||NaturalCure|Sing,IceBeam,ThunderWave,SoftBoiled|||F||||]Snorlax|||Immunity|BodySlam,Reflect,IceBeam,Rest|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Earthquake|||M||||\n",
        "Alakazam|||none|Psychic,SeismicToss,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Exeggutor|||none|SleepPowder,Psychic,DoubleEdge,Explosion|||||||]Lapras|||none|Sing,Blizzard,Thunderbolt,HyperBeam|||||||]Chansey|||none|IceBeam,Thunderbolt,ThunderWave,SoftBoiled|||F||||]Snorlax|||none|BodySlam,Reflect,HyperBeam,Rest|||||||]Tauros|||none|BodySlam,HyperBeam,Blizzard,Earthquake|||M||||\n",
        "Alakazam|||none|Psychic,SeismicToss,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Exeggutor|||none|SleepPowder,Psychic,DoubleEdge,Explosion|||||||]Slowbro|||none|Amnesia,Surf,ThunderWave,Rest||252,,252,252,252,252||,2,,,,|||]Chansey|||none|IceBeam,Thunderbolt,ThunderWave,SoftBoiled|||F||||]Snorlax|||none|BodySlam,Reflect,HyperBeam,Rest|||||||]Tauros|||none|BodySlam,HyperBeam,Blizzard,Earthquake|||M||||\n",
        "Gengar|||CursedBody|Hypnosis,NightShade,Thunderbolt,Explosion|||||||]Starmie|||none|Surf,Thunderbolt,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Alakazam|||none|Psychic,SeismicToss,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Chansey|||none|Sing,IceBeam,ThunderWave,SoftBoiled||252,,252,252,252,252|F|,2,,,,|||]Snorlax|||none|BodySlam,Reflect,HyperBeam,Rest|||||||]Tauros|||none|BodySlam,HyperBeam,Blizzard,Earthquake|||M||||\n",
        "Gengar|||NoAbility|Hypnosis,Psychic,Thunderbolt,Explosion|||||||]Exeggutor|||NoAbility|SleepPowder,Psychic,StunSpore,Explosion|||||||]Cloyster|||NoAbility|Blizzard,Clamp,Rest,Explosion|||||||]Chansey|||NaturalCure|SeismicToss,Reflect,ThunderWave,SoftBoiled|||||||]Snorlax|||Immunity|BodySlam,Earthquake,HyperBeam,SelfDestruct|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Thunderbolt|||||||\n",
        "Gengar|||NoAbility|NightShade,Thunderbolt,Hypnosis,Explosion|||||||]Exeggutor|||NoAbility|Psychic,SleepPowder,StunSpore,Explosion|||||||]Starmie|||NoAbility|Surf,Thunderbolt,ThunderWave,Recover|||||||]Zapdos|||NoAbility|Thunderbolt,DrillPeck,ThunderWave,Agility|||||||]Snorlax|||NoAbility|BodySlam,Earthquake,Counter,SelfDestruct|||||||]Tauros|||NoAbility|BodySlam,Earthquake,Blizzard,HyperBeam|||M||||\n",
        "Gengar|||NoAbility|Psychic,Thunderbolt,Hypnosis,Explosion|||||||]Exeggutor|||NoAbility|Psychic,StunSpore,MegaDrain,Explosion|||||||]Starmie|||NoAbility|Surf,Thunderbolt,ThunderWave,Recover|||||||]Chansey|||NoAbility|Sing,SeismicToss,ThunderWave,SoftBoiled|||F||||]Snorlax|||NoAbility|BodySlam,HyperBeam,Reflect,Rest|||||||]Tauros|||NoAbility|BodySlam,Earthquake,Blizzard,HyperBeam|||M||||\n",
        "Gengar|||NoAbility|Psychic,Thunderbolt,Hypnosis,Explosion|||||||]Exeggutor|||NoAbility|Psychic,StunSpore,MegaDrain,Explosion|||||||]Zapdos|||NoAbility|Thunderbolt,DrillPeck,ThunderWave,Agility|||||||]Chansey|||NoAbility|Sing,SeismicToss,ThunderWave,SoftBoiled|||F||||]Snorlax|||NoAbility|BodySlam,Earthquake,Reflect,Rest|||||||]Tauros|||NoAbility|BodySlam,Earthquake,Blizzard,HyperBeam|||M||||\n",
        "Gengar|||none|Hypnosis,NightShade,Thunderbolt,Explosion|||||||]Starmie|||Illuminate|Surf,Thunderbolt,ThunderWave,Recover||252,252,252,252,252,252|||||]Zapdos|||none|Thunderbolt,DrillPeck,ThunderWave,Agility|||||||]Chansey|||NaturalCure|Sing,IceBeam,ThunderWave,SoftBoiled||252,252,252,252,252,252|||||]Snorlax|||Immunity|BodySlam,Reflect,Earthquake,Rest|||M||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Earthquake|||||||\n",
        "Gengar|||none|Hypnosis,NightShade,Thunderbolt,Explosion|||||||]Zapdos|||none|Thunderbolt,DrillPeck,Agility,ThunderWave|||||||]Articuno|||none|Blizzard,Agility,DoubleEdge,HyperBeam|||||||]Chansey|||none|Sing,IceBeam,Counter,SoftBoiled||252,,252,252,252,252|F|,2,,,,|||]Snorlax|||none|BodySlam,Reflect,SelfDestruct,Rest|||||||]Tauros|||none|BodySlam,HyperBeam,Blizzard,Earthquake|||M||||\n",
        "Gengar|||none|Hypnosis,Psychic,Thunderbolt,Explosion|||||||]Cloyster|||ShellArmor|Blizzard,Clamp,Explosion,Rest|||||||]Alakazam|||none|Psychic,SeismicToss,ThunderWave,Recover|||||||]Chansey|||NaturalCure|Sing,IceBeam,ThunderWave,SoftBoiled|||||||]Snorlax|||Immunity|BodySlam,Reflect,HyperBeam,Rest|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Earthquake|||||||\n",
        "Gengar|||none|Hypnosis,Psychic,Thunderbolt,Explosion|||||||]Starmie|||Illuminate|Surf,Thunderbolt,ThunderWave,Recover||252,252,252,252,252,252|||||]Alakazam|||none|Psychic,SeismicToss,ThunderWave,Recover||252,252,252,252,252,252|||||]Chansey|||NaturalCure|Sing,IceBeam,ThunderWave,SoftBoiled||252,252,252,252,252,252|||||]Snorlax|||Immunity|BodySlam,Reflect,HyperBeam,Rest|||M||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Earthquake|||||||\n",
        "Jynx|||NoAbility|Blizzard,BodySlam,LovelyKiss,Rest|||F||||]Exeggutor|||NoAbility|Psychic,StunSpore,MegaDrain,Explosion|||||||]Gengar|||NoAbility|Psychic,Thunderbolt,Hypnosis,Explosion|||||||]Zapdos|||NoAbility|Thunderbolt,DrillPeck,ThunderWave,Agility|||||||]Snorlax|||NoAbility|BodySlam,Earthquake,SelfDestruct,HyperBeam|||||||]Tauros|||NoAbility|BodySlam,Earthquake,Thunderbolt,HyperBeam|||M||||\n",
        "Jynx|||NoAbility|LovelyKiss,Psychic,Blizzard,Rest|||F||||]Starmie|||NoAbility|Psychic,Blizzard,ThunderWave,Recover|||||||]Rhydon|||NoAbility|Earthquake,BodySlam,Substitute,RockSlide|||||||]Chansey|||NoAbility|IceBeam,Thunderbolt,ThunderWave,SoftBoiled|||F||||]Snorlax|||NoAbility|BodySlam,SelfDestruct,Reflect,Rest|||||||]Tauros|||NoAbility|BodySlam,Earthquake,Blizzard,HyperBeam|||M||||\n",
        "Jynx|||Oblivious|Blizzard,Psychic,LovelyKiss,Rest||252,,252,252,252,252||,2,,,,|||]Starmie|||Illuminate|Psychic,Thunderbolt,Recover,ThunderWave||252,,252,252,252,252||,2,,,,|||]Rhydon|||LightningRod|Earthquake,RockSlide,BodySlam,Substitute|||||||]Chansey|||NaturalCure|SeismicToss,Reflect,ThunderWave,SoftBoiled||252,,252,252,252,252||,2,,,,|||]Snorlax|||Immunity|BodySlam,Reflect,HyperBeam,Rest||252,252,252,252,252,252|||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Thunderbolt|||||||\n",
        "Jynx|||Oblivious|LovelyKiss,Blizzard,Psychic,Rest||252,,252,252,252,252||,2,,,,|||]Starmie|||Illuminate|Psychic,Blizzard,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Jolteon|||VoltAbsorb|Thunderbolt,ThunderWave,DoubleKick,Rest|||||||]Chansey|||NaturalCure|IceBeam,Thunderbolt,Counter,SoftBoiled||252,,252,252,252,252||,2,,,,|||]Snorlax|||Immunity|Amnesia,IceBeam,Reflect,Rest|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,FireBlast|||||||\n",
        "Jynx|||Oblivious|LovelyKiss,Blizzard,Psychic,Rest||252,252,252,252,252,252|||||]Starmie|||Illuminate|Surf,Thunderbolt,ThunderWave,Recover||252,252,252,252,252,252|||||]Rhydon|||LightningRod|Earthquake,BodySlam,Substitute,TailWhip|||||||]Chansey|||NaturalCure|IceBeam,Thunderbolt,ThunderWave,SoftBoiled||252,252,252,252,252,252|||||]Snorlax|||Immunity|BodySlam,Reflect,HyperBeam,Rest||252,252,252,252,252,252|||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Earthquake|||||||\n",
        "Jynx|||Oblivious|LovelyKiss,Blizzard,Psychic,Rest||252,252,252,252,252,252|||||]Starmie|||none|Surf,Thunderbolt,ThunderWave,Recover|||||||]Zapdos|||Pressure|Thunderbolt,DrillPeck,ThunderWave,Agility|||||||]Chansey|||NaturalCure|IceBeam,Counter,ThunderWave,SoftBoiled||252,252,252,252,252,252|||||]Snorlax|||Immunity|BodySlam,Reflect,Earthquake,Rest|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Earthquake|||||||\n",
        "Jynx|||Oblivious|LovelyKiss,Blizzard,Psychic,Rest|||||||]Cloyster|||ShellArmor|Blizzard,Clamp,Explosion,Rest|||||||]Alakazam|||Synchronize|Psychic,SeismicToss,ThunderWave,Recover|||||||]Chansey|||NaturalCure|IceBeam,Thunderbolt,ThunderWave,SoftBoiled|||||||]Snorlax|||Immunity|BodySlam,Reflect,HyperBeam,Rest|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Earthquake|||||||\n",
        "Jynx|||Oblivious|LovelyKiss,Blizzard,Psychic,Rest|||||||]Starmie|||NoAbility|Surf,Thunderbolt,ThunderWave,Recover|||||||]Alakazam|||Synchronize|Psychic,SeismicToss,ThunderWave,Recover|||||||]Chansey|||NaturalCure|IceBeam,Counter,ThunderWave,SoftBoiled|||||||]Snorlax|||Immunity|BodySlam,Reflect,HyperBeam,Rest|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Earthquake|||||||\n",
        "Jynx|||Oblivious|LovelyKiss,Psychic,Blizzard,Counter|||F||||]Exeggutor|||NoAbility|Psychic,StunSpore,MegaDrain,Explosion|||||||]Zapdos|||NoAbility|Thunderbolt,DrillPeck,ThunderWave,Agility|||||||]Chansey|||NoAbility|Sing,SeismicToss,ThunderWave,SoftBoiled|||F||||]Snorlax|||NoAbility|BodySlam,Surf,Counter,SelfDestruct|||||||]Tauros|||NoAbility|BodySlam,Earthquake,Blizzard,HyperBeam|||M||||\n",
        "Jynx|||none|LovelyKiss,Blizzard,Psychic,Rest||252,,252,252,252,252|F|,2,,,,|||]Starmie|||none|Psychic,Thunderbolt,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Alakazam|||none|Psychic,SeismicToss,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Chansey|||none|SeismicToss,Reflect,ThunderWave,SoftBoiled||252,,252,252,252,252|F|,2,,,,|||]Snorlax|||none|BodySlam,Reflect,SelfDestruct,Rest|||||||]Tauros|||none|BodySlam,HyperBeam,Blizzard,Earthquake|||M||||\n",
        "Starmie|||Illuminate|Blizzard,Psychic,ThunderWave,Recover|||||||]Cloyster|||ShellArmor|Blizzard,Clamp,Explosion,Rest|||||||]Alakazam|||none|Psychic,SeismicToss,ThunderWave,Recover|||||||]Chansey|||NaturalCure|Sing,IceBeam,ThunderWave,SoftBoiled|||||||]Snorlax|||Immunity|BodySlam,Reflect,HyperBeam,Rest|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Earthquake|||||||\n",
        "Starmie|||Illuminate|Blizzard,Psychic,ThunderWave,Recover|||||||]Exeggutor|||Chlorophyll|SleepPowder,Psychic,DoubleEdge,Explosion|||||||]Alakazam|||Synchronize|Psychic,SeismicToss,ThunderWave,Recover|||||||]Chansey|||NaturalCure|IceBeam,Thunderbolt,ThunderWave,SoftBoiled|||||||]Snorlax|||Immunity|BodySlam,Reflect,HyperBeam,Rest|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Earthquake|||||||\n",
        "Starmie|||Illuminate|Psychic,Blizzard,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Exeggutor|||Chlorophyll|SleepPowder,Psychic,DoubleEdge,Explosion|||||||]Gengar|||CursedBody|Hypnosis,NightShade,Thunderbolt,Explosion|||||||]Chansey|||NaturalCure|IceBeam,Thunderbolt,ThunderWave,SoftBoiled|Bashful|252,,252,252,252,252||,2,,,,|||]Snorlax|||Immunity|Amnesia,BodySlam,Blizzard,Rest|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Earthquake,Blizzard|||||||\n",
        "Starmie|||Illuminate|Psychic,Thunderbolt,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Rhydon|||LightningRod|Earthquake,RockSlide,BodySlam,Substitute|||||||]Slowbro|||Oblivious|Amnesia,Surf,ThunderWave,Rest||252,,252,252,252,252||,2,,,,|||]Chansey|||NaturalCure|Sing,IceBeam,Counter,SoftBoiled||252,,252,252,252,252||,2,,,,|||]Snorlax|||Immunity|BodySlam,Reflect,HyperBeam,Rest|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Earthquake|||||||\n",
        "Starmie|||NoAbility|Blizzard,Thunderbolt,ThunderWave,Recover|||||||]Exeggutor|||Chlorophyll|SleepPowder,Psychic,MegaDrain,Explosion|||||||]Zapdos|||Pressure|Thunderbolt,DrillPeck,ThunderWave,Agility|||||||]Chansey|||NaturalCure|IceBeam,Thunderbolt,ThunderWave,SoftBoiled||252,252,252,252,252,252|||||]Snorlax|||Immunity|BodySlam,Reflect,Earthquake,Rest|||||||]Tauros|||Intimidate|BodySlam,HyperBeam,Blizzard,Earthquake|||||||\n",
        "Starmie|||NoAbility|Psychic,Blizzard,ThunderWave,Recover|||||||]Exeggutor|||NoAbility|Psychic,SleepPowder,StunSpore,Explosion|||||||]Cloyster|||NoAbility|Blizzard,Clamp,Rest,Explosion|||||||]Chansey|||NoAbility|Sing,SeismicToss,ThunderWave,SoftBoiled|||F||||]Snorlax|||NoAbility|BodySlam,Earthquake,HyperBeam,SelfDestruct|||||||]Tauros|||NoAbility|BodySlam,Earthquake,Blizzard,HyperBeam|||M||||\n",
        "Starmie|||NoAbility|Psychic,Blizzard,ThunderWave,Recover|||||||]Exeggutor|||NoAbility|Psychic,SleepPowder,StunSpore,Explosion|||||||]Zapdos|||NoAbility|Thunderbolt,DrillPeck,ThunderWave,Agility|||||||]Chansey|||NoAbility|IceBeam,Thunderbolt,ThunderWave,SoftBoiled|||F||||]Snorlax|||NoAbility|BodySlam,Earthquake,Reflect,Rest|||||||]Tauros|||NoAbility|BodySlam,Earthquake,Blizzard,HyperBeam|||M||||\n",
        "Starmie|||NoAbility|Psychic,Blizzard,ThunderWave,Recover|||||||]Exeggutor|||NoAbility|Psychic,SleepPowder,StunSpore,MegaDrain|||||||]Rhydon|||NoAbility|Earthquake,BodySlam,Substitute,RockSlide|||||||]Chansey|||NoAbility|Sing,SeismicToss,ThunderWave,SoftBoiled|||F||||]Snorlax|||NoAbility|BodySlam,SelfDestruct,Reflect,Rest|||||||]Tauros|||NoAbility|BodySlam,Earthquake,Blizzard,HyperBeam|||M||||\n",
        "Starmie|||NoAbility|Surf,Thunderbolt,ThunderWave,Recover|||||||]Exeggutor|||none|SleepPowder,Psychic,StunSpore,Explosion|||||||]Rhydon|||none|Earthquake,BodySlam,Substitute,TailWhip|||||||]Chansey|||none|Sing,IceBeam,ThunderWave,SoftBoiled||252,252,252,252,252,252|F||||]Snorlax|||none|BodySlam,Reflect,HyperBeam,Rest|||||||]Tauros|||none|BodySlam,HyperBeam,Blizzard,Earthquake|||M||||\n",
        "Starmie|||none|Blizzard,Thunderbolt,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Cloyster|||none|Blizzard,Clamp,Rest,Explosion|||||||]Alakazam|||none|Psychic,SeismicToss,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Chansey|||none|Sing,IceBeam,ThunderWave,SoftBoiled||252,,252,252,252,252|F|,2,,,,|||]Snorlax|||none|BodySlam,Reflect,HyperBeam,Rest|||||||]Tauros|||none|BodySlam,HyperBeam,Blizzard,Thunderbolt|||M||||\n",
        "Starmie|||none|Blizzard,Thunderbolt,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Cloyster|||none|Blizzard,Clamp,Rest,Explosion|||||||]Jolteon|||none|Thunderbolt,DoubleKick,ThunderWave,Rest|||||||]Chansey|||none|Sing,IceBeam,ThunderWave,SoftBoiled||252,,252,252,252,252|F|,2,,,,|||]Snorlax|||none|BodySlam,Reflect,HyperBeam,Rest|||||||]Tauros|||none|BodySlam,HyperBeam,Blizzard,Earthquake|||M||||\n",
        "Starmie|||none|Psychic,Blizzard,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Exeggutor|||none|SleepPowder,Psychic,DoubleEdge,Explosion|||||||]Alakazam|||none|Psychic,SeismicToss,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Chansey|||none|IceBeam,Thunderbolt,ThunderWave,SoftBoiled||252,,252,252,252,252|F|,2,,,,|||]Snorlax|||none|BodySlam,Reflect,Earthquake,Rest|||||||]Tauros|||none|BodySlam,HyperBeam,Blizzard,Earthquake|||M||||\n",
        "Starmie|||none|Psychic,Blizzard,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Exeggutor|||none|SleepPowder,Psychic,StunSpore,Explosion|||||||]Rhydon|||none|Earthquake,BodySlam,Substitute,Rest|||||||]Chansey|||none|IceBeam,Thunderbolt,ThunderWave,SoftBoiled||252,,252,252,252,252|F|,2,,,,|||]Snorlax|||none|BodySlam,Reflect,HyperBeam,Rest|||||||]Tauros|||none|BodySlam,HyperBeam,Blizzard,Earthquake|||M||||\n",
        "Starmie|||none|Surf,Thunderbolt,ThunderWave,Recover||252,,252,252,252,252||,2,,,,|||]Exeggutor|||none|SleepPowder,Psychic,DoubleEdge,Explosion|||||||]Zapdos|||none|Thunderbolt,DrillPeck,ThunderWave,Agility|||||||]Chansey|||none|IceBeam,Thunderbolt,ThunderWave,SoftBoiled|||F||||]Snorlax|||none|BodySlam,Reflect,Earthquake,Rest|||||||]Tauros|||none|BodySlam,HyperBeam,Blizzard,Earthquake|||M||||\n",
    ],
    gen9ou: [
        "Acanthis|TornadusTherian|HeavyDutyBoots|Regenerator|BleakwindStorm,HeatWave,FocusBlast,NastyPlot|Timid|8,,,248,,252||,0,,,,|||,,,,,Steel]Sober to Death|Dondozo|HeavyDutyBoots|Unaware|Liquidation,BodyPress,SleepTalk,Rest|Impish|248,,252,,,8|F||||,,,,,Fairy]Disciples|TingLu|HeavyDutyBoots|VesselofRuin|Earthquake,Spikes,Protect,Rest|Careful|248,,8,,252,|||||,,,,,Ghost]Ptolemaea|Blissey|HeavyDutyBoots|NaturalCure|ShadowBall,CalmMind,SoftBoiled,HealBell|Bold|248,,252,,8,||,0,,,,|||,,,,,Fairy]Triton|Toxapex|HeavyDutyBoots|Regenerator|Surf,Toxic,Haze,Recover|Bold|248,,236,,,24|F|,0,,,,|||,,,,,Fairy]Helios|Corviknight|RockyHelmet|Pressure|IronHead,BodyPress,IronDefense,Roost|Impish|248,,140,,,120|F||||,,,,,Fighting",
        "Alomomola||RedCard|Regenerator|Wish,FlipTurn,Scald,Acrobatics|Relaxed|,4,252,,252,|F|,,,,,0|||,,,,,Flying]Slowking-Galar||HeavyDutyBoots|Regenerator|FutureSight,ChillyReception,IceBeam,SludgeBomb|Sassy|248,,8,,252,|F|,0,,,,0|||,,,,,Water]Kingambit||Leftovers|SupremeOverlord|SwordsDance,IronHead,KowtowCleave,SuckerPunch|Adamant|240,252,,,,16|F||||,,,,,Flying]Great Tusk||RockyHelmet|Protosynthesis|IceSpinner,RapidSpin,HeadlongRush,StealthRock|Jolly|,252,,,4,252|||||,,,,,Dragon]Cinderace||HeavyDutyBoots|Libero|CourtChange,PyroBall,Uturn,WillOWisp|Jolly|,252,,,4,252|F||||,,,,,Grass]Thundurus-Therian||ChoiceSpecs|VoltAbsorb|TeraBlast,VoltSwitch,GrassKnot,KnockOff|Timid|,,,252,4,252|||||,,,,,Flying",
        "Araquanid||CustapBerry|WaterBubble|StickyWeb,Liquidation,MirrorCoat,Lunge|Careful|248,,140,,112,8|||||,,,,,Ghost]Raging Bolt||BoosterEnergy|Protosynthesis|CalmMind,Thunderclap,DragonPulse,Thunderbolt|Modest|,,4,252,,252||,20,,,,|||,,,,,Bug]Iron Moth||BoosterEnergy|QuarkDrive|FieryDance,SludgeWave,Psychic,DazzlingGleam|Timid|,,,252,4,252||,0,,,,|||,,,,,Fairy]Great Tusk||BoosterEnergy|Protosynthesis|HeadlongRush,IceSpinner,HeadSmash,RapidSpin|Jolly|,252,4,,,252|||||,,,,,Ice]Gholdengo||AirBalloon|GoodasGold|NastyPlot,ShadowBall,Psyshock,DazzlingGleam|Timid|,,,252,4,252||,0,,,,|||,,,,,Fairy]Kingambit||LumBerry|SupremeOverlord|SwordsDance,KowtowCleave,TeraBlast,SuckerPunch|Adamant|,252,,,4,252|||||,,,,,Fairy",
        "Arboliva||Leftovers|SeedSower|GigaDrain,EarthPower,Substitute,TeraBlast|Modest|248,,48,168,32,12||,0,,,,|||,,,,,Fire]Gholdengo||ChoiceSpecs|GoodasGold|ShadowBall,MakeItRain,Trick,Psyshock|Timid|,,,252,4,252||,0,,,,|||,,,,,Steel]Garganacl||Leftovers|PurifyingSalt|SaltCure,Recover,BodyPress,IronDefense|Impish|248,,252,,,8|||||,,,,,Fairy]Great Tusk||RockyHelmet|Protosynthesis|KnockOff,BodyPress,StealthRock,RapidSpin|Impish|224,,248,,,36|||||,,,,,Grass]Toxapex||AssaultVest|Regenerator|Surf,SludgeBomb,AcidSpray,Infestation|Modest|224,,,252,,32||,0,,,,|||,,,,,Dark]Meowscarada||ChoiceScarf|Protean|FlowerTrick,KnockOff,Uturn,Trick|Jolly|,252,,,4,252|||||,,,,,Grass",
        "Behemoth|GreatTusk|HeavyDutyBoots|Protosynthesis|HeadlongRush,IceSpinner,RapidSpin,CloseCombat|Jolly|,252,4,,,252|||S||,,,,,Ground]Chaos|Heatran|AirBalloon|FlashFire|StealthRock,MagmaStorm,EarthPower,Taunt|Timid|,,,252,4,252|M|,0,,,,|S||,,,,,Fairy]Bandit|Weavile|ChoiceBand|Pressure|TripleAxel,KnockOff,IceShard,LowKick|Jolly|,252,,,4,252|M||S||,,,,,Ice]Celestine|Primarina|AssaultVest|LiquidVoice|PsychicNoise,Whirlpool,Moonblast,FlipTurn|Modest|248,,,208,52,|M||S||,,,,,Ground]Anonymous|Ogerpon|ChoiceBand|Defiant|IvyCudgel,KnockOff,Uturn,RockTomb|Jolly|,252,,,4,252|F||||,Poison,,,,Grass]Overseer|Pecharunt|RockyHelmet|PoisonPuppeteer|ShadowBall,MalignantChain,Recover,PartingShot|Bold|248,,252,,8,||,0,,,,|||,,,,,Ghost",
        "Blissey||HeavyDutyBoots|NaturalCure|CalmMind,Flamethrower,SeismicToss,SoftBoiled|Calm|4,,252,,252,||,0,,,,|||,,,,,Dark]Clefable||StickyBarb|MagicGuard|KnockOff,Moonblast,Moonlight,Wish|Bold|252,,252,,4,|F||||,,,,,Steel]Gliscor||ToxicOrb|PoisonHeal|Toxic,KnockOff,Spikes,Protect|Impish|244,,252,,12,|F|,,,,,24|||,,,,,Dragon]Ting-Lu||HeavyDutyBoots|VesselofRuin|StealthRock,Earthquake,Rest,Protect|Careful|252,,4,,252,|||||,,,,,Ghost]Dondozo||HeavyDutyBoots|Unaware|Waterfall,Curse,SleepTalk,Rest|Impish|252,,252,,4,|F|,,,,,18|||,,,,,Fighting]Amoonguss||HeavyDutyBoots|Regenerator|Toxic,FoulPlay,Synthesis,SeedBomb|Relaxed|252,,252,,4,|F||||,,,,,Steel",
        "Blissey||HeavyDutyBoots|NaturalCure|SoftBoiled,CalmMind,Flamethrower,SeismicToss|Calm|4,,252,,252,||,0,,,,|||,,,,,Dark]Amoonguss||HeavyDutyBoots|Regenerator|Toxic,FoulPlay,ClearSmog,Synthesis|Bold|248,,252,,8,||,0,,,,|||,,,,,Ghost]Bronzong||Leftovers|Levitate|Protect,PsychicNoise,StealthRock,BodyPress|Calm|248,,8,,252,||,0,,,,|||,,,,,Fighting]Gliscor||ToxicOrb|PoisonHeal|Protect,KnockOff,Spikes,Earthquake|Careful|244,,,,168,96|||||,,,,,Fairy]Dondozo||HeavyDutyBoots|Unaware|Curse,Rest,SleepTalk,Avalanche|Impish|252,,252,,4,|||||,,,,,Fighting]Alomomola||HeavyDutyBoots|Regenerator|Wish,Protect,Scald,FlipTurn|Relaxed|12,,252,,244,||,,,,,0|||,,,,,Steel",
        "Cinderace||HeavyDutyBoots|Blaze|PyroBall,Uturn,WillOWisp,CourtChange|Jolly|224,32,,,,252|M||||,,,,,Flying]Muk|MukAlola|Leftovers|PoisonTouch|KnockOff,PoisonJab,Protect,Taunt|Careful|252,4,,,252,|||||,,,,,Water]Dondozo||Leftovers|Unaware|Liquidation,BodyPress,Protect,Curse|Impish|252,,252,,4,|||||,,,,,Fairy]Landorus|LandorusTherian|RockyHelmet|Intimidate|Earthquake,Uturn,GrassKnot,Taunt|Impish|252,,248,,,8|M||||,,,,,Steel]Zamazenta||Leftovers|DauntlessShield|IronDefense,BodyPress,Crunch,Substitute|Jolly|252,,80,,,176|||||,,,,,Fire]Scream Tail||Leftovers|Protosynthesis|DazzlingGleam,Wish,Protect,Encore|Timid|120,,,,236,152||,0,,,,|||,,,,,Dark",
        "Cinderace||HeavyDutyBoots|Blaze|PyroBall,Uturn,WillOWisp,CourtChange|Jolly|224,32,,,,252|||||,,,,,Ghost]Zapdos||HeavyDutyBoots|Static|VoltSwitch,Hurricane,Roost,ThunderWave|Bold|252,,16,,,240||,0,,,,|||,,,,,Fairy]Ogerpon-Wellspring||WellspringMask|WaterAbsorb|KnockOff,Encore,PowerWhip,IvyCudgel|Jolly|,252,4,,,252|F||||,,,,,Water]Slowking-Galar||HeavyDutyBoots|Regenerator|IceBeam,FutureSight,SludgeBomb,ChillyReception|Bold|252,,208,,,48||,0,,,,|||,,,,,Dragon]Ting-Lu||Leftovers|VesselofRuin|Rest,StealthRock,Earthquake,Whirlwind|Careful|252,,,,240,16|||||,,,,,Ghost]Dondozo||HeavyDutyBoots|Unaware|Avalanche,BodyPress,Rest,SleepTalk|Relaxed|252,,252,,4,||,,,,,29|||,,,,,Fairy",
        "Cinderace||HeavyDutyBoots|Blaze|PyroBall,Uturn,WillOWisp,CourtChange|Jolly|248,,16,,12,232|M||||,,,,,Fire]Toxapex||AssaultVest|Regenerator|Surf,IceBeam,SludgeBomb,Infestation|Modest|248,,,224,,36||,0,,,,|||,,,,,Fairy]Dondozo||Leftovers|Unaware|Curse,Avalanche,Earthquake,Protect|Careful|,252,4,,248,4|||||,,,,,Ground]Taky|Hydreigon|Leftovers|Levitate|Substitute,NastyPlot,DracoMeteor,EarthPower|Timid|,,,252,4,252|F|,0,,,,|||,,,,,Steel]Great Tusk||RockyHelmet|Protosynthesis|BulkUp,Earthquake,KnockOff,RapidSpin|Impish|252,,204,,,52|||||,,,,,Fire]Chisy|ScreamTail|Leftovers|Protosynthesis|DazzlingGleam,Wish,Protect,Encore|Timid|120,,,,236,152||,0,,,,|||,,,,,Fairy",
        "Clefable||StickyBarb|MagicGuard|Moonblast,StealthRock,Moonlight,Encore|Bold|248,,236,,,24||,0,,,,|||,,,,,Steel]Gliscor||ToxicOrb|PoisonHeal|Earthquake,Spikes,Protect,Toxic|Impish|244,,16,,192,56|||||,,,,,Dragon]Milotic||HeavyDutyBoots|MarvelScale|Scald,MirrorCoat,Recover,Haze|Calm|252,,,,140,116||,0,,,,|||,,,,,Dragon]Mandibuzz||HeavyDutyBoots|Overcoat|FoulPlay,KnockOff,Roost,IronDefense|Impish|248,,244,,,16|||||,,,,,Fire]Skeledirge||HeavyDutyBoots|Unaware|TorchSong,Hex,SlackOff,WillOWisp|Bold|248,,88,,172,||,0,,,,|||,,,,,Water]Dragapult||HeavyDutyBoots|Infiltrator|Hex,DracoMeteor,Uturn,WillOWisp|Timid|,,,252,4,252|||||,,,,,Dragon",
        "Clodsire||HeavyDutyBoots|Unaware|Haze,Recover,Toxic,Earthquake|Careful|252,4,,,252,|||||,,,,,Steel]Dondozo||HeavyDutyBoots|Unaware|HeavySlam,BodyPress,Rest,SleepTalk|Impish|252,,252,,4,|||||,,,,,Fairy]Alomomola||HeavyDutyBoots|Regenerator|Whirlpool,Protect,ChillingWater,Wish|Relaxed|252,,252,4,,||,0,,,,0|||,,,,,Flying]Talonflame||HeavyDutyBoots|FlameBody|Defog,WillOWisp,Flamethrower,Roost|Timid|252,,192,,,64||,0,,,,|||,,,,,Ghost]Toxapex||HeavyDutyBoots|Regenerator|Toxic,Recover,Infestation,Haze|Bold|252,,252,,4,||,0,,,,|||,,,,,Fairy]Blissey||HeavyDutyBoots|NaturalCure|StealthRock,SeismicToss,CalmMind,SoftBoiled|Calm|252,,4,,252,||,0,,,,|||,,,,,Fairy",
        "Clodsire||HeavyDutyBoots|Unaware|Recover,Amnesia,Spikes,PoisonJab|Careful|248,,92,,168,|||||,,,,,Steel]Toxapex||HeavyDutyBoots|Regenerator|ToxicSpikes,Toxic,Recover,Haze|Bold|248,,252,,,8||,0,,,,|||,,,,,Fairy]Alomomola||HeavyDutyBoots|Regenerator|Wish,Protect,Whirlpool,ChillingWater|Bold|252,,252,,4,||,0,,,,|||,,,,,Flying]Dondozo||HeavyDutyBoots|Unaware|BodyPress,Rest,SleepTalk,Avalanche|Impish|252,,252,,4,|||||,,,,,Fighting]Blissey||HeavyDutyBoots|NaturalCure|StealthRock,SoftBoiled,Flamethrower,CalmMind|Calm|252,,4,,252,||,0,,,,|||,,,,,Water]Ting-Lu||HeavyDutyBoots|VesselofRuin|Spikes,StealthRock,Rest,Earthquake|Careful|252,,4,,252,|||||,,,,,Steel",
        "Clodsire||HeavyDutyBoots|Unaware|Spikes,Recover,Toxic,Earthquake|Impish|252,,252,,4,|||||,,,,,Water]Gholdengo||HeavyDutyBoots|GoodasGold|NastyPlot,Recover,MakeItRain,ShadowBall|Modest|252,,184,,,72||,0,,,,|||,,,,,Flying]Blissey||HeavyDutyBoots|NaturalCure|StealthRock,SoftBoiled,SeismicToss,ShadowBall|Calm|4,,252,,252,||,0,,,,|||,,,,,Fairy]Alomomola||HeavyDutyBoots|Regenerator|Wish,Protect,ChillingWater,Whirlpool|Impish|252,,252,,4,||,0,,,,|||,,,,,Ghost]Corviknight||RockyHelmet|Pressure|BodyPress,Roost,Defog,Uturn|Impish|252,,252,,4,||,,,,,0|||,,,,,Dark]Talonflame||HeavyDutyBoots|FlameBody|Flamethrower,AirSlash,WillOWisp,Roost|Bold|252,,204,,,52||,0,,,,|||,,,,,Normal",
        "Clodsire||HeavyDutyBoots|Unaware|Spikes,Recover,Toxic,Earthquake|Impish|252,,252,,4,|||||,,,,,Water]Gholdengo||HeavyDutyBoots|GoodasGold|NastyPlot,Recover,ShadowBall,MakeItRain|Modest|252,,252,4,,||,0,,,,|S||,,,,,Flying]Blissey||HeavyDutyBoots|NaturalCure|StealthRock,SoftBoiled,SeismicToss,CalmMind|Bold|252,,252,,4,||,0,,,,|||,,,,,Dark]Alomomola||HeavyDutyBoots|Regenerator|Wish,Protect,ChillingWater,Whirlpool|Impish|252,,252,,4,||,0,,,,|||,,,,,Ghost]Corviknight||RockyHelmet|Pressure|BodyPress,Roost,Defog,Uturn|Impish|252,,252,,4,||,,,,,0|||,,,,,Fairy]Tauros-Paldea-Blaze||HeavyDutyBoots|Intimidate|BodyPress,RagingBull,Rest,WillOWisp|Impish|252,,252,,4,|M||||,,,,,Fighting",
        "Club Paratwice|Zapdos|HeavyDutyBoots|Static|ThunderWave,VoltSwitch,Hurricane,Roost|Bold|252,,240,,,16||,0,,,,|||,,,,,Water]Iron Crown||ChoiceSpecs|QuarkDrive|TachyonCutter,Psyshock,PsychicNoise,VoltSwitch|Timid|,,4,252,,252||,20,,,,|||,,,,,Steel]Garganacl||Leftovers|PurifyingSalt|SaltCure,Earthquake,Curse,Recover|Careful|252,,4,,252,|||||,,,,,Water]Great Tusk||RockyHelmet|Protosynthesis|StealthRock,HeadlongRush,RapidSpin,IceSpinner|Jolly|,252,,,4,252|||||,,,,,Steel]Samurott-Hisui||AssaultVest|Sharpness|CeaselessEdge,RazorShell,SuckerPunch,KnockOff|Adamant|216,112,,,56,124|||||,,,,,Poison]Dragapult||HeavyDutyBoots|Infiltrator|DragonDarts,Hex,WillOWisp,Uturn|Hasty|,76,,180,,252|||||,,,,,Steel",
        "Corviknight||RockyHelmet|Pressure|BodyPress,Uturn,IronDefense,Roost|Impish|248,,200,,60,||,,,,,0|||,,,,,Fighting]Dragapult||Leftovers|Infiltrator|Substitute,DragonDarts,Hex,WillOWisp|Naive|,4,,252,,252|||||,,,,,Fairy]Skeledirge||HeavyDutyBoots|Unaware|TorchSong,Hex,WillOWisp,SlackOff|Bold|232,,56,,176,44||,0,,,,|||,,,,,Water]Ting-Lu||Leftovers|VesselofRuin|Rest,Spikes,Whirlwind,Earthquake|Careful|244,,,,252,12|||||,,,,,Ghost]Toxapex||ShucaBerry|Regenerator|Surf,Toxic,Recover,ToxicSpikes|Calm|248,,140,,120,||,0,,,,|||,,,,,Fairy]Great Tusk||Leftovers|Protosynthesis|StealthRock,Earthquake,KnockOff,RapidSpin|Impish|244,,208,,,56|||||,,,,,Water",
        "Cresselia||Leftovers|Levitate|CalmMind,Moonblast,StoredPower,Moonlight|Bold|252,,200,,,56||,0,,,,|||,,,,,Poison]Dragapult||LightClay|CursedBody|LightScreen,Reflect,WillOWisp,Curse|Timid|252,,,4,,252||,0,,,,|||,,,,,Fairy]Enamorus||LifeOrb|CuteCharm|EarthPower,CalmMind,Moonblast,Agility|Timid|,,4,252,,252||,0,,,,|||,,,,,Ground]Zamazenta||LumBerry|DauntlessShield|IronDefense,BodyPress,HeavySlam,Crunch|Jolly|,240,100,,,168|||||,,,,,Electric]Glimmora||FocusSash|Corrosion|SpikyShield,EnergyBall,Toxic,StealthRock|Timid|,,4,252,,252||,0,,,,|||,,,,,Ghost]Ursaluna||FlameOrb|Guts|Earthquake,Facade,FirePunch,SwordsDance|Adamant|96,252,,,,160|||||,,,,,Fairy",
        "Cuphead|TingLu|Leftovers|VesselofRuin|Spikes,Ruination,Earthquake,Whirlwind|Careful|252,,,,252,4|||||,,,,,Ghost]Dance Armstrong|Gliscor|ToxicOrb|PoisonHeal|SwordsDance,KnockOff,Facade,Protect|Jolly|244,12,,,,252|||||,,,,,Normal]Tchavis Scott|Sinistcha|HeavyDutyBoots|Heatproof|CalmMind,MatchaGotcha,ShadowBall,StrengthSap|Bold|248,,248,,,12||,0,,,,|||,,,,,Poison]My Little Pwny|Keldeo|HeavyDutyBoots|Justified|VacuumWave,Surf,FlipTurn,AuraSphere|Timid|,,4,252,,252|||||,,,,,Water]Tinkki Minaj|Tinkaton|AirBalloon|MoldBreaker|GigatonHammer,ThunderWave,Encore,StealthRock|Jolly|248,,28,,,232|F||||,,,,,Water]Secret Service|Dragonite|HeavyDutyBoots|Multiscale|DragonTail,ExtremeSpeed,Roost,Earthquake|Adamant|248,224,16,,,20|||||,,,,,Normal",
        "Daenerys|Kingambit|Leftovers|SupremeOverlord|KowtowCleave,IronHead,SuckerPunch,SwordsDance|Adamant|160,252,,,,96|F||||,,,,,Dark]Kristine|Cinderace|HeavyDutyBoots|Blaze|PyroBall,WillOWisp,CourtChange,Uturn|Jolly|144,112,,,,252|F||S||,,,,,Flying]Homelandor|LandorusTherian|RockyHelmet|Intimidate|EarthPower,Taunt,StealthRock,Uturn|Bold|248,,244,,,16|||S||,,,,,Dragon]WALL-Y|IronValiant|BoosterEnergy|QuarkDrive|Moonblast,CloseCombat,KnockOff,Encore|Naive|,176,,80,,252|||||,,,,,Ghost]Mr. Freeze|Kyurem|ChoiceSpecs|Pressure|DracoMeteor,FreezeDry,EarthPower,Blizzard|Timid|,,4,252,,252||,0,,,,|S||,,,,,Ice]SlodogChillionaire|SlowkingGalar|HeavyDutyBoots|Regenerator|Toxic,FutureSight,Surf,ChillyReception|Sassy|248,,8,,252,|M|,0,,,,0|S||,,,,,Water",
        "Darkrai||HeavyDutyBoots|BadDreams|DarkPulse,SludgeBomb,FocusBlast,NastyPlot|Timid|,,4,252,,252||,0,,,,|||,,,,,Poison]Manaphy||CovertCloak|Hydration|Surf,IceBeam,EnergyBall,TailGlow|Timid|56,,,200,,252||,0,,,,|||,,,,,Fairy]Landorus-Therian||RockyHelmet|Intimidate|EarthPower,Taunt,StealthRock,Uturn|Timid|248,,36,,,224|||||,,,,,Grass]Gholdengo||AirBalloon|GoodasGold|ShadowBall,MakeItRain,Psyshock,NastyPlot|Modest|208,,,112,,188||,0,,,,|||,,,,,Fairy]Raging Bolt||BoosterEnergy|Protosynthesis|Thunderbolt,DragonPulse,Thunderclap,CalmMind|Modest|196,,4,252,,56||,20,,,,|||,,,,,Bug]Iron Valiant||BoosterEnergy|QuarkDrive|Moonblast,CloseCombat,KnockOff,Thunderbolt|Naive|,16,,240,,252|||||,,,,,Dark",
        "Deoxys-Speed||RockyHelmet|Pressure|StealthRock,PsychoBoost,IceBeam,Taunt|Timid|248,,,200,,60||,0,,,,|||,,,,,Ghost]Dragonite||SpellTag|Multiscale|DragonDance,TeraBlast,LowKick,Encore|Jolly|,252,,,4,252|||||,,,,,Ghost]Darkrai||LifeOrb|BadDreams|DarkPulse,IceBeam,FocusBlast,SuckerPunch|Naive|,4,,252,,252|||||,,,,,Fighting]Iron Moth||BoosterEnergy|QuarkDrive|FieryDance,SludgeWave,DazzlingGleam,ToxicSpikes|Timid|,,124,132,,252||,0,,,,|||,,,,,Fairy]Scizor||ChoiceBand|Technician|BulletPunch,KnockOff,CloseCombat,Uturn|Adamant|116,252,,,,140|||||,,,,,Steel]Great Tusk||BoosterEnergy|Protosynthesis|BulkUp,HeadlongRush,IceSpinner,RapidSpin|Jolly|252,4,,,,252|||||,,,,,Electric",
        "Djengo Unrained|Gholdengo|HeavyDutyBoots|GoodasGold|MakeItRain,ShadowBall,NastyPlot,Recover|Bold|248,,176,,76,8||,0,,,,|||,,,,,Fairy]CClifford|Zamazenta|HeavyDutyBoots|DauntlessShield|CloseCombat,Crunch,StoneEdge,IceFang|Jolly|80,252,,,,176|||||,,,,,Fire]Up Bowl On Em|TingLu|Leftovers|VesselofRuin|Earthquake,Ruination,Whirlwind,Spikes|Careful|252,,,,252,4|||||,,,,,Water]Moonrock 350|Clefable|RockyHelmet|MagicGuard|Moonblast,KnockOff,StealthRock,Moonlight|Bold|252,,232,,24,|F||||,,,,,Water]Secret Service|Dragonite|HeavyDutyBoots|Multiscale|ExtremeSpeed,Earthquake,DragonTail,Roost|Adamant|240,252,,,,16|M||||,,,,,Normal]Dude Nasty Asf|DeoxysSpeed|HeavyDutyBoots|Pressure|PsychoBoost,Thunderbolt,IceBeam,NastyPlot|Modest|80,,,252,,176||,0,,,,|||,,,,,Electric",
        "Dondozo||Leftovers|Unaware|BodyPress,Avalanche,Rest,SleepTalk|Impish|248,,252,,,8|||||,,,,,Dragon]Blissey||Leftovers|NaturalCure|SeismicToss,SoftBoiled,CalmMind,StealthRock|Calm|20,,252,,236,||,0,,,,|||,,,,,Dark]Corviknight||Leftovers|Pressure|IronDefense,Defog,Roost,BodyPress|Impish|252,,252,,4,||,0,,,,|||,,,,,Fighting]Weezing-Galar||HeavyDutyBoots|NeutralizingGas|WillOWisp,Defog,PainSplit,StrangeSteam|Bold|252,,244,,,12||,0,,,,|||,,,,,Ghost]Gliscor||ToxicOrb|PoisonHeal|Earthquake,Spikes,KnockOff,Protect|Careful|244,,12,,252,||,,,,,24|||,,,,,Water]Toxapex||Leftovers|Regenerator|Toxic,Recover,ToxicSpikes,Haze|Calm|252,,,,252,4||,0,,,,|||,,,,,Steel",
        "Dragapult||ChoiceBand|ClearBody|DragonDarts,Uturn,QuickAttack,TeraBlast|Jolly|,252,,,4,252|||S||,,,,,Ghost]Primarina||HeavyDutyBoots|Torrent|CalmMind,Surf,Moonblast,PsychicNoise|Modest|248,,,80,,180||,0,,,,|||,,,,,Fairy]Iron Crown||AssaultVest|QuarkDrive|FutureSight,TachyonCutter,VoltSwitch,FocusBlast|Timid|,,,252,4,252||,20,,,,|||,,,,,Fighting]Samurott-Hisui||HeavyDutyBoots|Sharpness|CeaselessEdge,RazorShell,KnockOff,Encore|Jolly|,252,,,4,252|||||,,,,,Fire]Kingambit||Leftovers|SupremeOverlord|SwordsDance,KowtowCleave,SuckerPunch,IronHead|Adamant|208,252,,,,48|||||,,,,,Ghost]Landorus-Therian||RockyHelmet|Intimidate|StealthRock,EarthPower,Uturn,GrassKnot|Bold|252,,252,,,4|M||||,,,,,Water",
        "Dragapult||HeavyDutyBoots|CursedBody|DragonDarts,Hex,WillOWisp,Uturn|Naive|,60,,196,,252|||||,,,,,Dragon]Dragonite||HeavyDutyBoots|Multiscale|DragonDance,Earthquake,ExtremeSpeed,Roost|Adamant|104,252,,,,152|F||||,,,,,Normal]Ting-Lu||Leftovers|VesselofRuin|StealthRock,Ruination,Earthquake,Whirlwind|Careful|248,,,,252,8|||||,,,,,Water]Weezing-Galar||HeavyDutyBoots|NeutralizingGas|ToxicSpikes,WillOWisp,PainSplit,Defog|Bold|252,,252,,4,|F|,0,,,,|||,,,,,Ghost]Iron Crown||ChoiceSpecs|QuarkDrive|TachyonCutter,Psyshock,FocusBlast,VoltSwitch|Timid|,,4,252,,252||,20,,,,|||,,,,,Steel]Zapdos||HeavyDutyBoots|Static|VoltSwitch,Hurricane,HeatWave,Roost|Timid|40,,,252,,216||,0,,,,|||,,,,,Fairy",
        "Dragapult||HeavyDutyBoots|Infiltrator|DragonDarts,Hex,WillOWisp,Uturn|Naive|,60,,196,,252|||||,,,,,Fairy]Rotom-Wash||Leftovers|Levitate|VoltSwitch,HydroPump,PainSplit,ThunderWave|Bold|252,,212,,,44||,0,,,,|||,,,,,Steel]Landorus-Therian||SoftSand|Intimidate|StealthRock,Earthquake,SmackDown,Uturn|Jolly|8,240,,,8,252|M||||,,,,,Ground]Clefable||StickyBarb|MagicGuard|CalmMind,Flamethrower,Moonblast,Moonlight|Bold|252,,240,,,16||,0,,,,|||,,,,,Bug]Zamazenta||LifeOrb|DauntlessShield|CloseCombat,Crunch,HeavySlam,StoneEdge|Jolly|,252,,,4,252|||||,,,,,Steel]Samurott-Hisui||HeavyDutyBoots|Sharpness|CeaselessEdge,RazorShell,KnockOff,SuckerPunch|Jolly|,236,20,,,252|||||,,,,,Fire",
        "Dragonite||HeavyDutyBoots|Multiscale|DragonClaw,ExtremeSpeed,Earthquake,DragonDance|Adamant|,252,,,4,252|||||,,,,,Normal]Garchomp||FocusSash|RoughSkin|DragonTail,Earthquake,StealthRock,Spikes|Jolly|,252,,,4,252|||||,,,,,Ghost]Kingambit||Leftovers|SupremeOverlord|KowtowCleave,SuckerPunch,LowKick,SwordsDance|Adamant|100,252,,,,156|||||,,,,,Fighting]Iron Moth||BoosterEnergy|QuarkDrive|FieryDance,SludgeWave,DazzlingGleam,TeraBlast|Timid|120,,,132,4,252||,0,,,,|||,,,,,Ground]Gholdengo||EjectPack|GoodasGold|MakeItRain,ShadowBall,NastyPlot,Recover|Timid|96,,,160,,252||,0,,,,|||,,,,,Fairy]Walking Wake||BoosterEnergy|Protosynthesis|HydroPump,DragonPulse,Flamethrower,Agility|Modest|,,,252,4,252||,0,,,,|||,,,,,Water",
        "Dragonite||HeavyDutyBoots|Multiscale|DragonDance,Earthquake,ExtremeSpeed,IceSpinner|Jolly|,252,,,4,252|M||||,,,,,Ground]Ting-Lu||RockyHelmet|VesselofRuin|StealthRock,Earthquake,Ruination,Taunt|Jolly|184,60,,,12,252|||||,,,,,Ghost]Iron Moth||BoosterEnergy|QuarkDrive|FieryDance,SludgeWave,Substitute,DazzlingGleam|Timid|,,124,132,,252|||||,,,,,Ghost]Iron Valiant||BoosterEnergy|QuarkDrive|Moonblast,KnockOff,CloseCombat,Encore|Naive|,132,,124,,252|||||,,,,,Dark]Kyurem||LoadedDice|Pressure|DragonDance,IcicleSpear,ScaleShot,TeraBlast|Adamant|,252,,,4,252|||||,,,,,Fire]Gholdengo||ChoiceScarf|GoodasGold|MakeItRain,ShadowBall,DazzlingGleam,Trick|Timid|,,16,236,4,252||,0,,,,|||,,,,,Fairy",
        "Enamorus||HeavyDutyBoots|Contrary|Moonblast,TeraBlast,EarthPower,Superpower|Naive|,4,,252,,252|||||,,,,,Stellar]Kyurem||ChoiceScarf|Pressure|IceBeam,FreezeDry,EarthPower,DracoMeteor|Modest|,,,252,4,252||,0,,,,|||,,,,,Dragon]Landorus-Therian||RockyHelmet|Intimidate|Earthquake,Uturn,StealthRock,Taunt|Jolly|248,,8,,,252|||||,,,,,Ghost]Kingambit||AirBalloon|SupremeOverlord|SwordsDance,IronHead,LowKick,SuckerPunch|Jolly|,252,,,4,252|F||||,,,,,Ghost]Ogerpon-Wellspring||WellspringMask|WaterAbsorb|IvyCudgel,KnockOff,Uturn,Taunt|Jolly|,252,,,4,252|F||||,,,,,Water]Glimmora||RedCard|ToxicDebris|PowerGem,EarthPower,MortalSpin,Spikes|Bold|244,,216,,,48|F||||,,,,,Ghost",
        "Enamorus||Leftovers|CuteCharm|Agility,CalmMind,EarthPower,Moonblast|Modest|,,,252,4,252||,0,,,,|||,,,,,Ground]Great Tusk||BoosterEnergy|Protosynthesis|BulkUp,CloseCombat,HeadlongRush,IceSpinner|Jolly|252,4,,,,252|||S||,,,,,Ice]Kingambit||BlackGlasses|SupremeOverlord|SwordsDance,KowtowCleave,SuckerPunch,IronHead|Adamant|,252,,,4,252|F||||,,,,,Dark]Zamazenta||LumBerry|DauntlessShield|IronDefense,BodyPress,HeavySlam,Crunch|Jolly|,240,100,,,168|||||,,,,,Electric]Ceruledge||FocusSash|WeakArmor|SwordsDance,BitterBlade,ShadowSneak,CloseCombat|Jolly|,252,,,4,252|F||||,,,,,Flying]Samurott-Hisui||FocusSash|Torrent|CeaselessEdge,KnockOff,Surf,AquaJet|Naive|,252,,4,,252|F||||,,,,,Ghost",
        "Epione|Blissey|HeavyDutyBoots|NaturalCure|StealthRock,SeismicToss,CalmMind,SoftBoiled|Calm|4,,252,,252,||,0,,,,|S||,,,,,Water]Mimic|Ditto|AirBalloon|Imposter|Transform|Quiet|248,,8,252,,||,,,,,0|S||,,,,,Normal]Catfish|Dondozo|HeavyDutyBoots|Unaware|Rest,SleepTalk,BodyPress,Avalanche|Impish|248,,252,,8,|F||S||,,,,,Ghost]Venomous love|Toxapex|HeavyDutyBoots|Regenerator|Toxic,Haze,Recover,ToxicSpikes|Bold|248,,252,,8,||,0,,,,|S||,,,,,Fairy]Fish|Alomomola|HeavyDutyBoots|Regenerator|Wish,Protect,Whirlpool,ChillingWater|Bold|252,,252,,4,||,0,,,,|S||,,,,,Flying]Relic|TingLu|HeavyDutyBoots|VesselofRuin|Spikes,Protect,Rest,Earthquake|Careful|252,4,,,252,||,,,,,29|||,,,,,Steel",
        "Future Joint|IronValiant|ExpertBelt|QuarkDrive|Moonblast,Thunderbolt,ShadowBall,Psyshock|Modest|108,,,252,,148||,0,,,,|||,,,,,Fairy]Dirgiano Duvitsky|Skeledirge|Leftovers|Unaware|TorchSong,ShadowBall,SlackOff,Substitute|Modest|248,,,88,144,28||,0,,,,|||,,,,,Water]Tuskuthy|GreatTusk|Leftovers|Protosynthesis|Earthquake,KnockOff,RapidSpin,StealthRock|Impish|248,16,164,,,80|||||,,,,,Water]Erebus|Kingambit|Leftovers|SupremeOverlord|KowtowCleave,SuckerPunch,IronHead,SwordsDance|Adamant|248,56,,,204,|||||,,,,,Fairy]Triton|Toxapex|CovertCloak|Regenerator|Surf,Toxic,Haze,Recover|Bold|248,,140,,116,4||,0,,,,|||,,,,,Fairy]Shenlong|Dragonite|HeavyDutyBoots|Multiscale|ExtremeSpeed,Earthquake,DragonDance,Roost|Adamant|124,252,,,,132|||||,,,,,Normal",
        "Garchomp||LifeOrb|RoughSkin|Earthquake,DracoMeteor,Spikes,FireBlast|Naive|,4,,252,,252|||||,,,,,Fire]Gholdengo||ChoiceScarf|GoodasGold|Trick,Recover,ShadowBall,MakeItRain|Timid|,,,252,4,252||,0,,,,|||,,,,,Fairy]Toxapex||BlackSludge|Regenerator|Haze,Surf,Recover,Toxic|Calm|252,,,,236,20||,0,,,,|||,,,,,Steel]Garganacl||Leftovers|PurifyingSalt|StealthRock,SaltCure,Recover,Protect|Impish|252,,240,,,16|||||,,,,,Fairy]Great Tusk||Leftovers|Protosynthesis|Earthquake,RapidSpin,KnockOff,BulkUp|Impish|252,,104,,,152|||||,,,,,Water]Iron Valiant||ChoiceSpecs|QuarkDrive|Trick,Moonblast,ShadowBall,Psyshock|Timid|,,,252,4,252||,0,,,,|||,,,,,Fairy",
        "Garganacl||Leftovers|PurifyingSalt|IronDefense,BodyPress,SaltCure,Recover|Careful|248,,,,252,8|F||||,,,,,Electric]Dragapult||HeavyDutyBoots|ClearBody|DracoMeteor,Hex,WillOWisp,Uturn|Timid|,,40,216,,252|F||||,,,,,Dragon]Ogerpon-Wellspring||WellspringMask|WaterAbsorb|SwordsDance,IvyCudgel,Trailblaze,Synthesis|Adamant|224,36,,,,248|F||||,,,,,Water]Moltres||HeavyDutyBoots|FlameBody|Flamethrower,WillOWisp,Roar,Roost|Bold|248,,104,,144,12||,0,,,,|||,,,,,Fairy]Zamazenta||ExpertBelt|DauntlessShield|CloseCombat,HeavySlam,IceFang,Crunch|Jolly|,252,,,4,252|||||,,,,,Dark]Iron Treads||Leftovers|QuarkDrive|StealthRock,Earthquake,KnockOff,RapidSpin|Careful|224,,,,252,32|||||,,,,,Dark",
        "Gholdengo||ChoiceScarf|GoodasGold|ShadowBall,MakeItRain,Trick,NastyPlot|Timid|,,,252,4,252||,0,,,,|||]Great Tusk||Leftovers|Protosynthesis|HeadlongRush,KnockOff,CloseCombat,RapidSpin|Jolly|,252,,,4,252|||||,,,,,Steel]Rotom-Wash||Leftovers|Levitate|VoltSwitch,HydroPump,Protect,WillOWisp|Bold|252,,252,,4,||,0,,,,|||,,,,,Water]Iron Valiant||BoosterEnergy|QuarkDrive|Moonblast,CloseCombat,Thunderbolt,Psyshock|Naive|,4,,252,,252|||||,,,,,Electric]Dragonite||HeavyDutyBoots|Multiscale|DragonDance,FirePunch,Earthquake,ExtremeSpeed|Jolly|,252,4,,,252|||||,,,,,Normal]Ting-Lu||Leftovers|VesselofRuin|Earthquake,StealthRock,Whirlwind,Spikes|Careful|252,,4,,252,|||||,,,,,Poison",
        "Gholdengo||HeavyDutyBoots|GoodasGold|FocusBlast,ThunderWave,Hex,Recover|Bold|252,,248,,,8||,0,,,,|||,,,,,Fairy]Ting-Lu||HeavyDutyBoots|VesselofRuin|Whirlwind,Earthquake,Ruination,Spikes|Careful|252,,4,,252,|||||,,,,,Water]Deoxys-Speed||HeavyDutyBoots|Pressure|KnockOff,IceBeam,PsychoBoost,Superpower|Naive|,200,,252,,56|||||,,,,,Fighting]Garganacl||Leftovers|PurifyingSalt|SaltCure,Protect,StealthRock,Recover|Impish|252,,176,,80,|||||,,,,,Water]Ogerpon||HeavyDutyBoots|Defiant|KnockOff,IvyCudgel,Uturn,Encore|Jolly|,252,4,,,252|F||||,,,,,Grass]Gliscor||ToxicOrb|PoisonHeal|Uturn,Toxic,Earthquake,Protect|Careful|244,,12,,252,||,,,,,28|||,,,,,Water",
        "Gholdengo||HeavyDutyBoots|GoodasGold|NastyPlot,MakeItRain,ShadowBall,Recover|Modest|248,,8,252,,||,0,,,,|||,,,,,Fairy]Iron Valiant||HeavyDutyBoots|QuarkDrive|KnockOff,Moonblast,CloseCombat,Thunderbolt|Naive|,4,,252,,252|||||,,,,,Fairy]Clefable||Leftovers|MagicGuard|StealthRock,Wish,Moonblast,Protect|Bold|252,,252,,,4||,0,,,,|||,,,,,Ghost]Clodsire||HeavyDutyBoots|WaterAbsorb|Spikes,Toxic,Earthquake,Recover|Careful|248,,,,248,12|||||,,,,,Fairy]Skeledirge||HeavyDutyBoots|Unaware|TorchSong,WillOWisp,Hex,SlackOff|Calm|248,,8,,252,||,0,,,,|||,,,,,Fairy]Corviknight||Leftovers|Pressure|IronDefense,Uturn,BodyPress,Roost|Impish|248,,252,,8,|||||,,,,,Fire",
        "Gholdengo||Leftovers|GoodasGold|NastyPlot,Psyshock,ShadowBall,Recover|Bold|176,,188,,,144||,0,,,,|||,,,,,Dark]Gliscor||ToxicOrb|PoisonHeal|Earthquake,KnockOff,Protect,Spikes|Careful|244,,,,252,12|||||,,,,,Water]Garganacl||Leftovers|PurifyingSalt|Curse,SaltCure,Earthquake,Recover|Impish|248,,144,,116,|||||,,,,,Fairy]Empoleon||Leftovers|Competitive|FlipTurn,Roar,Surf,Roost|Calm|248,,,,252,8|||||,,,,,Water]Great Tusk||HeavyDutyBoots|Protosynthesis|RapidSpin,Earthquake,IceSpinner,StealthRock|Impish|216,,252,,,40|||||,,,,,Grass]Zamazenta||HeavyDutyBoots|DauntlessShield|PsychicFangs,CloseCombat,StoneEdge,Crunch|Adamant|,252,4,,,252|||||,,,,,Fighting",
        "Gholdengo||ShucaBerry|GoodasGold|NastyPlot,MakeItRain,ShadowBall,Recover|Modest|192,,,152,16,148||,0,,,,|||,,,,,Steel]Darkrai||HeavyDutyBoots|BadDreams|DarkPulse,SludgeBomb,IceBeam,WillOWisp|Timid|,,,252,4,252||,0,,,,|||,,,,,Poison]Ting-Lu||RockyHelmet|VesselofRuin|StealthRock,Spikes,Earthquake,Ruination|Careful|240,,16,,248,4|||||,,,,,Water]Zapdos||HeavyDutyBoots|Static|VoltSwitch,Hurricane,ThunderWave,Roost|Bold|248,,244,,,16||,0,,,,|||,,,,,Dragon]Primarina||HeavyDutyBoots|Torrent|Moonblast,Surf,Encore,FlipTurn|Modest|76,,,252,,180|||||,,,,,Steel]Gliscor||ToxicOrb|PoisonHeal|SwordsDance,Facade,KnockOff,Protect|Jolly|244,,,,240,24|||||,,,,,Normal",
        "Glimmora||RedCard|ToxicDebris|PowerGem,EarthPower,StealthRock,Spikes|Calm|252,,40,,160,56||,0,,,,|||,,,,,Ghost]Zamazenta||Leftovers|DauntlessShield|IronDefense,BodyPress,Crunch,Substitute|Jolly|112,40,104,,,252|||||,,,,,Fire]Latias||Leftovers|Levitate|DrainingKiss,StoredPower,CalmMind,Agility|Timid|208,,148,68,,84||,0,,,,|||,,,,,Fairy]Gholdengo||CustapBerry|GoodasGold|NastyPlot,ShadowBall,Thunderbolt,DazzlingGleam|Modest|192,,,92,,224||,0,,,,|||,,,,,Fairy]Dragonite||HeavyDutyBoots|Multiscale|DragonDance,ExtremeSpeed,Earthquake,IceSpinner|Adamant|,252,,,4,252|||||,,,,,Normal]Weavile||LifeOrb|Pickpocket|SwordsDance,TripleAxel,UpperHand,KnockOff|Jolly|,252,,,4,252|||||,,,,,Ice",
        "Gliscor||ToxicOrb|PoisonHeal|Earthquake,Toxic,Protect,Spikes|Impish|244,,168,,,96|||||,,,,,Fairy]Alomomola||AssaultVest|Regenerator|FlipTurn,PlayRough,IcyWind,MirrorCoat|Sassy|216,40,,,252,|F|,,,,,0|||,,,,,Poison]Pecharunt||HeavyDutyBoots|PoisonPuppeteer|MalignantChain,FoulPlay,PartingShot,Recover|Bold|248,,108,,,152||,0,,,,|||,,,,,Dark]Great Tusk||RockyHelmet|Protosynthesis|HeadlongRush,IceSpinner,RapidSpin,StealthRock|Jolly|,204,,,52,252|||||,,,,,Ice]Weavile||HeavyDutyBoots|Pressure|TripleAxel,KnockOff,IceShard,SwordsDance|Jolly|,252,,,4,252|F||||,,,,,Ice]Iron Crown||ChoiceSpecs|QuarkDrive|TachyonCutter,PsychicNoise,Psyshock,VoltSwitch|Timid|,,,252,4,252||,20,,,,|||,,,,,Steel",
        "Gliscor||ToxicOrb|PoisonHeal|Facade,KnockOff,SwordsDance,Protect|Impish|244,,88,,,176|F||||,,,,,Normal]Meowscarada||HeavyDutyBoots|Protean|KnockOff,Uturn,TripleAxel,Spikes|Jolly|,252,,,4,252|F||||,,,,,Ghost]Ursaluna||HeavyDutyBoots|Bulletproof|HeadlongRush,IcePunch,Rest,SleepTalk|Adamant|184,56,,,208,60|F||||,,,,,Steel]Skarmory||RockyHelmet|Sturdy|BraveBird,StealthRock,Roost,Whirlwind|Impish|240,44,216,,,8|F||||,,,,,Dragon]Dragapult||HeavyDutyBoots|Infiltrator|Hex,DracoMeteor,Uturn,WillOWisp|Timid|,,,252,4,252|F||||,,,,,Ghost]Slowking-Galar||HeavyDutyBoots|Regenerator|SludgeBomb,PsychicNoise,ThunderWave,ChillyReception|Sassy|248,,8,,252,||,0,,,,0|||,,,,,Water",
        "Gliscor||ToxicOrb|PoisonHeal|SwordsDance,Earthquake,IceFang,Protect|Adamant|244,80,,,36,148|||||,,,,,Fairy]Walking Wake||HeavyDutyBoots|Protosynthesis|DracoMeteor,Scald,KnockOff,Roar|Timid|,,4,252,,252|||||,,,,,Fairy]Slowking-Galar||HeavyDutyBoots|Regenerator|FutureSight,SludgeBomb,Flamethrower,ChillyReception|Sassy|252,,4,,252,||,0,,,,|||,,,,,Water]Zamazenta||HeavyDutyBoots|DauntlessShield|CloseCombat,StoneEdge,Crunch,HeavySlam|Jolly|72,252,,,,184|||||,,,,,Steel]Ting-Lu||Leftovers|VesselofRuin|StealthRock,Spikes,Earthquake,Ruination|Impish|252,,16,,236,4|||||,,,,,Ghost]Kingambit||BlackGlasses|SupremeOverlord|SuckerPunch,SwordsDance,IronHead,KowtowCleave|Adamant|,252,,,4,252|||||,,,,,Dark",
        "Gliscor||ToxicOrb|PoisonHeal|SwordsDance,Facade,Earthquake,Protect|Careful|244,,,,240,24|||||,,,,,Normal]Alomomola||RedCard|Regenerator|Wish,Protect,FlipTurn,Acrobatics|Relaxed|248,,252,,8,||,,,,,0|||,,,,,Flying]Gholdengo||ChoiceScarf|GoodasGold|ShadowBall,MakeItRain,Trick,FocusBlast|Timid|,,4,252,,252||,0,,,,|||,,,,,Fighting]Hoopa-Unbound||AssaultVest|Magician|KnockOff,PsychicNoise,Thunderbolt,GunkShot|Quiet|248,,244,16,,|||||,,,,,Fairy]Cinderace||HeavyDutyBoots|Libero|PyroBall,SuckerPunch,WillOWisp,CourtChange|Jolly|,252,,,4,252|||||,,,,,Fire]Zamazenta||Leftovers|DauntlessShield|IronDefense,BodyPress,Crunch,Roar|Jolly|104,,236,,,168|||||,,,,,Steel",
        "Great Tusk||AssaultVest|Protosynthesis|HeadlongRush,IceSpinner,TemperFlare,RapidSpin|Adamant|8,252,,,64,184|||||,,,,,Fire]Kingambit||AirBalloon|SupremeOverlord|KowtowCleave,LowKick,SuckerPunch,SwordsDance|Adamant|80,252,,,,176|||||,,,,,Ghost]Raging Bolt||LifeOrb|Protosynthesis|DragonPulse,WeatherBall,SolarBeam,Thunderclap|Modest|,,4,252,,252||,20,,,,|||,,,,,Flying]Walking Wake||WiseGlasses|Protosynthesis|HydroSteam,DracoMeteor,WeatherBall,FlipTurn|Timid|,,12,244,,252|||||,,,,,Stellar]Ninetales||HeatRock|Drought|Overheat,ScorchingSands,Encore,HealingWish|Timid|16,,,252,,240||,0,,,,|||,,,,,Ghost]Ceruledge||HeavyDutyBoots|WeakArmor|Poltergeist,BitterBlade,ShadowSneak,SwordsDance|Adamant|232,252,4,,,20|||||,,,,,Fairy",
        "Great Tusk||Leftovers|Protosynthesis|Earthquake,KnockOff,RapidSpin,StealthRock|Impish|248,16,164,,,80|||||,,,,,Water]Breloom||ChoiceBand|Technician|BulletSeed,CloseCombat,MachPunch,RockTomb|Adamant|,252,,,4,252|||||,,,,,Fire]Dragapult||Leftovers|Infiltrator|DragonDarts,Hex,WillOWisp,Substitute|Mild|,80,,176,,252|||||,,,,,Fairy]Kingambit||Leftovers|SupremeOverlord|SwordsDance,IronHead,KowtowCleave,SuckerPunch|Adamant|4,252,,,,252|||||,,,,,Fire]Rotom-Wash||Leftovers|Levitate|HydroPump,VoltSwitch,ThunderWave,Protect|Calm|252,,,,244,12||,0,,,,|||,,,,,Fairy]Cinderace||HeavyDutyBoots|Libero|PyroBall,Uturn,WillOWisp,CourtChange|Jolly|248,,16,,12,232|||||,,,,,Flying",
        "Great Tusk||RockyHelmet|Protosynthesis|HeadlongRush,IceSpinner,RapidSpin,StealthRock|Jolly|252,,,,4,252|||||,,,,,Fire]Landorus-Therian||ChoiceScarf|Intimidate|Earthquake,TeraBlast,StoneEdge,Uturn|Jolly|4,252,,,,252|||||,,,,,Flying]Slowking-Galar||AssaultVest|Regenerator|FutureSight,SludgeBomb,Psyshock,Flamethrower|Relaxed|248,,192,,68,||,0,,,,0|||,,,,,Water]Ogerpon-Wellspring||WellspringMask|WaterAbsorb|IvyCudgel,KnockOff,PowerWhip,Uturn|Adamant|,252,,,4,252|F||||,,,,,Water]Raging Bolt||BoosterEnergy|Protosynthesis|Thunderbolt,Thunderclap,DragonPulse,CalmMind|Modest|120,,,252,,136||,20,,,,|||,,,,,Fairy]Kingambit||Leftovers|SupremeOverlord|IronHead,LowKick,SuckerPunch,SwordsDance|Adamant|240,252,,,,16|||||,,,,,Ghost",
        "Greninja-Bond||LifeOrb|BattleBond|Surf,DarkPulse,GrassKnot,WaterShuriken|Timid|,,,252,4,252|||||,,,,,Ghost]Pelipper||DampRock|Drizzle|Surf,KnockOff,Uturn,Roost|Bold|248,,252,,8,|||||,,,,,Ground]Barraskewda||ChoiceBand|SwiftSwim|AquaJet,FlipTurn,Liquidation,PsychicFangs|Adamant|,252,,,4,252|||||,,,,,Water]Zapdos||HeavyDutyBoots|Static|Thunder,Hurricane,Uturn,Roost|Timid|248,,8,,,252|||||,,,,,Steel]Manaphy||HeavyDutyBoots|Hydration|Surf,StoredPower,TailGlow,Rest|Timid|,,4,252,,252||,0,,,,|||,,,,,Fairy]Ting-Lu||RedCard|VesselofRuin|StealthRock,Earthquake,Whirlwind,Spikes|Careful|252,,4,,252,|||||,,,,,Ghost",
        "Greninja||ChoiceSpecs|Protean|HydroPump,DarkPulse,Spikes,IceBeam|Timid|,,,252,4,252||,0,,,,|||,,,,,Water]Clodsire||HeavyDutyBoots|WaterAbsorb|StealthRock,Earthquake,Toxic,Recover|Careful|248,,8,,252,|||||,,,,,Dark]Gholdengo||CovertCloak|GoodasGold|ShadowBall,MakeItRain,Recover,NastyPlot|Timid|,,,252,4,252||,0,,,,|||,,,,,Flying]Iron Valiant||ChoiceScarf|QuarkDrive|Moonblast,Psyshock,Trick,Thunderbolt|Timid|,,4,252,,252||,0,,,,|||,,,,,Fairy]Great Tusk||Leftovers|Protosynthesis|Earthquake,BodyPress,KnockOff,RapidSpin|Impish|252,,208,,4,44|||||,,,,,Water]Corviknight||Leftovers|Pressure|BraveBird,Uturn,Roost,BodyPress|Impish|248,,252,,8,|||||,,,,,Flying",
        "Hoopa-Unbound||EjectPack|Magician|HyperspaceFury,DrainPunch,Psychic,TrickRoom|Brave|248,252,,8,,||,,,,,0|||,,,,,Dark]Ursaluna||FlameOrb|Guts|Facade,HeadlongRush,FirePunch,SwordsDance|Brave|252,252,,,4,||,,,,,0|||,,,,,Normal]Hatterene||FocusSash|MagicBounce|DazzlingGleam,Psychic,TrickRoom,HealingWish|Quiet|252,,4,252,,||,0,,,,0|||,,,,,Ghost]Cresselia||Leftovers|Levitate|Moonblast,Moonlight,TrickRoom,LunarDance|Calm|252,,176,,80,||,0,,,,0|||,,,,,Poison]Iron Hands||BoosterEnergy|QuarkDrive|WildCharge,DrainPunch,IcePunch,SwordsDance|Adamant|252,252,,,4,|||||,,,,,Flying]Kingambit||Leftovers|SupremeOverlord|KowtowCleave,IronHead,SuckerPunch,SwordsDance|Adamant|252,252,,,4,|||||,,,,,Flying",
        "Hydrapple||HeavyDutyBoots|Regenerator|NastyPlot,LeafStorm,FickleBeam,EarthPower|Modest|224,,,220,,64||,0,,,,|||,,,,,Fairy]Tinkaton||Leftovers|MoldBreaker|GigatonHammer,KnockOff,Encore,StealthRock|Jolly|248,,,,28,232|||||,,,,,Water]Moltres||HeavyDutyBoots|FlameBody|Roar,Flamethrower,Uturn,Roost|Calm|248,,,,216,12|||||,,,,,Fairy]Rotom-Wash||RockyHelmet|Levitate|VoltSwitch,WillOWisp,PainSplit,HydroPump|Bold|248,,216,,,44||,0,,,,|||,,,,,Steel]Great Tusk||HeavyDutyBoots|Protosynthesis|KnockOff,HeadlongRush,RapidSpin,IceSpinner|Jolly|,252,,,4,252|||||,,,,,Fairy]Darkrai||Leftovers|BadDreams|DarkPulse,IceBeam,NastyPlot,ThunderWave|Timid|,,4,252,,252||,0,,,,|||,,,,,Ghost",
        "Hydrapple||HeavyDutyBoots|StickyHold|Recover,BodyPress,GigaDrain,TeraBlast|Bold|244,,252,,12,||,0,,,,27|||,,,,,Ice]Blissey||HeavyDutyBoots|NaturalCure|SoftBoiled,CalmMind,StealthRock,SeismicToss|Calm|4,,252,,252,||,0,,,,|||,,,,,Water]Dondozo||HeavyDutyBoots|Unaware|Waterfall,Rest,SleepTalk,Curse|Impish|252,,252,,4,|||||,,,,,Fighting]Clodsire||HeavyDutyBoots|Unaware|Recover,Amnesia,PoisonSting,Bulldoze|Careful|116,,148,,244,|||||,,,,,Steel]Alomomola||HeavyDutyBoots|Regenerator|Wish,Protect,Scald,FlipTurn|Relaxed|12,,252,,244,||,,,,,0|||,,,,,Dark]Gliscor||ToxicOrb|PoisonHeal|Protect,KnockOff,Spikes,PoisonJab|Impish|244,,252,,12,|||||,,,,,Ghost",
        "Iron Jugulis||BoosterEnergy|QuarkDrive|KnockOff,Hurricane,EarthPower,Taunt|Naive|,4,,252,,252|||||,,,,,Steel]Ogerpon-Wellspring||WellspringMask|WaterAbsorb|IvyCudgel,HornLeech,Spikes,Uturn|Jolly|,252,,,4,252|F||||,,,,,Water]TWINKATON|Tinkaton|AirBalloon|Pickpocket|StealthRock,PlayRough,ThunderWave,Encore|Jolly|248,,,,24,236|||||,,,,,Water]Iron Moth||BoosterEnergy|QuarkDrive|FieryDance,SludgeWave,Psychic,DazzlingGleam|Timid|,,124,132,,252||,0,,,,|||,,,,,Fairy]Great Tusk||BoosterEnergy|Protosynthesis|HeadlongRush,IceSpinner,BulkUp,RapidSpin|Jolly|252,,4,,,252|||||,,,,,Poison]Dragapult||ChoiceSpecs|ClearBody|DracoMeteor,ShadowBall,Thunderbolt,Uturn|Timid|,,,252,4,252|||||,,,,,Ghost",
        "Iron Valiant||ChoiceSpecs|QuarkDrive|FocusBlast,Moonblast,ShadowBall,Psyshock|Timid|,,,252,4,252||,0,,,,|||]Great Tusk||Leftovers|Protosynthesis|RapidSpin,BodyPress,Earthquake,KnockOff|Impish|252,,216,,,40|||||,,,,,Fighting]Iron Moth||HeavyDutyBoots|QuarkDrive|Flamethrower,SludgeWave,ToxicSpikes,MorningSun|Timid|,,,252,4,252||,0,,,,|||]Ting-Lu||Leftovers|VesselofRuin|StealthRock,Earthquake,Ruination,Whirlwind|Careful|252,,4,,252,|||||,,,,,Water]Scizor||Leftovers|Technician|SwordsDance,BulletPunch,CloseCombat,Uturn|Adamant|252,252,,,4,|||||,,,,,Steel]Dragonite||HeavyDutyBoots|Multiscale|DragonDance,ExtremeSpeed,Earthquake,FirePunch|Jolly|,252,,,4,252|||||,,,,,Normal",
        "Kingambit||Leftovers|SupremeOverlord|SwordsDance,IronHead,KowtowCleave,SuckerPunch|Adamant|112,252,,,,144|||||,,,,,Fire]Glimmora||ChoiceScarf|ToxicDebris|SludgeWave,EarthPower,PowerGem,Spikes|Timid|,,,252,4,252||,0,,,,|||,,,,,Ground]Great Tusk||FocusSash|Protosynthesis|StealthRock,RapidSpin,HeadlongRush,CloseCombat|Adamant|,252,,,4,252|||||,,,,,Water]Hatterene||Leftovers|MagicBounce|CalmMind,StoredPower,DrainingKiss,Protect|Bold|252,,180,,,76||,0,,,,|S||,,,,,Water]Gholdengo||CovertCloak|GoodasGold|ShadowBall,MakeItRain,Recover,NastyPlot|Modest|176,,,252,12,68||,0,,,,|||,,,,,Flying]Dragonite||RockyHelmet|Multiscale|BodyPress,Hurricane,IceBeam,Roost|Bold|252,,252,4,,||,0,,,,|||,,,,,Water",
        "Kurama Chakra Mode|Ninetales|HeatRock|Drought|Flamethrower,Encore,HealingWish,WillOWisp|Timid|248,,20,,,240||,0,,,,|||,,,,,Ghost]Slayboi Carti|Ceruledge|FocusSash|WeakArmor|SwordsDance,BitterBlade,SolarBlade,TeraBlast|Jolly|,252,4,,,252|||||,,,,,Fairy]Venu Vidi Vici|Venusaur|LifeOrb|Chlorophyll|Growth,GigaDrain,SludgeBomb,WeatherBall|Modest|,,4,252,,252||,0,,,,|||,,,,,Fire]Die w A Smile|Hatterene|EjectButton|MagicBounce|HealingWish,PsychicNoise,DazzlingGleam,Nuzzle|Bold|252,,176,,72,8|F||||,,,,,Steel]Lynrd Spynrd|GreatTusk|RockyHelmet|Protosynthesis|HeadlongRush,IceSpinner,RapidSpin,StealthRock|Jolly|252,4,,,,252|||||,,,,,Fire]Dubya|WalkingWake|ChoiceSpecs|Protosynthesis|HydroSteam,DracoMeteor,Flamethrower,DragonPulse|Modest|,,4,252,,252||,0,,,,|||,,,,,Water",
        "Kyotaro|Kyurem|Leftovers|Pressure|Substitute,Protect,FreezeDry,EarthPower|Timid|56,,,200,,252||,0,,,,|S||,,,,,Ground]Goliath|TingLu|Leftovers|VesselofRuin|StealthRock,Rest,Ruination,Earthquake|Careful|248,,8,,252,|||||,,,,,Fairy]Yami|Corviknight|RockyHelmet|Pressure|Roost,BraveBird,Uturn,IronDefense|Relaxed|248,,252,,8,|M|,,,,,0|S||,,,,,Water]Sentinel|WeezingGalar|HeavyDutyBoots|NeutralizingGas|WillOWisp,StrangeSteam,Defog,PainSplit|Bold|248,,252,,8,|M|,0,,,,|S||,,,,,Fairy]Trojan|Zamazenta|ChoiceBand|DauntlessShield|CloseCombat,StoneEdge,HeavySlam,Crunch|Jolly|,252,4,,,252|||S||,,,,,Fighting]Serpentine|Toxapex|AssaultVest|Regenerator|Surf,SludgeBomb,IceBeam,AcidSpray|Modest|248,,8,252,,|M|,0,,,,|S||,,,,,Water",
        "Kyurem||Leftovers|Pressure|Substitute,EarthPower,FreezeDry,Protect|Timid|52,,,204,,252||,0,,,,|||,,,,,Ground]Corviknight||RockyHelmet|Pressure|Defog,BraveBird,Roost,Uturn|Impish|248,,252,,8,|||||,,,,,Dragon]Ting-Lu||Leftovers|VesselofRuin|Earthquake,Payback,Rest,SleepTalk|Careful|252,,4,,252,|||||,,,,,Water]Dondozo||Leftovers|Unaware|Curse,Waterfall,BodyPress,Rest|Impish|248,,252,,8,|||||,,,,,Dark]Slowking-Galar||HeavyDutyBoots|Regenerator|FutureSight,SludgeBomb,Toxic,ChillyReception|Sassy|252,,16,,240,||,0,,,,0||99|,,,,,Water]Cinderace||HeavyDutyBoots|Blaze|PyroBall,WillOWisp,Uturn,CourtChange|Jolly|232,24,,,,252|||||,,,,,Flying",
        "Kyurem||Leftovers|Pressure|Substitute,EarthPower,FreezeDry,Protect|Timid|64,,,220,,224||,0,,,,|||,,,,,Ground]Corviknight||RockyHelmet|Pressure|Defog,BraveBird,Roost,Uturn|Impish|248,,252,,8,|||||,,,,,Dragon]Ting-Lu||Leftovers|VesselofRuin|Earthquake,Payback,Rest,SleepTalk|Careful|252,,4,,252,|||||,,,,,Water]Dondozo||Leftovers|Unaware|Curse,Waterfall,BodyPress,Rest|Impish|248,,252,,8,|||||,,,,,Dark]Slowking-Galar||HeavyDutyBoots|Regenerator|FutureSight,SludgeBomb,Toxic,ChillyReception|Sassy|252,,16,,240,||,0,,,,0||99|,,,,,Water]Cinderace||HeavyDutyBoots|Blaze|PyroBall,WillOWisp,Uturn,CourtChange|Jolly|232,24,,,,252|||||,,,,,Flying",
        "Kyurem||LoadedDice|Pressure|DragonDance,IcicleSpear,TeraBlast,ScaleShot|Adamant|,252,,,4,252|||||,,,,,Electric]Iron Moth||BoosterEnergy|QuarkDrive|FieryDance,SludgeWave,DazzlingGleam,Psychic|Timid|,,124,132,,252||,0,,,,|||,,,,,Fairy]Kingambit||Leftovers|SupremeOverlord|IronHead,SuckerPunch,LowKick,SwordsDance|Adamant|200,252,,,,56|M||||,,,,,Ghost]Landorus-Therian||RockyHelmet|Intimidate|StealthRock,Earthquake,StoneEdge,GrassKnot|Jolly|252,,4,,,252|M||||,,,,,Water]Hatterene||CustapBerry|MagicBounce|HealingWish,DazzlingGleam,PsychicNoise,Nuzzle|Bold|252,,204,,,52|F||||,,,,,Water]Zamazenta||Leftovers|DauntlessShield|IronDefense,BodyPress,Crunch,Substitute|Jolly|104,124,96,,,184|||||,,,,,Fire",
        "Landorus-Therian||RockyHelmet|Intimidate|EarthPower,Uturn,StealthRock,Taunt|Timid|252,,,4,,252|||||,,,,,Dragon]Zamazenta||Leftovers|DauntlessShield|IronDefense,BodyPress,Roar,Crunch|Jolly|252,,80,,,176|||||,,,,,Fire]Raging Bolt||BoosterEnergy|Protosynthesis|CalmMind,Thunderbolt,Thunderclap,DragonPulse|Modest|4,,,252,,252||,20,,,,|||,,,,,Fairy]Darkrai||HeavyDutyBoots|BadDreams|WillOWisp,DarkPulse,IceBeam,SludgeBomb|Timid|,,,252,4,252||,0,,,,|||,,,,,Poison]Gholdengo||AirBalloon|GoodasGold|Hex,ThunderWave,Recover,MakeItRain|Bold|252,,196,,,60|||||,,,,,Fairy]Dragonite||HeavyDutyBoots|Multiscale|DragonDance,Earthquake,ExtremeSpeed,Encore|Adamant|,252,,,4,252|||||,,,,,Normal",
        "Landorus-Therian||RockyHelmet|Intimidate|Earthquake,StoneEdge,Uturn,StealthRock|Jolly|248,,28,,,232|||||,,,,,Dragon]Zamazenta||HeavyDutyBoots|DauntlessShield|CloseCombat,Crunch,StoneEdge,HeavySlam|Jolly|,252,,,4,252|||||,,,,,Fire]Kingambit||Leftovers|SupremeOverlord|IronHead,LowKick,SuckerPunch,SwordsDance|Adamant|120,136,,,,252|F||||,,,,,Ghost]Samurott-Hisui||AssaultVest|Sharpness|CeaselessEdge,RazorShell,SuckerPunch,KnockOff|Adamant|248,144,,,56,60|||||,,,,,Poison]Pecharunt||HeavyDutyBoots|PoisonPuppeteer|Hex,MalignantChain,PartingShot,Recover|Timid|248,,,8,,252||,0,,,,|||,,,,,Ghost]Hydreigon||Leftovers|Levitate|DracoMeteor,FlashCannon,Substitute,NastyPlot|Timid|48,,36,188,,236||,0,,,,|||,,,,,Steel",
        "Leviathan|Dragapult|ChoiceBand|Infiltrator|DragonDarts,Uturn,TeraBlast,SuckerPunch|Adamant|,252,,,4,252|M||S||,,,,,Ghost]Overgod|Kingambit|LumBerry|SupremeOverlord|KowtowCleave,SwordsDance,TeraBlast,SuckerPunch|Adamant|,252,4,,,252|M||S||,,,,,Fairy]Spirit Rush|Gholdengo|ChoiceSpecs|GoodasGold|FocusBlast,Psyshock,ShadowBall,MakeItRain|Timid|,,,252,4,252||,0,,,,|||,,,,,Ghost]Calamity|IronValiant|BoosterEnergy|QuarkDrive|KnockOff,SpiritBreak,SwordsDance,CloseCombat|Jolly|,252,4,,,252|||S||,,,,,Ghost]Phantom|RotomWash|ChoiceScarf|Levitate|HydroPump,Trick,VoltSwitch,TeraBlast|Timid|,,,252,4,252||,0,,,,|S||,,,,,Fairy]Zangetsu|Garchomp|LifeOrb|RoughSkin|DracoMeteor,FireBlast,Earthquake,StealthRock|Naive|,4,,252,,252|M||S||,,,,,Fire",
        "Liberacy|Cinderace|HeavyDutyBoots|Libero|PyroBall,GunkShot,Uturn,ZenHeadbutt|Adamant|,252,4,,,252|M||S||,,,,,Poison]Behemoth|GreatTusk|BoosterEnergy|Protosynthesis|StealthRock,HeadlongRush,KnockOff,RapidSpin|Jolly|248,4,4,,,252|||S||,,,,,Ground]Phantom|Gholdengo|ChoiceScarf|GoodasGold|Trick,ShadowBall,MakeItRain,FocusBlast|Timid|,,,252,4,252||,0,,,,|S||,,,,,Fighting]Overgod|Kingambit|ChoiceBand|SupremeOverlord|SuckerPunch,IronHead,KowtowCleave,LowKick|Adamant|248,252,8,,,|M||S||,,,,,Dark]Fin...|RotomWash|Leftovers|Levitate|Protect,VoltSwitch,HydroPump,WillOWisp|Bold|248,,252,,8,||,0,,,,|S||,,,,,Ghost]Leviathan|Dragapult|ChoiceSpecs|Infiltrator|ShadowBall,DracoMeteor,Flamethrower,Uturn|Timid|,,,252,4,252|M||S||,,,,,Ghost",
        "Lloyd|Torkoal|HeatRock|Drought|Overheat,RapidSpin,StealthRock,Earthquake|Quiet|104,,,236,168,|F||||,,,,,Fairy]Randy|GreatTusk|CovertCloak|Protosynthesis|Earthquake,IceSpinner,BulkUp,RapidSpin|Jolly|252,4,,,,252|||||,,,,,Steel]Noel|SlitherWing|AssaultVest|Protosynthesis|Uturn,FirstImpression,Earthquake,LowKick|Adamant|168,252,,,,88|||||,,,,,Electric]Elie|Venusaur|LifeOrb|Chlorophyll|Growth,GigaDrain,SludgeBomb,WeatherBall|Timid|,,4,252,,252||29,0,,,,|||,,,,,Fire]Tio|WalkingWake|ChoiceSpecs|Protosynthesis|HydroSteam,DracoMeteor,Flamethrower,FlipTurn|Timid|8,,4,244,,252|||||,,,,,Water]Wazy|Heatran|AirBalloon|FlashFire|MagmaStorm,EarthPower,SolarBeam,Taunt|Modest|224,,,40,,244||,0,,,,|||,,,,,Ghost",
        "Lokix||ChoiceBand|TintedLens|FirstImpression,Uturn,KnockOff,LeechLife|Adamant|40,252,,,,216|||||,,,,,Bug]Cinderace||HeavyDutyBoots|Blaze|WillOWisp,Uturn,PyroBall,CourtChange|Jolly|252,24,,,,232|||||,,,,,Ghost]Great Tusk||Leftovers|Protosynthesis|BulkUp,RapidSpin,Earthquake,KnockOff|Impish|252,,92,,,164|||||,,,,,Poison]Alomomola||AssaultVest|Regenerator|FlipTurn,MirrorCoat,AquaJet,BodySlam|Sassy|252,,4,,252,||,,,,,0|||,,,,,Normal]Enamorus||ChoiceScarf|Contrary|Moonblast,EarthPower,TeraBlast,HealingWish|Modest|,,,252,4,252|||||,,,,,Stellar]Slowking-Galar||ShucaBerry|Regenerator|ChillyReception,FutureSight,SludgeBomb,IceBeam|Relaxed|252,,252,4,,||,0,,,,0|||,,,,,Water",
        "Lokix||HeavyDutyBoots|TintedLens|FirstImpression,Uturn,KnockOff,SuckerPunch|Adamant|,252,4,,,252|||||,,,,,Bug]Pecharunt||HeavyDutyBoots|PoisonPuppeteer|MalignantChain,FoulPlay,PartingShot,Recover|Bold|252,,228,,,28||,0,,,,|||,,,,,Dark]Zamazenta||HeavyDutyBoots|DauntlessShield|CloseCombat,Crunch,StoneEdge,IceFang|Jolly|,252,,,4,252|||||,,,,,Fire]Ting-Lu||Leftovers|VesselofRuin|StealthRock,Whirlwind,Earthquake,Ruination|Careful|252,,,,240,16|||||,,,,,Ghost]Latios||SoulDew|Levitate|DracoMeteor,PsychicNoise,FlipTurn,Recover|Timid|,,4,252,,252|M||||,,,,,Steel]Gliscor||ToxicOrb|PoisonHeal|Spikes,KnockOff,Uturn,Protect|Careful|244,,36,,228,||,,,,,30|||,,,,,Water",
        "Love|Meowscarada|ChoiceBand|Protean|KnockOff,Uturn,FlowerTrick,ThunderPunch|Jolly|,252,,,4,252|||||,,,,,Fire]Peace|Skeledirge|HeavyDutyBoots|Unaware|Hex,TorchSong,SlackOff,WillOWisp|Calm|248,,,,244,16||,0,,,,|||,,,,,Fairy]Kindness|GreatTusk|HeavyDutyBoots|Protosynthesis|RapidSpin,HeadlongRush,StealthRock,KnockOff|Jolly|,252,,,4,252|||||,,,,,Steel]Friendship|RotomWash|Leftovers|Levitate|WillOWisp,Protect,HydroPump,VoltSwitch|Calm|248,,148,,112,||,0,,,,23|||,,,,,Fairy]Violence|Arboliva|Leftovers|SeedSower|Substitute,GigaDrain,EarthPower,TeraBlast|Modest|240,,,252,,16||,0,,,,|||,,,,,Fire]Justice|Kingambit|Leftovers|SupremeOverlord|SwordsDance,KowtowCleave,IronHead,SuckerPunch|Adamant|240,252,,,,16|||||,,,,,Fire",
        "Maushold||WideLens|Technician|TidyUp,PopulationBomb,Bite,Encore|Jolly|,252,,,4,252|||||,,,,,Ghost]Rillaboom||ChoiceBand|GrassySurge|WoodHammer,KnockOff,GrassyGlide,Uturn|Adamant|,252,4,,,252|||||,,,,,Grass]Great Tusk||Leftovers|Protosynthesis|BulkUp,HeadlongRush,RapidSpin,IceSpinner|Jolly|252,4,,,,252|||||,,,,,Ice]Heatran||Leftovers|FlashFire|StealthRock,Taunt,EarthPower,MagmaStorm|Calm|252,,,,136,120||,0,,,,|||,,,,,Grass]Enamorus||ChoiceScarf|Contrary|Moonblast,EarthPower,MysticalFire,HealingWish|Timid|,,,252,4,252|F|,0,,,,|||,,,,,Fairy]devindian|Kingambit|Leftovers|SupremeOverlord|SwordsDance,KowtowCleave,SuckerPunch,IronHead|Adamant|236,252,,,,20|||||,,,,,Flying",
        "Meruem|Kingambit|Leftovers|SupremeOverlord|SuckerPunch,IronHead,KowtowCleave,SwordsDance|Adamant|164,252,,,,92|||||,,,,,Ghost]Moebius|DeoxysSpeed|LifeOrb|Pressure|NastyPlot,FocusBlast,PsychoBoost,ShadowBall|Modest|28,,,252,,228||,0,,,,|||,,,,,Psychic]Anomaly|GreatTusk|BoosterEnergy|Protosynthesis|HeadlongRush,IceSpinner,RapidSpin,CloseCombat|Jolly|,252,4,,,252|||||,,,,,Steel]Yoshi|Dragonite|ChoiceBand|Multiscale|Outrage,ExtremeSpeed,IceSpinner,FirePunch|Adamant|16,252,,,,240|||S||,,,,,Normal]Anomаly|IronMoth|BoosterEnergy|QuarkDrive|FieryDance,SludgeWave,TeraBlast,ToxicSpikes|Timid|,,124,132,,252|||||,,,,,Ground]Siren|Primarina|AssaultVest|Torrent|Surf,Moonblast,Whirlpool,PsychicNoise|Modest|76,,,252,,180||,0,,,,|||,,,,,Poison",
        "Moltres||HeavyDutyBoots|FlameBody|Flamethrower,Roar,Uturn,Roost|Bold|248,,248,,,12|||||,,,,,Grass]Zamazenta||HeavyDutyBoots|DauntlessShield|CloseCombat,Crunch,StoneEdge,IceFang|Jolly|,252,,,4,252|||||,,,,,Fire]Darkrai||HeavyDutyBoots|BadDreams|DarkPulse,KnockOff,SludgeBomb,IceBeam|Timid|,,4,252,,252|||S||,,,,,Poison]Hydrapple||HeavyDutyBoots|Regenerator|NastyPlot,FickleBeam,GigaDrain,EarthPower|Bold|244,,172,88,,4|M|,0,,,,|||,,,,,Poison]Ting-Lu||Leftovers|VesselofRuin|Spikes,Ruination,Earthquake,Whirlwind|Careful|248,,8,,252,|||||,,,,,Water]Tinkaton||AirBalloon|MoldBreaker|StealthRock,GigatonHammer,Encore,ThunderWave|Jolly|240,36,,,,232|||||,,,,,Ghost",
        "Moltres||HeavyDutyBoots|FlameBody|Flamethrower,ScorchingSands,Roar,Roost|Bold|248,,236,,,24||,0,,,,|||,,,,,Grass]Gliscor||ToxicOrb|PoisonHeal|Earthquake,Uturn,StealthRock,Protect|Careful|244,,36,,228,|||||,,,,,Fairy]Zamazenta||HeavyDutyBoots|DauntlessShield|CloseCombat,Crunch,StoneEdge,Roar|Jolly|,252,,,4,252|||||,,,,,Dark]Great Tusk||Leftovers|Protosynthesis|HeadlongRush,IceSpinner,KnockOff,RapidSpin|Jolly|,252,4,,,252|||||,,,,,Ground]Slowking-Galar||HeavyDutyBoots|Regenerator|SludgeBomb,FutureSight,ThunderWave,ChillyReception|Sassy|252,,16,,240,||,0,,,,0|||,,,,,Water]Kyurem||Leftovers|Pressure|Substitute,Protect,FreezeDry,EarthPower|Timid|56,,,200,,252||,0,,,,|||,,,,,Ground",
        "Ogerpon-Wellspring||WellspringMask|WaterAbsorb|IvyCudgel,HornLeech,PlayRough,Taunt|Jolly|,252,,,4,252|F||||,,,,,Water]Kingambit||Leftovers|SupremeOverlord|SwordsDance,KowtowCleave,IronHead,SuckerPunch|Adamant|240,252,,,,16|||||,,,,,Ghost]Landorus-Therian||RockyHelmet|Intimidate|Earthquake,Uturn,StealthRock,Taunt|Jolly|248,,8,,,252|||||,,,,,Fairy]Glimmora||PowerHerb|ToxicDebris|MeteorBeam,EarthPower,MortalSpin,Spikes|Modest|4,,8,248,,248|||||,,,,,Ghost]Dragapult||ChoiceSpecs|ClearBody|ShadowBall,DracoMeteor,Flamethrower,Uturn|Timid|,,4,252,,252|||||,,,,,Ghost]Iron Valiant||BoosterEnergy|QuarkDrive|SwordsDance,SpiritBreak,KnockOff,Encore|Jolly|,252,,,4,252|||||,,,,,Steel",
        "Ogerpon-Wellspring||WellspringMask|WaterAbsorb|IvyCudgel,KnockOff,Uturn,Spikes|Jolly|,252,,,4,252|F||||,,,,,Water]Garganacl||Leftovers|PurifyingSalt|SaltCure,StealthRock,Protect,Recover|Careful|252,,52,,204,|||||,,,,,Fairy]Great Tusk||HeavyDutyBoots|Protosynthesis|HeadlongRush,IceSpinner,KnockOff,RapidSpin|Jolly|,252,,,4,252|||||,,,,,Ice]Dragonite||HeavyDutyBoots|Multiscale|ExtremeSpeed,Earthquake,IceSpinner,DragonDance|Adamant|,252,,,4,252|||||,,,,,Normal]Moltres||HeavyDutyBoots|FlameBody|Flamethrower,BraveBird,Roost,WillOWisp|Adamant|248,16,232,,,12|||||,,,,,Fairy]Darkrai||ChoiceScarf|BadDreams|DarkPulse,IceBeam,SludgeBomb,Trick|Timid|,,,252,4,252||,0,,,,|||,,,,,Poison",
        "Quagsire||HeavyDutyBoots|Unaware|Recover,StealthRock,Toxic,Earthquake|Impish|252,,252,,4,|||||,,,,,Fairy]Blissey||HeavyDutyBoots|NaturalCure|SoftBoiled,CalmMind,Flamethrower,SeismicToss|Calm|4,,252,,252,||,0,,,,|||,,,,,Dark]Toxapex||HeavyDutyBoots|Regenerator|Toxic,ToxicSpikes,Recover,Haze|Careful|248,,,,252,8||,0,,,,|||,,,,,Steel]Sinistcha||HeavyDutyBoots|Heatproof|StrengthSap,MatchaGotcha,FoulPlay,Hex|Bold|160,,252,,,96||,0,,,,|||,,,,,Fairy]Gliscor||ToxicOrb|PoisonHeal|Protect,KnockOff,Spikes,Earthquake|Careful|244,,,,168,96|||||,,,,,Steel]Corviknight||HeavyDutyBoots|Pressure|Roost,IronDefense,BodyPress,Defog|Bold|60,,252,,,196||,0,,,,|||,,,,,Fighting",
        "Raging Bolt||BoosterEnergy|Protosynthesis|Thunderbolt,DragonPulse,Thunderclap,CalmMind|Modest|248,,8,252,,||,20,,,,|||,,,,,Fairy]Iron Treads||BoosterEnergy|QuarkDrive|Earthquake,IceSpinner,StoneEdge,RapidSpin|Jolly|64,252,,,,192|||||,,,,,Ground]Ting-Lu||WeaknessPolicy|VesselofRuin|Earthquake,Payback,StealthRock,Ruination|Adamant|248,124,68,,68,|||||,,,,,Water]Gholdengo||AirBalloon|GoodasGold|ShadowBall,MakeItRain,Recover,NastyPlot|Modest|248,,,72,,188||,0,,,,|||,,,,,Fairy]Zamazenta||Leftovers|DauntlessShield|BodyPress,Crunch,IronDefense,Roar|Jolly|,4,252,,,252|||||,,,,,Fire]Tornadus-Therian||LifeOrb|Regenerator|BleakwindStorm,DarkPulse,FocusBlast,NastyPlot|Timid|,,4,252,,252||,0,,,,|||,,,,,Dark",
        "Ribombee||FocusSash|ShieldDust|Moonblast,Psychic,StickyWeb,StunSpore|Timid|,,,252,4,252||,0,,,,|||,,,,,Ghost]Ogerpon-Wellspring||WellspringMask|WaterAbsorb|SwordsDance,PlayRough,PowerWhip,IvyCudgel|Jolly|,252,4,,,252|F||||,,,,,Water]Iron Moth||BoosterEnergy|QuarkDrive|FieryDance,SludgeWave,Psychic,Substitute|Timid|,,,252,4,252||,0,,,,|||,,,,,Ghost]Kingambit||LumBerry|SupremeOverlord|SwordsDance,IronHead,LowKick,SuckerPunch|Adamant|,252,,,4,252|||||,,,,,Fighting]Gholdengo||AirBalloon|GoodasGold|ShadowBall,Psyshock,NastyPlot,DazzlingGleam|Timid|,,,252,4,252||,0,,,,|||,,,,,Fairy]Iron Valiant||BoosterEnergy|QuarkDrive|CloseCombat,KnockOff,Moonblast,DestinyBond|Naive|,252,,4,,252||,,,,0,|||,,,,,Ghost",
        "Ribombee||FocusSash|ShieldDust|StickyWeb,Moonblast,SkillSwap,PsychicNoise|Timid|,,192,64,,252||,0,,,,|||,,,,,Ghost]Great Tusk||BoosterEnergy|Protosynthesis|HeadlongRush,CloseCombat,RapidSpin,HeadSmash|Jolly|,252,4,,,252|||||,,,,,Ghost]Gholdengo||AirBalloon|GoodasGold|NastyPlot,ShadowBall,MakeItRain,DazzlingGleam|Timid|,,,252,4,252||,0,,,,|||,,,,,Fairy]Manaphy||Leftovers|Hydration|TailGlow,Surf,AlluringVoice,Substitute|Timid|,,16,252,,240||,0,,,,|||,,,,,Fairy]Kingambit||AirBalloon|SupremeOverlord|SwordsDance,KowtowCleave,TeraBlast,SuckerPunch|Jolly|,252,,,4,252|||||,,,,,Fairy]Iron Valiant||BoosterEnergy|QuarkDrive|CalmMind,Moonblast,ShadowBall,VacuumWave|Timid|,,,252,4,252||,0,,,,|||,,,,,Ghost",
        "Rotom-Wash||Leftovers|Levitate|VoltSwitch,HydroPump,WillOWisp,Protect|Bold|252,,252,,4,||,0,,,,|||,,,,,Ghost]Glimmora||ChoiceScarf|ToxicDebris|Spikes,EarthPower,PowerGem,SludgeWave|Timid|,,4,252,,252||,0,,,,|||,,,,,Ground]Great Tusk||Leftovers|Protosynthesis|Earthquake,RapidSpin,KnockOff,StealthRock|Impish|252,,188,,,68|||||,,,,,Water]Kingambit||Leftovers|SupremeOverlord|KowtowCleave,IronHead,SuckerPunch,SwordsDance|Adamant|112,252,,,,144|||||,,,,,Flying]Skeledirge||HeavyDutyBoots|Unaware|Substitute,TorchSong,ShadowBall,SlackOff|Modest|252,,40,88,100,28||,0,,,,|||,,,,,Fairy]Dragapult||HeavyDutyBoots|Infiltrator|WillOWisp,Uturn,Hex,DracoMeteor|Timid|,,,252,4,252|||||,,,,,Dragon",
        "Samurott-Hisui||AssaultVest|Sharpness|CeaselessEdge,RazorShell,SuckerPunch,KnockOff|Adamant|160,76,,,172,100|F||||,,,,,Poison]Tornadus-Therian||LifeOrb|Regenerator|NastyPlot,BleakwindStorm,HeatWave,GrassKnot|Timid|,,16,252,,240||,0,,,,|||,,,,,Steel]Pecharunt||HeavyDutyBoots|PoisonPuppeteer|MalignantChain,FoulPlay,Recover,PartingShot|Bold|252,,212,,12,32||,0,,,,|||,,,,,Dark]Great Tusk||HeavyDutyBoots|Protosynthesis|StealthRock,HeadlongRush,KnockOff,RapidSpin|Jolly|144,112,,,,252|||||,,,,,Ground]Clefable||Leftovers|MagicGuard|CalmMind,Moonblast,Flamethrower,Moonlight|Bold|252,,192,,,64||,0,,,,|||,,,,,Water]Zamazenta||AssaultVest|DauntlessShield|CloseCombat,Crunch,StoneEdge,HeavySlam|Jolly|,252,,,4,252|||||,,,,,Steel",
        "Samurott-Hisui||FocusSash|Sharpness|CeaselessEdge,AquaJet,SacredSword,RazorShell|Jolly|96,156,,,4,252|F||||,,,,,Fairy]Kingambit||Leftovers|SupremeOverlord|SwordsDance,KowtowCleave,TeraBlast,SuckerPunch|Adamant|248,252,,,,8|F||||,,,,,Fairy]Sandy Shocks||BoosterEnergy|Protosynthesis|StealthRock,EarthPower,Thunderbolt,TeraBlast|Timid|48,,,208,,252||,0,,,,|||,,,,,Ice]Gholdengo||AirBalloon|GoodasGold|ThunderWave,MakeItRain,ShadowBall,FocusBlast|Modest|248,,,172,,88||,0,,,,|||,,,,,Water]Iron Valiant||BoosterEnergy|QuarkDrive|SwordsDance,KnockOff,CloseCombat,Encore|Jolly|,252,,,4,252|||||,,,,,Dark]Walking Wake||BoosterEnergy|Protosynthesis|Agility,DragonPulse,Substitute,Surf|Modest|32,,,252,,224||,0,,,,|||,,,,,Water",
        "Samurott-Hisui||HeavyDutyBoots|Sharpness|CeaselessEdge,AquaCutter,KnockOff,SuckerPunch|Adamant|72,252,,,,184|||||,,,,,Poison]Gliscor||ToxicOrb|PoisonHeal|Earthquake,Toxic,Protect,Uturn|Careful|244,,36,,228,||,,,,,25|||,,,,,Ghost]Great Tusk||RockyHelmet|Protosynthesis|HeadlongRush,CloseCombat,IceSpinner,RapidSpin|Jolly|184,72,,,,252|||||,,,,,Fairy]Tinkaton||AirBalloon|Pickpocket|StealthRock,GigatonHammer,KnockOff,Encore|Jolly|248,,,,28,232|||||,,,,,Ghost]Dragapult||HeavyDutyBoots|Infiltrator|DracoMeteor,Hex,WillOWisp,Uturn|Timid|,,56,200,,252|||||,,,,,Dragon]Garganacl||Leftovers|PurifyingSalt|Curse,SaltCure,Earthquake,Recover|Careful|252,,52,,204,|||||,,,,,Water",
        "Skarmory||RockyHelmet|Sturdy|Whirlwind,BraveBird,StealthRock,Roost|Impish|248,,252,,,8|||||,,,,,Dragon]Ting-Lu||HeavyDutyBoots|VesselofRuin|Ruination,Earthquake,Whirlwind,Spikes|Careful|252,,,,252,4|||||,,,,,Ghost]Zamazenta||HeavyDutyBoots|DauntlessShield|CloseCombat,Crunch,HeavySlam,StoneEdge|Adamant|,252,,,4,252|||||,,,,,Fire]Weavile||HeavyDutyBoots|Pickpocket|TripleAxel,IceShard,KnockOff,LowKick|Jolly|,252,,,4,252|||||,,,,,Ice]Kyurem||HeavyDutyBoots|Pressure|ScaleShot,IceBeam,FreezeDry,EarthPower|Timid|,,4,252,,252|||||,,,,,Fairy]Slowking-Galar||HeavyDutyBoots|Regenerator|FutureSight,SludgeBomb,ChillyReception,Toxic|Sassy|252,,4,,252,||,0,,,,0|||,,,,,Water",
        "Slowking||HeavyDutyBoots|Regenerator|LightScreen,Reflect,SlackOff,Surf|Sassy|248,,8,,252,||,0,,,,1|||,,,,,Steel]Toxapex||HeavyDutyBoots|Regenerator|Toxic,SludgeBomb,Recover,Haze|Bold|248,,252,,,8||,0,,,,|||,,,,,Fairy]Alomomola||HeavyDutyBoots|Regenerator|Wish,Protect,Whirlpool,ChillingWater|Bold|252,,252,,4,||,0,,,,|||,,,,,Flying]Dondozo||HeavyDutyBoots|Unaware|Earthquake,Rest,SleepTalk,Avalanche|Impish|252,,252,,4,|||||,,,,,Ghost]Blissey||HeavyDutyBoots|NaturalCure|SoftBoiled,CalmMind,StealthRock,SeismicToss|Calm|4,,252,,252,||,0,,,,|||,,,,,Dark]Great Tusk||HeavyDutyBoots|Protosynthesis|KnockOff,RapidSpin,Rest,BodyPress|Impish|220,,252,,,36|||||,,,,,Ghost",
        "Smooth Criminal|Skarmory|RockyHelmet|Sturdy|Whirlwind,BraveBird,Spikes,Roost|Impish|248,,252,,8,|||||,,,,,Dragon]Rock With You|TingLu|Leftovers|VesselofRuin|StealthRock,Earthquake,Whirlwind,Ruination|Careful|252,,,,248,8|||||,,,,,Water]Beat It|Zamazenta|HeavyDutyBoots|DauntlessShield|CloseCombat,HeavySlam,Crunch,StoneEdge|Jolly|80,252,,,,176|||||,,,,,Steel]Poison Young Thing|Pecharunt|HeavyDutyBoots|PoisonPuppeteer|PartingShot,FoulPlay,MalignantChain,Recover|Bold|252,,196,,,60||,0,,,,|||,,,,,Dark]Moonwalker|Clefable|Leftovers|MagicGuard|CalmMind,Moonblast,Moonlight,KnockOff|Bold|252,,248,,,8|||||,,,,,Steel]Thriller|WalkingWake|HeavyDutyBoots|Protosynthesis|Surf,DracoMeteor,KnockOff,FlipTurn|Timid|,,12,244,,252|||||,,,,,Water",
        "Symbolism|IronMoth|BoosterEnergy|QuarkDrive|FieryDance,Psychic,SludgeWave,Substitute|Timid|,,124,132,,252||,0,,,,|S||,,,,,Ghost]Overgod|Kingambit|LumBerry|SupremeOverlord|SwordsDance,KowtowCleave,SuckerPunch,IronHead|Jolly|,252,4,,,252|M||S||,,,,,Dark]Calamity|IronValiant|BoosterEnergy|QuarkDrive|Substitute,CalmMind,ShadowBall,Moonblast|Timid|,,,252,4,252||,0,,,,|S||,,,,,Ghost]Kong|Rillaboom|ChoiceBand|GrassySurge|WoodHammer,GrassyGlide,Uturn,KnockOff|Adamant|,252,4,,,252|M||S||,,,,,Grass]Spirit Monger|OgerponCornerstone|CornerstoneMask|Sturdy|Spikes,IvyCudgel,PowerWhip,Taunt|Jolly|,252,,,4,252|F||||,Poison,,,,Rock]Byakko|RagingBolt|BoosterEnergy|Protosynthesis|Thunderclap,TeraBlast,DragonPulse,CalmMind|Modest|,,,252,4,252||,20,,,,|||,,,,,Water",
        "Symbolism|IronMoth|BoosterEnergy|QuarkDrive|Substitute,FieryDance,TeraBlast,SludgeWave|Timid|,,124,132,,252||,0,,,,|S||,,,,,Ground]Behemoth|Mamoswine|FocusSash|ThickFat|IceShard,Earthquake,StealthRock,Endeavor|Jolly|,252,4,,,252|M||S||,,,,,Ghost]Braveheart|IronCrown|ChoiceSpecs|QuarkDrive|PsychicNoise,VoltSwitch,TachyonCutter,FocusBlast|Timid|,,,252,4,252||,20,,,,|||,,,,,Steel]Terminator|IronHands|BoosterEnergy|QuarkDrive|SwordsDance,CloseCombat,WildCharge,IcePunch|Adamant|248,252,8,,,|||S||,,,,,Flying]Liberacy|Meowscarada|ChoiceScarf|Protean|Trick,Uturn,TripleAxel,KnockOff|Jolly|,252,4,,,252|M||S||,,,,,Ghost]Celestine|Primarina|Leftovers|LiquidVoice|DrainingKiss,CalmMind,PsychicNoise,Substitute|Bold|240,,252,,,16|M|,0,,,,|S||,,,,,Ghost",
        "Ting-Lu||Leftovers|VesselofRuin|Earthquake,Payback,Rest,SleepTalk|Careful|248,,8,,252,|||||,,,,,Fairy]Slowking-Galar||HeavyDutyBoots|Regenerator|PsychicNoise,SludgeBomb,IceBeam,ChillyReception|Relaxed|248,,252,,8,|M|,0,,,,0|||,,,,,Fairy]Corviknight||RockyHelmet|Pressure|BodyPress,Defog,Roost,Uturn|Relaxed|248,,252,,8,|M|,,,,,0|||,,,,,Dragon]Cinderace||HeavyDutyBoots|Libero|PyroBall,SuckerPunch,Uturn,CourtChange|Jolly|,252,4,,,252|M||||,,,,,Fire]Ogerpon-Wellspring||WellspringMask|WaterAbsorb|IvyCudgel,KnockOff,Uturn,Encore|Jolly|,252,,,4,252|F||||,,,,,Water]Raging Bolt||ChoiceSpecs|Protosynthesis|VoltSwitch,DracoMeteor,Thunderbolt,Thunderclap|Modest|112,,,252,,144||,20,,,,|||,,,,,Fairy",
        "Ting-Lu||Leftovers|VesselofRuin|Earthquake,Ruination,StealthRock,Whirlwind|Careful|248,,40,,216,4|||||,,,,,Ghost]Skarmory||RockyHelmet|Sturdy|BodyPress,Spikes,IronDefense,Roost|Impish|248,,164,,,96||,0,,,,|||,,,,,Fighting]Gholdengo||AirBalloon|GoodasGold|ShadowBall,MakeItRain,NastyPlot,Recover|Bold|248,,176,,12,72||,0,,,,|||,,,,,Fairy]Meowscarada||HeavyDutyBoots|Protean|KnockOff,FlowerTrick,Uturn,TripleAxel|Jolly|,252,,,4,252|||||,,,,,Electric]Zamazenta||HeavyDutyBoots|DauntlessShield|CloseCombat,Crunch,StoneEdge,HeavySlam|Jolly|104,252,,,,152|||||,,,,,Stellar]Toxapex||HeavyDutyBoots|Regenerator|Surf,Toxic,Haze,Recover|Calm|248,,,,240,20||,0,,,,|||,,,,,Fairy",
        "Ting-Lu||Leftovers|VesselofRuin|StealthRock,Earthquake,Ruination,Whirlwind|Impish|252,,24,,228,4|||||,,,,,Water]Dragapult||HeavyDutyBoots|Infiltrator|DracoMeteor,Hex,WillOWisp,Uturn|Timid|,,4,252,,252|||||,,,,,Dragon]Slowking-Galar||HeavyDutyBoots|Regenerator|FutureSight,SludgeBomb,Flamethrower,ChillyReception|Sassy|252,,4,,252,||,0,,,,|||,,,,,Water]Kingambit||Leftovers|SupremeOverlord|SuckerPunch,SwordsDance,IronHead,KowtowCleave|Adamant|232,252,,,,24|||||,,,,,Dark]Gliscor||ToxicOrb|PoisonHeal|SwordsDance,Earthquake,KnockOff,Protect|Jolly|244,,,,244,20|||||,,,,,Fairy]Zamazenta||HeavyDutyBoots|DauntlessShield|CloseCombat,StoneEdge,Crunch,Roar|Jolly|80,252,,,,176|||||,,,,,Fire",
        "Tinkaton||AirBalloon|MoldBreaker|StealthRock,GigatonHammer,ThunderWave,Encore|Jolly|248,28,,,,232|||||,,,,,Water]Great Tusk||HeavyDutyBoots|Protosynthesis|HeadlongRush,CloseCombat,IceSpinner,RapidSpin|Jolly|,224,,,32,252|||||,,,,,Fighting]Garganacl||Leftovers|PurifyingSalt|SaltCure,Curse,Avalanche,Recover|Careful|248,,4,,252,4|||||,,,,,Water]Keldeo||ChoiceSpecs|Justified|Surf,SecretSword,IcyWind,VacuumWave|Timid|,,,252,4,252||,0,,,,|||,,,,,Water]Zapdos||HeavyDutyBoots|Static|VoltSwitch,Hurricane,ThunderWave,Roost|Bold|248,,252,,8,||,0,,,,30|||,,,,,Grass]Dragapult||HeavyDutyBoots|Infiltrator|DragonDarts,Hex,Uturn,WillOWisp|Naive|,80,,176,,252|||||,,,,,Fairy",
        "Torkoal||HeatRock|Drought|Eruption,Overheat,Earthquake,StealthRock|Quiet|104,,,252,152,|||||,,,,,Ground]Hatterene||AirBalloon|MagicBounce|TrickRoom,PsychicNoise,DazzlingGleam,HealingWish|Relaxed|252,,252,,4,||,0,,,,0|||,,,,,Ghost]Raging Bolt||LifeOrb|Protosynthesis|Thunderclap,WeatherBall,DragonPulse,SolarBeam|Modest|,,36,252,,220||,20,,,,|||,,,,,Ghost]Slither Wing||AssaultVest|Protosynthesis|Uturn,FirstImpression,Earthquake,TemperFlare|Adamant|40,252,,,,216|||||,,,,,Fire]Venusaur||LifeOrb|Chlorophyll|Growth,GigaDrain,WeatherBall,Earthquake|Naive|,4,,252,,252|||||,,,,,Fire]Walking Wake||WiseGlasses|Protosynthesis|HydroSteam,WeatherBall,DracoMeteor,FlipTurn|Timid|12,,,244,,252|||||,,,,,Fairy",
        "Toxapex||RedCard|Regenerator|Surf,Toxic,Recover,Haze|Calm|252,,136,,120,||,0,,,,|S||,,,,,Fairy]Kingambit||Leftovers|SupremeOverlord|SwordsDance,IronHead,KowtowCleave,SuckerPunch|Adamant|4,252,,,,252|||||,,,,,Fire]Great Tusk||Leftovers|Protosynthesis|Earthquake,KnockOff,RapidSpin,StealthRock|Impish|248,16,164,,,80|||||,,,,,Water]Corviknight||Leftovers|Pressure|BodyPress,BraveBird,Roost,Uturn|Impish|248,,236,,,24|||||,,,,,Fighting]Hatterene||Leftovers|MagicBounce|DrainingKiss,StoredPower,CalmMind,Nuzzle|Bold|248,,176,,,84|||||,,,,,Water]Dragapult||Leftovers|Infiltrator|Substitute,DragonDarts,Hex,WillOWisp|Naughty|,16,,252,,240|||||,,,,,Fairy",
        "Tyranitar||HeavyDutyBoots|SandStream|KnockOff,StoneEdge,LowKick,IcePunch|Jolly|52,204,,,,252|||||,,,,,Fighting]Gholdengo||AirBalloon|GoodasGold|Hex,MakeItRain,Recover,ThunderWave|Bold|248,,232,,,28||,0,,,,|||,,,,,Fairy]Zamazenta||HeavyDutyBoots|DauntlessShield|CloseCombat,Crunch,HeavySlam,Roar|Jolly|,252,,,4,252|||||,,,,,Ghost]Rillaboom||HeavyDutyBoots|GrassySurge|WoodHammer,GrassyGlide,KnockOff,Uturn|Adamant|4,252,,,,252|||||,,,,,Grass]Garchomp||RockyHelmet|RoughSkin|Earthquake,DragonTail,StealthRock,Spikes|Impish|248,,216,,,44|||||,,,,,Ghost]Zapdos||HeavyDutyBoots|Static|VoltSwitch,Hurricane,Roost,HeatWave|Timid|80,,,180,8,240||,0,,,,|||,,,,,Fire",
        "Viper|WeezingGalar|HeavyDutyBoots|NeutralizingGas|StrangeSteam,Defog,PainSplit,WillOWisp|Bold|248,,236,16,,8||,0,,,,|||,,,,,Grass]Alluka|Chansey|Eviolite|NaturalCure|SoftBoiled,SeismicToss,ShadowBall,CalmMind|Bold|,,252,,248,8||,0,,,,|||,,,,,Dark]Nargacuga|Gliscor|ToxicOrb|PoisonHeal|Protect,StealthRock,KnockOff,Toxic|Jolly|244,,36,,220,8|||||,,,,,Ghost]Gobul|Dondozo|Leftovers|Unaware|WaveCrash,BodyPress,Rest,SleepTalk|Impish|248,,252,,,8|||||,,,,,Fighting]Water Prison|Toxapex|AssaultVest|Regenerator|Surf,SludgeBomb,IceBeam,Infestation|Modest|248,,,252,,8|M|,0,,,,|||,,,,,Steel]Kushala Daora|Corviknight|Leftovers|Pressure|Defog,Roost,IronDefense,BodyPress|Impish|248,,140,,,120||,0,,,,|S||,,,,,Fighting",
        "Volcanion||HeavyDutyBoots|WaterAbsorb|SteamEruption,Flamethrower,TeraBlast,Taunt|Modest|152,,,252,,104||,0,,,,|||,,,,,Fairy]Dragapult||ChoiceBand|Infiltrator|DragonDarts,PhantomForce,DoubleEdge,Uturn|Jolly|,252,,,4,252|||||,,,,,Steel]Garganacl||Leftovers|PurifyingSalt|StealthRock,Recover,SaltCure,Protect|Careful|252,,,,252,4|||||,,,,,Water]Great Tusk||Leftovers|Protosynthesis|Earthquake,IceSpinner,KnockOff,RapidSpin|Impish|252,,176,,,80|||||,,,,,Steel]Zapdos||HeavyDutyBoots|Static|Hurricane,Discharge,Roost,Uturn|Bold|248,,240,,,20|||||,,,,,Steel]Heatran||AirBalloon|FlashFire|Taunt,MagmaStorm,EarthPower,TeraBlast|Modest|184,,,196,,128||,0,,,,|||,,,,,Ghost",
        "Weavile||HeavyDutyBoots|Pressure|SwordsDance,IcicleCrash,KnockOff,IceShard|Jolly|,252,,,4,252|F||||,,,,,Fire]Dragonite||HeavyDutyBoots|Multiscale|ExtremeSpeed,Earthquake,DragonDance,Roost|Adamant|248,104,52,,,104|F||||,,,,,Normal]Gliscor||ToxicOrb|PoisonHeal|SwordsDance,Facade,KnockOff,Protect|Careful|244,,,,168,96|F||||,,,,,Normal]Clefable||HeavyDutyBoots|Unaware|ThunderWave,CalmMind,Moonblast,Moonlight|Bold|248,,248,,,12|F|,0,,,,|||,,,,,Poison]Corviknight||RockyHelmet|Pressure|IronDefense,BodyPress,BraveBird,Roost|Impish|136,,252,,,120|F||||,,,,,Water]Ting-Lu||Leftovers|VesselofRuin|StealthRock,Earthquake,Ruination,Spikes|Careful|244,,12,,252,|||||,,,,,Ghost",
        "Weavile||HeavyDutyBoots|Pressure|TripleAxel,KnockOff,IceShard,SwordsDance|Jolly|,252,,,4,252|F||||,,,,,Ghost]Pecharunt||HeavyDutyBoots|PoisonPuppeteer|MalignantChain,Hex,PartingShot,Recover|Bold|248,,200,,56,4||,0,,,,|||,,,,,Ghost]Ting-Lu||RockyHelmet|VesselofRuin|Earthquake,Spikes,Ruination,Whirlwind|Impish|240,12,16,,220,20|||||,,,,,Water]Tinkaton||AirBalloon|Pickpocket|StealthRock,GigatonHammer,Encore,ThunderWave|Jolly|240,36,,,,232|||||,,,,,Water]Keldeo||HeavyDutyBoots|Justified|HydroPump,SecretSword,FlipTurn,VacuumWave|Timid|,,,252,4,252|||||,,,,,Fighting]Dragonite||HeavyDutyBoots|Multiscale|ExtremeSpeed,Earthquake,Hurricane,Roost|Lonely|224,252,,,,32|F||||,,,,,Normal",
        "Zamazenta||ChestoBerry|DauntlessShield|IronDefense,BodyPress,Crunch,Rest|Jolly|,96,160,,,252|||||,,,,,Fire]Ninetales-Alola||LightClay|SnowWarning|AuroraVeil,FreezeDry,Encore,Roar|Timid|248,,,8,,252||,0,,,,|||,,,,,Poison]Darkrai||Leftovers|BadDreams|NastyPlot,DarkPulse,Psyshock,FocusBlast|Timid|,,,252,4,252||,0,,,,|||,,,,,Poison]Ceruledge||CovertCloak|FlashFire|BulkUp,BitterBlade,ShadowSneak,Taunt|Careful|248,,60,,116,84|||||,,,,,Bug]Gliscor||ToxicOrb|PoisonHeal|SwordsDance,Facade,Earthquake,Agility|Jolly|244,20,36,,112,96|M||||,,,,,Normal]Hatterene||Leftovers|MagicBounce|CalmMind,Psyshock,DrainingKiss,MysticalFire|Bold|248,,196,,,64||,0,,,,|||,,,,,Water",
        "Zamazenta||ExpertBelt|DauntlessShield|CloseCombat,Crunch,StoneEdge,IceFang|Jolly|,252,,,4,252|||||,,,,,Dark]Ogerpon-Wellspring||WellspringMask|WaterAbsorb|IvyCudgel,PowerWhip,KnockOff,Uturn|Jolly|,252,,,4,252|F||||,,,,,Water]Slowking-Galar||AssaultVest|Regenerator|Psyshock,Flamethrower,SludgeBomb,IceBeam|Modest|208,,,216,80,4||,0,,,,|||,,,,,Water]Great Tusk||HeavyDutyBoots|Protosynthesis|HeadlongRush,IceSpinner,RapidSpin,StealthRock|Jolly|,252,,,4,252|||S||,,,,,Fire]Kingambit||Leftovers|SupremeOverlord|KowtowCleave,SuckerPunch,LowKick,SwordsDance|Adamant|200,252,,,,56|M||||,,,,,Ghost]Zapdos||HeavyDutyBoots|Static|Hurricane,VoltSwitch,ThunderWave,Roost|Bold|248,,200,,,60||,0,,,,|||,,,,,Grass",
        "Zamazenta||HeavyDutyBoots|DauntlessShield|CloseCombat,StoneEdge,IronHead,Crunch|Adamant|,252,4,,,252|||||,,,,,Steel]Moltres||HeavyDutyBoots|FlameBody|Flamethrower,WillOWisp,Roost,Uturn|Calm|248,,,,248,12|||||,,,,,Dragon]Ogerpon-Wellspring||WellspringMask|WaterAbsorb|KnockOff,Synthesis,PowerWhip,IvyCudgel|Jolly|,252,4,,,252|F||||,,,,,Water]Great Tusk||RockyHelmet|Protosynthesis|StealthRock,Earthquake,KnockOff,RapidSpin|Impish|252,,172,,,84|||||,,,,,Fire]Clefable||Leftovers|MagicGuard|Moonblast,Encore,Protect,Wish|Bold|252,,240,,,16||,0,,,,|||,,,,,Fairy]Ting-Lu||Leftovers|VesselofRuin|Earthquake,Rest,Whirlwind,Spikes|Careful|252,,,,248,8|||||,,,,,Ghost",
        "Zamazenta||Leftovers|DauntlessShield|BodyPress,Crunch,Roar,IronDefense|Jolly|240,16,56,,,196|||||,,,,,Fire]Kingambit||Leftovers|SupremeOverlord|SwordsDance,SuckerPunch,KowtowCleave,IronHead|Adamant|208,252,,,,48|||||,,,,,Ghost]Samurott-Hisui||FocusSash|Sharpness|CeaselessEdge,RazorShell,KnockOff,SwordsDance|Adamant|,252,,,4,252|||||,,,,,Ghost]Landorus-Therian||RockyHelmet|Intimidate|StealthRock,Earthquake,Uturn,Taunt|Jolly|252,4,,,,252|M||||,,,,,Water]Iron Valiant||BoosterEnergy|QuarkDrive|CalmMind,Moonblast,Psyshock,Thunderbolt|Timid|4,,,252,,252||,0,,,,|||,,,,,Electric]Iron Moth||BoosterEnergy|QuarkDrive|FieryDance,SludgeWave,TeraBlast,Psychic|Timid|,,124,132,,252||,0,,,,|||,,,,,Ground",
        "Zapdos||HeavyDutyBoots|Static|Hurricane,WeatherBall,VoltSwitch,Roost|Timid|248,,76,,,184||,0,,,,|||,,,,,Grass]Hydrapple||HeavyDutyBoots|Regenerator|DracoMeteor,GigaDrain,EarthPower,NastyPlot|Modest|248,,176,16,56,12||,0,,,,|||,,,,,Steel]Zamazenta||HeavyDutyBoots|DauntlessShield|CloseCombat,Crunch,StoneEdge,HeavySlam|Jolly|,252,,,4,252|||||,,,,,Steel]Slowking-Galar||HeavyDutyBoots|Regenerator|SludgeBomb,FutureSight,Flamethrower,ChillyReception|Sassy|248,,8,,252,||,0,,,,0|||,,,,,Water]Tinkaton||AirBalloon|MoldBreaker|GigatonHammer,ThunderWave,Encore,StealthRock|Jolly|248,,,,24,236|||||,,,,,Water]Ting-Lu||Leftovers|VesselofRuin|Earthquake,BodyPress,Ruination,Whirlwind|Impish|248,,244,,,16|||||,,,,,Ghost",
        "Zoroark-Hisui||ChoiceSpecs|Illusion|HyperVoice,ShadowBall,FocusBlast,GrassKnot|Timid|,,,252,4,252||,0,,,,|||,,,,,Fighting]Erebus|Kingambit|Leftovers|SupremeOverlord|KowtowCleave,SuckerPunch,IronHead,SwordsDance|Adamant|80,252,,,8,168|||||,,,,,Fire]Tuskuthy|GreatTusk|Leftovers|Protosynthesis|Earthquake,KnockOff,RapidSpin,StealthRock|Impish|248,16,164,,,80|||||,,,,,Water]Hail Joint|Slowking|HeavyDutyBoots|Regenerator|Surf,FutureSight,SlackOff,ChillyReception|Sassy|248,,8,,252,||,0,,,,|||,,,,,Water]Mav's Joint|IronValiant|ChoiceScarf|QuarkDrive|Moonblast,CloseCombat,KnockOff,Trick|Naive|,184,,72,,252|||||,,,,,Dark]Shenlong|Dragonite|HeavyDutyBoots|Multiscale|ExtremeSpeed,Earthquake,DragonDance,Roost|Adamant|124,252,,,,132|||||,,,,,Normal",
        "rafe|Dragapult|ChoiceBand|Infiltrator|DragonDarts,PhantomForce,TeraBlast,Uturn|Adamant|,252,4,,,252|M||S||,,,,,Fighting]dushane|Munkidori|HeavyDutyBoots|ToxicChain|FutureSight,FocusBlast,Psychic,Uturn|Timid|,,,252,4,252|||||,,,,,Fighting]jamie|GreatTusk|Leftovers|Protosynthesis|Earthquake,IceSpinner,RapidSpin,StealthRock|Impish|244,4,176,,,84|||S||,,,,,Fairy]sully|Gholdengo|CovertCloak|GoodasGold|MakeItRain,ShadowBall,NastyPlot,Recover|Modest|248,,144,24,,92||,0,,,,|S||,,,,,Flying]dris|SamurottHisui|AssaultVest|Sharpness|CeaselessEdge,RazorShell,SacredSword,AquaJet|Careful|252,56,,,100,100|F||S||,,,,,Poison]modie|Zapdos|HeavyDutyBoots|Static|Hurricane,Substitute,VoltSwitch,Roost|Timid|,,,248,8,252||,0,,,,|S||,,,,,Steel",
    ],
};

export function getSampleTeam(format: string): string {
    const internalFormat = format
        .replace("_ou_all_formats", "ou")
        .replace("_ou_only_format", "ou");
    const teams = sampleTeams[internalFormat];
    if (!teams || teams.length === 0) {
        throw new Error(`No sample teams found for format: ${format}`);
    }
    return teams[Math.floor(Math.random() * teams.length)];
}

export function generateTeamFromFormat(format: string): string {
    const species2Sets = sets[format];
    const validator = new TeamValidator(
        format
            .replace("_ou_all_formats", "ou")
            .replace("_ou_only_format", "ou"),
    );

    while (true) {
        const packedSets: Set<string> = new Set();
        const species2Choose = Object.keys(species2Sets).filter(
            (species) => species2Sets[species].length > 0,
        );

        while (packedSets.size < 6) {
            const species =
                species2Choose[
                    Math.floor(Math.random() * species2Choose.length)
                ];
            species2Choose.splice(species2Choose.indexOf(species), 1);
            const speciesSets = species2Sets[species];
            const packedSet =
                speciesSets[Math.floor(Math.random() * speciesSets.length)];

            packedSets.add(packedSet);
        }

        const packedTeam = Array.from(packedSets).join("]");
        const errors = validator.validateTeam(Teams.unpack(packedTeam));

        if (errors === null) {
            return packedTeam;
        }
    }
}

export function generateTeamFromIndices(
    smogonFormat: string,
    speciesIndices?: number[],
    packedSetIndices?: number[],
): string | null {
    if (
        speciesIndices !== undefined &&
        packedSetIndices !== undefined &&
        !smogonFormat.endsWith("randombattle")
    ) {
        const packedSets = [];

        const speciesKeys = Object.values(jsonDatum.species);

        for (const [
            memberIndex,
            packedSetIndex,
        ] of packedSetIndices.entries()) {
            const species = speciesKeys[speciesIndices[memberIndex]];

            const speciesPackedSets = lookUpSetsList(smogonFormat, species);
            if (speciesPackedSets.length === 0) {
                throw new Error(`No sets found for species: ${species}`);
            }

            const speciesPackedSet = speciesPackedSets[packedSetIndex];
            if (!speciesPackedSet) {
                throw new Error(
                    `No packed set found for species: ${species} @ ${packedSetIndex}`,
                );
            }

            packedSets.push(speciesPackedSet);
        }

        return packedSets.join("]");
    } else if (smogonFormat.endsWith("randombattle")) {
        return null;
    } else {
        throw new Error(
            `Invalid format: ${smogonFormat}. Must end with 'randombattle' or provide indices.`,
        );
    }
}

function int16ArrayToBitIndices(arr: Int16Array): number[] {
    const indices: number[] = [];

    for (let i = 0; i < arr.length; i++) {
        let num = arr[i];

        // Process each of the 16 bits in the int16 value
        for (let bitPosition = 0; bitPosition < 16; bitPosition++) {
            // Check if the least significant bit is 1
            if ((num & 1) !== 0) {
                indices.push(i * 16 + bitPosition); // Calculate the bit index
            }

            // Right shift the number to check the next bit
            num >>>= 1;
        }
    }

    return indices;
}

function bigIntToInt16Array(value: bigint): Int16Array {
    // Determine the number of 16-bit chunks needed to store the BigInt
    const bitSize = value.toString(2).length; // Number of bits in the BigInt
    const chunkCount = Math.ceil(bitSize / 16);

    // Create an Int16Array to store the chunks
    const result = new Int16Array(chunkCount);

    // Mask to extract 16 bits
    const mask = BigInt(0xffff);

    for (let i = 0; i < chunkCount; i++) {
        // Extract the lower 16 bits
        result[i] = Number(value & mask);
        // Shift the BigInt to the right by 16 bits
        value >>= BigInt(16);
    }

    return result;
}

const entityPrivateArrayToObject = (array: Int16Array) => {
    const moveIndicies = Array.from(
        array.slice(
            EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID0,
            EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID3 + 1,
        ),
    );

    return {
        species:
            jsonDatum["species"][
                array[
                    EntityPrivateNodeFeature
                        .ENTITY_PRIVATE_NODE_FEATURE__SPECIES
                ]
            ],
        item: jsonDatum["items"][
            array[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ITEM]
        ],
        ability:
            jsonDatum["abilities"][
                array[
                    EntityPrivateNodeFeature
                        .ENTITY_PRIVATE_NODE_FEATURE__ABILITY
                ]
            ],
        moves: moveIndicies.map((index) => jsonDatum["moves"][index]),
        teraType:
            jsonDatum["typechart"][
                array[
                    EntityPrivateNodeFeature
                        .ENTITY_PRIVATE_NODE_FEATURE__TERA_TYPE
                ]
            ],
    };
};

const entityPublicArrayToObject = (array: Int16Array) => {
    const volatilesFlat = array.slice(
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES0,
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES8 + 1,
    );
    const volatilesIndices = int16ArrayToBitIndices(volatilesFlat);

    const typechangeFlat = array.slice(
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE0,
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE1 + 1,
    );
    const typechangeIndices = int16ArrayToBitIndices(typechangeFlat);

    return {
        hp:
            array[
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO
            ] / MAX_RATIO_TOKEN,
        fainted:
            !!array[
                EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED
            ],
        volatiles: volatilesIndices.map(
            (index) => jsonDatum["volatileStatus"][index],
        ),
        typechange: typechangeIndices.map(
            (index) => jsonDatum["typechart"][index],
        ),
        active: array[
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
        ],
        side: array[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE],
        status: jsonDatum["status"][
            array[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__STATUS]
        ],
    };
};

const entityRevealedArrayToObject = (array: Int16Array) => {
    const moveIndicies = Array.from(
        array.slice(
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID0,
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID3 + 1,
        ),
    );

    return {
        species:
            jsonDatum["species"][
                array[
                    EntityRevealedNodeFeature
                        .ENTITY_REVEALED_NODE_FEATURE__SPECIES
                ]
            ],
        item: jsonDatum["items"][
            array[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ITEM]
        ],
        ability:
            jsonDatum["abilities"][
                array[
                    EntityRevealedNodeFeature
                        .ENTITY_REVEALED_NODE_FEATURE__ABILITY
                ]
            ],
        moves: moveIndicies.map((index) => jsonDatum["moves"][index]),
        teraType:
            jsonDatum["typechart"][
                array[
                    EntityRevealedNodeFeature
                        .ENTITY_REVEALED_NODE_FEATURE__TERA_TYPE
                ]
            ],
    };
};

const moveArrayToObject = (array: Int16Array) => {
    return {
        pp_ratio:
            array[MovesetFeature.MOVESET_FEATURE__PP_RATIO] / MAX_RATIO_TOKEN,
        move_id:
            jsonDatum["moves"][array[MovesetFeature.MOVESET_FEATURE__MOVE_ID]],
        pp: array[MovesetFeature.MOVESET_FEATURE__PP],
        maxpp: array[MovesetFeature.MOVESET_FEATURE__MAXPP],
        has_pp: !!array[MovesetFeature.MOVESET_FEATURE__HAS_PP],
        action_type: array[MovesetFeature.MOVESET_FEATURE__ACTION_TYPE],
        entity_idx: array[MovesetFeature.MOVESET_FEATURE__ENTITY_IDX],
    };
};

const entityEdgeArrayToObject = (array: Int16Array) => {
    const minorArgsFlat = array.slice(
        EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG0,
        EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG3 + 1,
    );
    const minorArgIndices = int16ArrayToBitIndices(minorArgsFlat);

    return {
        majorArg:
            jsonDatum["battleMajorArgs"][
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG]
            ],
        minorArgs: minorArgIndices.map(
            (index) => jsonDatum["battleMinorArgs"][index],
        ),
        move: jsonDatum["moves"][
            array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN]
        ],
        damage:
            array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO] /
            MAX_RATIO_TOKEN,
        heal:
            array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO] /
            MAX_RATIO_TOKEN,
        num_from_sources:
            array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_SOURCES],
        from_source: [
            jsonDatum["Effect"][
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN0]
            ],
            jsonDatum["Effect"][
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN1]
            ],
            jsonDatum["Effect"][
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN2]
            ],
            jsonDatum["Effect"][
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN3]
            ],
            jsonDatum["Effect"][
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN4]
            ],
        ],
        boosts: {
            EDGE_BOOST_ATK_VALUE:
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_ATK_VALUE],
            EDGE_BOOST_DEF_VALUE:
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_DEF_VALUE],
            EDGE_BOOST_SPA_VALUE:
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPA_VALUE],
            EDGE_BOOST_SPD_VALUE:
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPD_VALUE],
            EDGE_BOOST_SPE_VALUE:
                array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_SPE_VALUE],
            EDGE_BOOST_ACCURACY_VALUE:
                array[
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_ACCURACY_VALUE
                ],
            EDGE_BOOST_EVASION_VALUE:
                array[
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__BOOST_EVASION_VALUE
                ],
        },
        status: jsonDatum["status"][
            array[EntityEdgeFeature.ENTITY_EDGE_FEATURE__STATUS_TOKEN]
        ],
    };
};

const fieldArrayToObject = (array: Int16Array) => {
    const mySideConditionsFlat = array.slice(
        FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS0,
        FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS1 + 1,
    );
    const mySideConditionsIndices =
        int16ArrayToBitIndices(mySideConditionsFlat);
    const oppSideConditionsFlat = array.slice(
        FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS0,
        FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS1 + 1,
    );
    const oppSideConditionsIndices = int16ArrayToBitIndices(
        oppSideConditionsFlat,
    );

    return {
        mySideConditions: mySideConditionsIndices.map((index) => {
            return jsonDatum["sideCondition"][index];
        }),
        myNumSpikes: array[FieldFeature.FIELD_FEATURE__MY_SPIKES],
        oppSideConditions: oppSideConditionsIndices.map((index) => {
            return jsonDatum["sideCondition"][index];
        }),
        oppNumSpikes: array[FieldFeature.FIELD_FEATURE__OPP_SPIKES],
        turnOrder: array[FieldFeature.FIELD_FEATURE__TURN_ORDER_VALUE],
        requestCount: array[FieldFeature.FIELD_FEATURE__REQUEST_COUNT],
        weatherId:
            jsonDatum["weather"][array[FieldFeature.FIELD_FEATURE__WEATHER_ID]],
        weatherMinDuration:
            array[FieldFeature.FIELD_FEATURE__WEATHER_MIN_DURATION],
        weatherMaxDuration:
            array[FieldFeature.FIELD_FEATURE__WEATHER_MAX_DURATION],
    };
};

const WEATHERS = {
    sand: "sandstorm",
    sun: "sunnyday",
    rain: "raindance",
    hail: "hail",
    snow: "snowscape",
    harshsunshine: "desolateland",
    heavyrain: "primordialsea",
    strongwinds: "deltastream",
};

function isMySide(n: number, playerIndex: number) {
    return +(n === playerIndex);
}

const enumDatumPrefixCache = new WeakMap<object, string>();
const sanitizeKeyCache = new Map<string, string>();

function getPrefix<T extends EnumMappings>(enumDatum: T): string | null {
    if (enumDatumPrefixCache.has(enumDatum)) {
        return enumDatumPrefixCache.get(enumDatum)!;
    }

    for (const key in enumDatum) {
        const prefix = key.split("__")[0];
        enumDatumPrefixCache.set(enumDatum, prefix);
        return prefix;
    }

    return null; // Handle cases where enumDatum has no keys
}

function SanitizeKey<T extends EnumMappings>(
    enumDatum: T,
    key: string,
): string {
    const prefix = getPrefix(enumDatum);
    if (!prefix) {
        throw new Error(
            "Prefix could not be determined for the given enumDatum",
        );
    }

    // Construct the raw key
    const rawKey = `${prefix}__${key}`;

    // Check if the sanitized key is cached
    if (sanitizeKeyCache.has(rawKey)) {
        return sanitizeKeyCache.get(rawKey)!;
    }

    // Sanitize the key (remove non-alphanumeric characters and make uppercase)
    const sanitizedKey = rawKey.replace(/\W/g, "").toUpperCase();

    // Cache the sanitized key
    sanitizeKeyCache.set(rawKey, sanitizedKey);
    return sanitizedKey;
}

export function IndexValueFromEnum<T extends EnumMappings>(
    enumDatum: T,
    key: string,
): T[keyof T] {
    const sanitizedKey = SanitizeKey(enumDatum, key) as keyof T;

    // Retrieve the value from the enumDatum using the sanitized key
    const value = enumDatum[sanitizedKey];
    if (value === undefined) {
        throw new Error(`${sanitizedKey.toString()} not in mapping`);
    }
    return value;
}

export function concatenateArrays<T extends TypedArray>(arrays: T[]): T {
    // Step 1: Calculate the total length
    let totalLength = 0;
    for (const arr of arrays) {
        totalLength += arr.length;
    }

    // Step 2: Create a new array using the constructor of the first array in the list
    const result = new (arrays[0].constructor as { new (length: number): T })(
        totalLength,
    );

    // Step 3: Copy each array into the result
    let offset = 0;
    for (const arr of arrays) {
        result.set(arr, offset);
        offset += arr.length;
    }

    return result;
}

const POKEMON_ARRAY_CONSTRUCTOR = Int16Array;

function getBlankPublicPokemonArr() {
    return new POKEMON_ARRAY_CONSTRUCTOR(numPublicEntityNodeFeatures);
}

function getBlankRevealedPokemonArr() {
    return new POKEMON_ARRAY_CONSTRUCTOR(numRevealedEntityNodeFeatures);
}

function getBlankPrivatePokemonArr() {
    return new POKEMON_ARRAY_CONSTRUCTOR(numPrivateEntityNodeFeatures);
}

function getUnkRevealedPokemon() {
    const data = getBlankRevealedPokemonArr();
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES] =
        SpeciesEnum.SPECIES_ENUM___UNK;

    // Item
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ITEM] =
        ItemsEnum.ITEMS_ENUM___UNK;

    // Ability
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY] =
        AbilitiesEnum.ABILITIES_ENUM___UNK;

    // Moves
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID0] =
        MovesEnum.MOVES_ENUM___UNK;
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID1] =
        MovesEnum.MOVES_ENUM___UNK;
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID2] =
        MovesEnum.MOVES_ENUM___UNK;
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__MOVEID3] =
        MovesEnum.MOVES_ENUM___UNK;

    // Teratype
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__TERA_TYPE] =
        TypechartEnum.TYPECHART_ENUM___UNK;
    return data;
}

function getUnkPublicPokemon() {
    const data = getBlankPublicPokemonArr();

    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ITEM_EFFECT] =
        ItemeffecttypesEnum.ITEMEFFECTTYPES_ENUM___NULL;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__GENDER] =
        GendernameEnum.GENDERNAME_ENUM___UNK;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP] = 100;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MAXHP] = 100;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO] =
        MAX_RATIO_TOKEN; // Full Health;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__STATUS] =
        StatusEnum.STATUS_ENUM___NULL;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TOXIC_TURNS] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SLEEP_TURNS] = 0;
    data[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BEING_CALLED_BACK
    ] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TRAPPED] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__NEWLY_SWITCHED] =
        0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LEVEL] = 100;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HAS_STATUS] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_ATK_VALUE] =
        0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_DEF_VALUE] =
        0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPA_VALUE] =
        0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPD_VALUE] =
        0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPE_VALUE] =
        0;
    data[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_ACCURACY_VALUE
    ] = 0;
    data[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_EVASION_VALUE
    ] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES0] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES1] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES2] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES3] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES4] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES5] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES6] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES7] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES8] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE0] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE1] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__NUM_MOVES] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LAST_MOVE] =
        MovesEnum.MOVES_ENUM___UNK;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MEGA] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__PRIMAL] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP0] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP1] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP2] = 0;
    data[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MOVEPP3] = 0;

    return data;
}

function getUnkPokemon(n: number) {
    const publicData = getUnkPublicPokemon();
    const revealedData = getUnkRevealedPokemon();

    // Side
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE] = n;
    return { publicData, revealedData };
}

const unkPokemon0 = getUnkPokemon(0);
const unkPokemon1 = getUnkPokemon(1);

function getNullPokemon() {
    const data = getBlankPublicPokemonArr();
    data[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES] =
        SpeciesEnum.SPECIES_ENUM___NULL;
    return data;
}

const nullPokemon = getNullPokemon();

function tryFindIndex(enumDatum: EnumMappings, keys: string[]) {
    for (const key of keys) {
        try {
            return IndexValueFromEnum(enumDatum, key);
        } catch (err) {
            console.log(err);
            continue;
        }
    }
    throw new Error(`None of the keys ${keys} found in enum mapping`);
}

function getArrayFromPrivatePokemon(
    candidate: Pokemon | null | undefined,
    pokemonSet: PokemonSet,
) {
    const dataArr = getBlankPrivatePokemonArr();

    if (candidate === null || candidate === undefined) {
        return dataArr;
    }

    let pokemon: Pokemon;
    if (
        candidate.volatiles.transform !== undefined &&
        candidate.volatiles.transform.pokemon !== undefined
    ) {
        pokemon = candidate.volatiles.transform.pokemon as Pokemon;
    } else {
        pokemon = candidate;
    }

    if (pokemonSet === null || pokemonSet === undefined) {
        throw new Error("Private data requested for null or undefined set");
    }

    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__SPECIES] =
        tryFindIndex(SpeciesEnum, [
            pokemon.baseSpecies.id,
            pokemon.baseSpecies.baseSpecies.toLowerCase(),
        ]);

    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ITEM] =
        !!pokemonSet.item
            ? IndexValueFromEnum(ItemsEnum, pokemonSet.item)
            : ItemsEnum.ITEMS_ENUM___NULL;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ABILITY] =
        IndexValueFromEnum(AbilitiesEnum, pokemonSet.ability);

    const moveset = pokemonSet.moves;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID0] =
        moveset[0]
            ? IndexValueFromEnum(MovesEnum, moveset[0])
            : MovesEnum.MOVES_ENUM___NULL;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID1] =
        moveset[1]
            ? IndexValueFromEnum(MovesEnum, moveset[1])
            : MovesEnum.MOVES_ENUM___NULL;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID2] =
        moveset[2]
            ? IndexValueFromEnum(MovesEnum, moveset[2])
            : MovesEnum.MOVES_ENUM___NULL;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__MOVEID3] =
        moveset[3]
            ? IndexValueFromEnum(MovesEnum, moveset[3])
            : MovesEnum.MOVES_ENUM___NULL;

    const evs = pokemonSet.evs ?? {};
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__EV_HP] =
        evs.hp ?? 0;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__EV_ATK] =
        evs.atk ?? 0;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__EV_DEF] =
        evs.def ?? 0;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__EV_SPA] =
        evs.spa ?? 0;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__EV_SPD] =
        evs.spd ?? 0;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__EV_SPE] =
        evs.spe ?? 0;

    const ivs = pokemonSet.ivs ?? {};
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__IV_HP] =
        ivs.hp ?? 0;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__IV_ATK] =
        ivs.atk ?? 0;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__IV_DEF] =
        ivs.def ?? 0;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__IV_SPA] =
        ivs.spa ?? 0;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__IV_SPD] =
        ivs.spd ?? 0;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__IV_SPE] =
        ivs.spe ?? 0;

    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__NATURE] =
        pokemonSet.nature
            ? IndexValueFromEnum(NaturesEnum, pokemonSet.nature)
            : NaturesEnum.NATURES_ENUM__SERIOUS;
    dataArr[EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__TERA_TYPE] =
        pokemonSet.teraType
            ? IndexValueFromEnum(TypechartEnum, pokemonSet.teraType)
            : TypechartEnum.TYPECHART_ENUM___NULL;
    return dataArr;
}

function getArrayFromPublicPokemon(
    candidate: Pokemon | null,
    relativeSide: number,
): {
    publicData: Int16Array;
    revealedData: Int16Array;
} {
    const { publicData, revealedData } =
        relativeSide === 0 ? getUnkPokemon(0) : getUnkPokemon(1);

    if (candidate === null || candidate === undefined) {
        return { publicData, revealedData: nullPokemon };
    }

    let pokemon: Pokemon;
    let isTransformed = false;
    if (
        candidate.volatiles.transform !== undefined &&
        candidate.volatiles.transform.pokemon !== undefined
    ) {
        pokemon = candidate.volatiles.transform.pokemon as Pokemon;
        isTransformed = true;
    } else {
        pokemon = candidate;
    }

    // Terastallized
    revealedData[
        EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__TERA_TYPE
    ] = pokemon.terastallized
        ? IndexValueFromEnum(TypechartEnum, pokemon.terastallized)
        : TypechartEnum.TYPECHART_ENUM___UNK;

    const ability = pokemon.ability;

    // We take candidate item here instead of pokemons since
    // transformed does not copy item
    const item = candidate.item ?? candidate.lastItem;
    const itemEffect = candidate.itemEffect ?? candidate.lastItemEffect;

    // Moves are stored on candidate
    const moveSlots = candidate.moveSlots.slice(0, 4);
    const moveIds = [];
    const movePps = [];

    if (moveSlots) {
        for (const move of moveSlots) {
            let { id } = move;
            if (id.startsWith("return")) {
                id = "return" as ID;
            } else if (id.startsWith("frustration")) {
                id = "frustration" as ID;
            } else if (id.startsWith("hiddenpower")) {
                const power = parseInt(id.slice(-2));
                if (isNaN(power)) {
                    id = "hiddenpower" as ID;
                } else {
                    id = id.slice(0, -2) as ID;
                }
            }
            const ppUsed = move.ppUsed;
            const maxPP = isTransformed
                ? 5
                : pokemon.side.battle.gens.dex.moves.get(id).pp;

            // Remove pp up assumption (5/8)
            const correctUsed =
                (isNaN(ppUsed) ? +!!ppUsed : ppUsed) *
                (isTransformed ? 1 : 5 / 8);

            moveIds.push(IndexValueFromEnum(MovesEnum, id));
            movePps.push(Math.floor((31 * correctUsed) / maxPP));
        }
    }
    let remainingIndex: MoveIndex = moveSlots.length as MoveIndex;

    for (remainingIndex; remainingIndex < 4; remainingIndex++) {
        moveIds.push(MovesEnum.MOVES_ENUM___UNK);
        movePps.push(0);
    }

    revealedData[
        EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES
    ] = tryFindIndex(SpeciesEnum, [
        pokemon.baseSpecies.id,
        pokemon.baseSpecies.baseSpecies.toLowerCase(),
    ]);
    revealedData[EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ITEM] =
        item
            ? IndexValueFromEnum<typeof ItemsEnum>(ItemsEnum, item)
            : ItemsEnum.ITEMS_ENUM___UNK;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ITEM_EFFECT
    ] = itemEffect
        ? IndexValueFromEnum(ItemeffecttypesEnum, itemEffect)
        : ItemeffecttypesEnum.ITEMEFFECTTYPES_ENUM___NULL;

    const possibleAbilities = Object.values(pokemon.baseSpecies.abilities);
    if (ability) {
        if (ability === "noability" || ability === "none") {
            revealedData[
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY
            ] = AbilitiesEnum.ABILITIES_ENUM__NOABILITY;
        } else {
            const actualAbility = IndexValueFromEnum(AbilitiesEnum, ability);
            revealedData[
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY
            ] = actualAbility;
        }
    } else if (possibleAbilities.length === 1) {
        const onlyAbility = possibleAbilities[0]
            ? IndexValueFromEnum(AbilitiesEnum, possibleAbilities[0])
            : AbilitiesEnum.ABILITIES_ENUM___UNK;
        revealedData[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY
        ] = onlyAbility;
    } else {
        revealedData[
            EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__ABILITY
        ] = AbilitiesEnum.ABILITIES_ENUM___UNK;
    }

    // We take candidate lastMove here instead of pokemons since
    // transformed does not lastMove
    if (candidate.lastMove === "") {
        publicData[
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LAST_MOVE
        ] = MovesEnum.MOVES_ENUM___NULL;
    } else if (candidate.lastMove === "switch-in") {
        publicData[
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LAST_MOVE
        ] = MovesEnum.MOVES_ENUM___SWITCH_IN;
    } else {
        publicData[
            EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LAST_MOVE
        ] = IndexValueFromEnum(MovesEnum, candidate.lastMove);
    }

    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__GENDER] =
        IndexValueFromEnum(GendernameEnum, pokemon.gender);
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE] =
        +candidate.isActive();
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED] =
        +candidate.fainted;

    // We take candidate HP here instead of pokemons since
    // transformed does not copy HP
    const isHpBug = !candidate.fainted && candidate.hp === 0;
    const hp = isHpBug ? 100 : candidate.hp;
    const maxHp = isHpBug ? 100 : candidate.maxhp;
    const hpRatio = hp / maxHp;
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP] = hp;
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__MAXHP] =
        maxHp;
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HP_RATIO] =
        Math.floor(MAX_RATIO_TOKEN * hpRatio);

    // We take candidate status here instead of pokemons since
    // transformed does not copy status
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__STATUS] =
        candidate.status
            ? IndexValueFromEnum(StatusEnum, candidate.status)
            : StatusEnum.STATUS_ENUM___NULL;
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__HAS_STATUS] =
        candidate.status ? 1 : 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TOXIC_TURNS
    ] = candidate.statusState.toxicTurns;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SLEEP_TURNS
    ] = candidate.statusState.sleepTurns;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BEING_CALLED_BACK
    ] = candidate.beingCalledBack ? 1 : 0;
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TRAPPED] =
        candidate.trapped ? 1 : 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__NEWLY_SWITCHED
    ] = candidate.newlySwitched ? 1 : 0;

    // We take pokemon level here
    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__LEVEL] =
        pokemon.level;

    for (let moveIndex = 0; moveIndex < 4; moveIndex++) {
        revealedData[
            EntityRevealedNodeFeature[
                `ENTITY_REVEALED_NODE_FEATURE__MOVEID${moveIndex as MoveIndex}`
            ]
        ] = moveIds[moveIndex];
        publicData[
            EntityPublicNodeFeature[
                `ENTITY_PUBLIC_NODE_FEATURE__MOVEPP${moveIndex as MoveIndex}`
            ]
        ] = movePps[moveIndex];
    }

    // Only copy candidate volatiles
    let volatiles = BigInt(0b0);
    for (const [key] of Object.entries(candidate.volatiles)) {
        const index = getVolatileStatusToken(
            key.startsWith("fallen") ? "fallen" : key,
        );
        volatiles |= BigInt(1) << BigInt(index);
    }
    publicData.set(
        bigIntToInt16Array(volatiles),
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__VOLATILES0,
    );

    publicData[EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__SIDE] =
        relativeSide;

    // Only copy pokemon boosts
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_ATK_VALUE
    ] = pokemon.boosts.atk ?? 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_DEF_VALUE
    ] = pokemon.boosts.def ?? 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPA_VALUE
    ] = pokemon.boosts.spa ?? 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPD_VALUE
    ] = pokemon.boosts.spd ?? 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_SPE_VALUE
    ] = pokemon.boosts.spe ?? 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_EVASION_VALUE
    ] = pokemon.boosts.evasion ?? 0;
    publicData[
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__BOOST_ACCURACY_VALUE
    ] = pokemon.boosts.accuracy ?? 0;

    // Copy candidate type change
    let typeChanged = BigInt(0b0);
    const typechangeVolatile = candidate.volatiles.typechange;
    if (typechangeVolatile) {
        if (typechangeVolatile.apparentType) {
            for (const type of typechangeVolatile.apparentType.split("/")) {
                const index =
                    type === "???"
                        ? TypechartEnum.TYPECHART_ENUM__TYPELESS
                        : IndexValueFromEnum(TypechartEnum, type);
                typeChanged |= BigInt(1) << BigInt(index);
            }
        }
    }
    publicData.set(
        bigIntToInt16Array(typeChanged),
        EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__TYPECHANGE0,
    );

    return { publicData, revealedData };
}

class Edge {
    player: TrainablePlayerAI;

    entityPublicData: Int16Array;
    entityRevealedData: Int16Array;
    entityEdgeData: Int16Array;
    fieldData: Int16Array;

    unkEntityIndex: number;

    constructor(player: TrainablePlayerAI) {
        this.player = player;

        this.entityPublicData = new Int16Array(
            12 * numPublicEntityNodeFeatures,
        );
        this.entityRevealedData = new Int16Array(
            12 * numRevealedEntityNodeFeatures,
        );
        this.entityEdgeData = new Int16Array(12 * numEntityEdgeFeatures);
        this.fieldData = new Int16Array(numFieldFeatures);

        this.unkEntityIndex = player.publicBattle.sides
            .map((x) =>
                x.team
                    .slice(0, 6)
                    .reduce(
                        (a, b) =>
                            +player.eventHandler.identToIndex.has(
                                b.originalIdent,
                            ) + a,
                        0,
                    ),
            )
            .reduce((a, b) => a + b);

        this.updateSideData();
        this.updateFieldData();
    }

    clone() {
        const edge = new Edge(this.player);
        edge.entityPublicData.set(this.entityPublicData);
        edge.entityRevealedData.set(this.entityRevealedData);
        edge.entityEdgeData.set(this.entityEdgeData);
        edge.fieldData.set(this.fieldData);
        return edge;
    }

    updateSideData() {
        const playerIndex = this.player.getPlayerIndex()!;

        let publicOffset = 0;
        let revealedOffset = 0;

        for (const side of this.player.publicBattle.sides) {
            const relativeSide = isMySide(side.n, this.player.getPlayerIndex());

            this.updateEntityData(side, relativeSide);
            this.updateSideConditionData(side, relativeSide);

            const teamLength = side.team.slice(0, 6).length;
            publicOffset += teamLength * numPublicEntityNodeFeatures;
            revealedOffset += teamLength * numRevealedEntityNodeFeatures;
        }
        for (const side of this.player.publicBattle.sides) {
            const relativeSide = isMySide(side.n, this.player.getPlayerIndex());

            const team = side.team.slice(0, 6);
            const { revealedData, publicData } = relativeSide
                ? unkPokemon1
                : unkPokemon0;
            for (let i = team.length; i < side.totalPokemon; i++) {
                this.entityPublicData.set(publicData, publicOffset);
                this.entityRevealedData.set(revealedData, revealedOffset);
                publicOffset += numPublicEntityNodeFeatures;
                revealedOffset += numRevealedEntityNodeFeatures;
            }
            for (let i = side.totalPokemon; i < 6; i++) {
                this.entityRevealedData.set(nullPokemon, revealedOffset);
                revealedOffset += numRevealedEntityNodeFeatures;
            }
        }
    }

    updateEntityData(side: Side, relativeSide: number) {
        const team = side.team.slice(0, side.totalPokemon);

        for (const pokemon of team) {
            const { revealedData, publicData } = getArrayFromPublicPokemon(
                pokemon,
                relativeSide,
            );
            const index =
                this.player.eventHandler.identToIndex.get(
                    pokemon.originalIdent,
                ) ?? this.unkEntityIndex;
            this.entityRevealedData.set(
                revealedData,
                index * numRevealedEntityNodeFeatures,
            );
            this.entityPublicData.set(
                publicData,
                index * numPublicEntityNodeFeatures,
            );
            if (index === undefined) {
                this.unkEntityIndex += 1;
            }
        }
    }

    updateSideConditionData(side: Side, relativeSide: number) {
        let sideConditionBuffer = BigInt(0b0);
        for (const [id] of Object.entries(side.sideConditions)) {
            const featureIndex = IndexValueFromEnum(SideconditionEnum, id);
            sideConditionBuffer |= BigInt(1) << BigInt(featureIndex);
        }
        this.fieldData.set(
            bigIntToInt16Array(sideConditionBuffer),
            relativeSide
                ? FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS0
                : FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS0,
        );
        if (side.sideConditions.spikes) {
            this.setFieldFeature({
                featureIndex: relativeSide
                    ? FieldFeature.FIELD_FEATURE__MY_SPIKES
                    : FieldFeature.FIELD_FEATURE__OPP_SPIKES,
                value: side.sideConditions.spikes.level,
            });
        }
        if (side.sideConditions.toxicspikes) {
            this.setFieldFeature({
                featureIndex: relativeSide
                    ? FieldFeature.FIELD_FEATURE__MY_TOXIC_SPIKES
                    : FieldFeature.FIELD_FEATURE__OPP_TOXIC_SPIKES,
                value: side.sideConditions.toxicspikes.level,
            });
        }
    }

    updateFieldData() {
        const field = this.player.publicBattle.field;
        const weatherIndex = field.weatherState.id
            ? IndexValueFromEnum(
                  WeatherEnum,
                  WEATHERS[field.weatherState.id as never],
              )
            : WeatherEnum.WEATHER_ENUM___NULL;

        this.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__WEATHER_ID,
            value: weatherIndex,
        });
        this.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__WEATHER_MAX_DURATION,
            value: field.weatherState.maxDuration,
        });
        this.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__WEATHER_MIN_DURATION,
            value: field.weatherState.minDuration,
        });
    }

    setEntityEdgeFeature(args: {
        edgeIndex: number;
        featureIndex: EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap];
        value: number;
    }) {
        const { edgeIndex, featureIndex, value } = args;
        if (edgeIndex > 11 || edgeIndex < 0) {
            throw new Error("edgeIndex out of bounds");
        }
        if (featureIndex === undefined) {
            throw new Error("featureIndex cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        this.entityEdgeData[edgeIndex * numEntityEdgeFeatures + featureIndex] =
            value;
    }

    setFieldFeature(args: {
        featureIndex: FieldFeatureMap[keyof FieldFeatureMap];
        value: number;
    }) {
        const { featureIndex, value } = args;
        if (featureIndex === undefined) {
            throw new Error("Index cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        this.fieldData[featureIndex] = value;
    }

    getEntityEdgeFeature(args: {
        edgeIndex: number;
        featureIndex: EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap];
    }) {
        const { edgeIndex, featureIndex } = args;
        const index = edgeIndex * numEntityEdgeFeatures + featureIndex;
        return this.entityEdgeData.at(index);
    }

    getFieldFeature(args: {
        featureIndex: FieldFeatureMap[keyof FieldFeatureMap];
    }) {
        const { featureIndex } = args;
        return this.fieldData[featureIndex];
    }

    addMajorArg(args: { argName: MajorArgNames; edgeIndex: number }) {
        const { argName, edgeIndex } = args;
        const index = IndexValueFromEnum(BattlemajorargsEnum, argName);
        this.setEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG,
            edgeIndex,
            value: index,
        });
    }

    updateEdgeFromOf(args: { effect: Partial<Effect>; edgeIndex: number }) {
        const { effect, edgeIndex } = args;
        const { effectType } = effect;
        if (effectType) {
            const fromTypeToken = IndexValueFromEnum(
                EffecttypesEnum,
                effectType,
            );
            const fromSourceToken = getEffectToken(effect);

            const numFromTypes =
                this.getEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_TYPES,
                    edgeIndex,
                }) ?? 0;
            const numFromSources =
                this.getEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_SOURCES,
                    edgeIndex,
                }) ?? 0;

            if (numFromTypes < 5) {
                this.setEntityEdgeFeature({
                    featureIndex:
                        (EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_TYPE_TOKEN0 +
                            numFromTypes) as EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap],
                    edgeIndex,
                    value: fromTypeToken,
                });
                this.setEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_TYPES,
                    edgeIndex,
                    value: numFromTypes + 1,
                });
            }
            if (numFromSources < 5) {
                this.setEntityEdgeFeature({
                    featureIndex:
                        (EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN0 +
                            numFromSources) as EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap],
                    edgeIndex,
                    value: fromSourceToken,
                });
                this.setEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_SOURCES,
                    edgeIndex,
                    value: numFromSources + 1,
                });
            }
        }
    }
}

function getEffectToken(effect: Partial<Effect>): number {
    const { id } = effect;
    if (id) {
        let value = undefined;
        const testIds: string[] = [id];
        let haveAdded = false;
        while (testIds.length > 0) {
            const testId = testIds.shift()!;
            try {
                value = IndexValueFromEnum(EffectEnum, testId);
                return value;
            } catch (err) {
                if (testIds.length === 0) {
                    if (!haveAdded) {
                        testIds.push(
                            ...[
                                id.slice("move".length),
                                id.slice("item".length),
                                id.slice("pokemon".length),
                                id.slice("ability".length),
                                id.slice("condition".length),
                            ],
                        );
                        haveAdded = true;
                    } else {
                        return EffectEnum.EFFECT_ENUM___UNK;
                    }
                }
            }
        }
        return EffectEnum.EFFECT_ENUM___UNK;
    }
    return EffectEnum.EFFECT_ENUM___NULL;
}

function getVolatileStatusToken(id: string): number {
    let value = undefined;
    const testIds: string[] = [id];
    let haveAdded = false;
    while (testIds.length > 0) {
        const testId = testIds.shift()!;
        try {
            value = IndexValueFromEnum(VolatilestatusEnum, testId);
            return value;
        } catch (err) {
            if (testIds.length === 0) {
                if (!haveAdded) {
                    testIds.push(
                        ...[
                            id.slice("move".length),
                            id.slice("item".length),
                            id.slice("ability".length),
                            id.slice("condition".length),
                        ],
                    );
                    haveAdded = true;
                } else {
                    console.log(id, err);
                    return VolatilestatusEnum.VOLATILESTATUS_ENUM___UNK;
                }
            }
        }
    }
    throw new Error("Volatile status token not found");
}

export class EdgeBuffer {
    player: TrainablePlayerAI;

    entityPublicData: Int16Array;
    entityRevealedData: Int16Array;
    entityEdgeData: Int16Array;
    fieldData: Int16Array;

    entityPublicCursor: number;
    entityRevealedCursor: number;
    entityEdgeCursor: number;
    fieldCursor: number;

    prevEntityPublicCursor: number;
    prevEntityRevealedCursor: number;
    prevEntityEdgeCursor: number;
    prevFieldCursor: number;

    numEdges: number;
    maxEdges: number;

    constructor(player: TrainablePlayerAI) {
        this.player = player;

        const maxEdges = 4000;
        this.maxEdges = maxEdges;

        this.entityPublicData = new Int16Array(
            maxEdges * 12 * numPublicEntityNodeFeatures,
        );
        this.entityRevealedData = new Int16Array(
            maxEdges * 12 * numRevealedEntityNodeFeatures,
        );
        this.entityEdgeData = new Int16Array(
            maxEdges * 12 * numEntityEdgeFeatures,
        );
        this.fieldData = new Int16Array(maxEdges * numFieldFeatures);

        this.entityPublicCursor = 0;
        this.entityRevealedCursor = 0;
        this.entityEdgeCursor = 0;
        this.fieldCursor = 0;

        this.prevEntityPublicCursor = 0;
        this.prevEntityRevealedCursor = 0;
        this.prevEntityEdgeCursor = 0;
        this.prevFieldCursor = 0;

        this.numEdges = 0;
    }

    setLatestEntityEdgeFeature(args: {
        edgeIndex: number;
        featureIndex: EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap];
        value: number;
    }) {
        const { featureIndex, edgeIndex, value } = args;
        if (featureIndex === undefined) {
            throw new Error("Index cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        const index =
            this.prevEntityEdgeCursor +
            edgeIndex * numEntityEdgeFeatures +
            featureIndex;
        this.entityEdgeData[index] = value;
    }

    getLatestEntityEdgeFeature(args: {
        edgeIndex: number;
        featureIndex: EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap];
    }) {
        const { featureIndex, edgeIndex } = args;
        const index =
            this.prevEntityEdgeCursor +
            edgeIndex * numEntityEdgeFeatures +
            featureIndex;
        const value = this.entityEdgeData.at(index);
        if (value === undefined) {
            throw new Error(
                `Feature index ${featureIndex} not found for edge index ${edgeIndex}`,
            );
        }
        return value;
    }

    updateLatestMinorArgs(args: {
        argName: MinorArgNames;
        edgeIndex: number;
        precision?: number;
    }) {
        const { argName, edgeIndex } = args;
        const precision = args.precision ?? 16;

        const index = IndexValueFromEnum(BattleminorargsEnum, argName);
        const featureIndex = {
            0: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG0,
            1: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG1,
            2: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG2,
            3: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MINOR_ARG3,
        }[Math.floor(index / precision)];
        if (featureIndex === undefined) {
            throw new Error();
        }
        const currentValue = this.getLatestEntityEdgeFeature({
            featureIndex,
            edgeIndex,
        })!;
        const newValue = currentValue | (1 << index % precision);
        this.setLatestEntityEdgeFeature({
            featureIndex,
            edgeIndex,
            value: newValue,
        });
    }

    updateLatestMajorArg(args: { argName: MajorArgNames; edgeIndex: number }) {
        const { argName, edgeIndex } = args;
        const index = IndexValueFromEnum(BattlemajorargsEnum, argName);
        this.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MAJOR_ARG,
            edgeIndex,
            value: index,
        });
    }

    setLatestFieldFeature(args: {
        featureIndex: FieldFeatureMap[keyof FieldFeatureMap];
        value: number;
    }) {
        const { featureIndex, value } = args;
        if (featureIndex === undefined) {
            throw new Error("Index cannot be undefined");
        }
        if (value === undefined) {
            throw new Error("Value cannot be undefined");
        }
        const index = this.prevFieldCursor + featureIndex;
        this.fieldData[index] = value;
    }

    getLatestFieldFeature(args: {
        featureIndex: FieldFeatureMap[keyof FieldFeatureMap];
    }) {
        const { featureIndex } = args;
        const index = this.prevFieldCursor + featureIndex;
        return this.fieldData[index];
    }

    updateLatestEdgeFromOf(args: {
        effect: Partial<Effect>;
        edgeIndex: number;
    }) {
        const { effect, edgeIndex } = args;
        const { id, effectType, kind } = effect;
        const trueEffectType = effectType === undefined ? kind : effectType;
        if (trueEffectType !== undefined && id !== undefined) {
            const fromTypeToken = IndexValueFromEnum(
                EffecttypesEnum,
                trueEffectType,
            );
            const fromSourceToken = getEffectToken(effect);
            const numFromTypes =
                this.getLatestEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_TYPES,
                    edgeIndex,
                }) ?? 0;
            const numFromSources =
                this.getLatestEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_SOURCES,
                    edgeIndex,
                }) ?? 0;
            if (numFromTypes < 5) {
                this.setLatestEntityEdgeFeature({
                    featureIndex:
                        (EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_TYPE_TOKEN0 +
                            numFromTypes) as EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap],
                    edgeIndex,
                    value: fromTypeToken,
                });
                this.setLatestEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_TYPES,
                    edgeIndex,
                    value: numFromTypes + 1,
                });
            }
            if (numFromSources < 5) {
                this.setLatestEntityEdgeFeature({
                    featureIndex:
                        (EntityEdgeFeature.ENTITY_EDGE_FEATURE__FROM_SOURCE_TOKEN0 +
                            numFromSources) as EntityEdgeFeatureMap[keyof EntityEdgeFeatureMap],
                    edgeIndex,
                    value: fromSourceToken,
                });
                this.setLatestEntityEdgeFeature({
                    featureIndex:
                        EntityEdgeFeature.ENTITY_EDGE_FEATURE__NUM_FROM_SOURCES,
                    edgeIndex,
                    value: numFromSources + 1,
                });
            }
        }
    }

    addEdge(edge: Edge) {
        this.entityPublicData.set(
            edge.entityPublicData,
            this.entityPublicCursor,
        );
        this.entityRevealedData.set(
            edge.entityRevealedData,
            this.entityRevealedCursor,
        );
        this.entityEdgeData.set(edge.entityEdgeData, this.entityEdgeCursor);
        this.fieldData.set(edge.fieldData, this.fieldCursor);

        this.prevEntityPublicCursor = this.entityPublicCursor;
        this.prevEntityRevealedCursor = this.entityRevealedCursor;
        this.prevEntityEdgeCursor = this.entityEdgeCursor;
        this.prevFieldCursor = this.fieldCursor;

        this.entityPublicCursor += 12 * numPublicEntityNodeFeatures;
        this.entityRevealedCursor += 12 * numRevealedEntityNodeFeatures;
        this.entityEdgeCursor += 12 * numEntityEdgeFeatures;
        this.fieldCursor += numFieldFeatures;

        this.numEdges += 1;
    }

    getHistory(numHistory: number = NUM_HISTORY) {
        const historyLength = Math.max(1, Math.min(this.numEdges, numHistory));
        const historyEntityPublic = new Uint8Array(
            this.entityPublicData.slice(
                this.entityPublicCursor -
                    historyLength * 12 * numPublicEntityNodeFeatures,
                this.entityPublicCursor,
            ).buffer,
        );
        const historyEntityRevealed = new Uint8Array(
            this.entityRevealedData.slice(
                this.entityRevealedCursor -
                    historyLength * 12 * numRevealedEntityNodeFeatures,
                this.entityRevealedCursor,
            ).buffer,
        );
        const historyEntityEdges = new Uint8Array(
            this.entityEdgeData.slice(
                this.entityEdgeCursor -
                    historyLength * 12 * numEntityEdgeFeatures,
                this.entityEdgeCursor,
            ).buffer,
        );
        const historyField = new Uint8Array(
            this.fieldData.slice(
                this.fieldCursor - historyLength * numFieldFeatures,
                this.fieldCursor,
            ).buffer,
        );
        return {
            historyEntityPublic,
            historyEntityRevealed,
            historyEntityEdges,
            historyField,
            historyLength,
        };
    }

    static toReadableHistory(args: {
        historyEntityPublicBuffer: Uint8Array;
        historyEntityRevealedBuffer: Uint8Array;
        historyEntityEdgesBuffer: Uint8Array;
        historyFieldBuffer: Uint8Array;
        historyLength: number;
    }) {
        const {
            historyEntityPublicBuffer,
            historyEntityRevealedBuffer,
            historyEntityEdgesBuffer,
            historyFieldBuffer,
            historyLength,
        } = args;
        const historyItems = [];
        const historyEntityPublic = new Int16Array(
            historyEntityPublicBuffer.buffer,
        );
        const historyEntityRevealed = new Int16Array(
            historyEntityRevealedBuffer.buffer,
        );
        const historyEntityEdges = new Int16Array(
            historyEntityEdgesBuffer.buffer,
        );
        const historyField = new Int16Array(historyFieldBuffer.buffer);

        for (
            let historyIndex = 0;
            historyIndex < historyLength;
            historyIndex++
        ) {
            const stepEntityPublic = historyEntityPublic.slice(
                historyIndex * 12 * numPublicEntityNodeFeatures,
                (historyIndex + 1) * 12 * numPublicEntityNodeFeatures,
            );
            const stepEntityRevealed = historyEntityRevealed.slice(
                historyIndex * 12 * numRevealedEntityNodeFeatures,
                (historyIndex + 1) * 12 * numRevealedEntityNodeFeatures,
            );
            const stepEntityEdges = historyEntityEdges.slice(
                historyIndex * 12 * numEntityEdgeFeatures,
                (historyIndex + 1) * 12 * numEntityEdgeFeatures,
            );
            const stepField = historyField.slice(
                historyIndex * numFieldFeatures,
                (historyIndex + 1) * numFieldFeatures,
            );
            const oneToEleven = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
            historyItems.push({
                public: oneToEleven.map((memberIndex) => {
                    const start = memberIndex * numPublicEntityNodeFeatures;
                    const end = (memberIndex + 1) * numPublicEntityNodeFeatures;
                    return entityPublicArrayToObject(
                        stepEntityPublic.slice(start, end),
                    );
                }),
                revealed: oneToEleven.map((memberIndex) => {
                    const start = memberIndex * numRevealedEntityNodeFeatures;
                    const end =
                        (memberIndex + 1) * numRevealedEntityNodeFeatures;
                    return entityRevealedArrayToObject(
                        stepEntityRevealed.slice(start, end),
                    );
                }),
                edges: oneToEleven.map((memberIndex) => {
                    const start = memberIndex * numEntityEdgeFeatures;
                    const end = (memberIndex + 1) * numEntityEdgeFeatures;
                    return entityEdgeArrayToObject(
                        stepEntityEdges.slice(start, end),
                    );
                }),
                field: fieldArrayToObject(stepField),
            });
        }
        return historyItems;
    }
}

export class EventHandler implements Protocol.Handler {
    readonly player: TrainablePlayerAI;

    turnOrder: number;
    turnNum: number;
    timestamp: number;
    edgeBuffer: EdgeBuffer;
    identToIndex: Map<PokemonIdent | SideID, number>;

    constructor(player: TrainablePlayerAI) {
        this.player = player;

        this.edgeBuffer = new EdgeBuffer(player);
        this.turnOrder = 0;
        this.turnNum = 0;
        this.timestamp = 0;

        this.identToIndex = new Map<PokemonIdent, number>();
    }

    getPokemon(
        pokemonid: PokemonIdent,
        isPublic: boolean = true,
    ): {
        pokemon: Pokemon | null;
        index: number;
    } {
        if (
            !pokemonid ||
            pokemonid === "??" ||
            pokemonid === "null" ||
            pokemonid === "false"
        ) {
            return { pokemon: null, index: -1 };
        }

        if (isPublic) {
            const { pokemonid: parsedPokemonid } =
                this.player.publicBattle.parsePokemonId(pokemonid);

            if (!this.identToIndex.has(parsedPokemonid)) {
                this.identToIndex.set(parsedPokemonid, this.identToIndex.size);
            }

            return {
                pokemon: this.player.publicBattle.getPokemon(pokemonid),
                index: this.identToIndex.get(parsedPokemonid) ?? -1,
            };
        } else {
            const { siden, pokemonid: parsedPokemonid } =
                this.player.privateBattle.parsePokemonId(pokemonid);
            const side = this.player.privateBattle.sides[siden];
            for (const [index, pokemon] of side.team.entries()) {
                if (pokemon.originalIdent === parsedPokemonid) {
                    return { pokemon, index };
                }
            }
            return { pokemon: null, index: -1 };
        }
    }

    getMove(ident?: string) {
        return this.player.publicBattle.get("moves", ident) as Partial<Move> &
            NA;
    }

    getAbility(ident?: string) {
        return this.player.publicBattle.get(
            "abilities",
            ident,
        ) as Partial<Ability> & NA;
    }

    getItem(ident: string) {
        return this.player.publicBattle.get("items", ident) as Partial<Item> &
            NA;
    }

    getCondition(ident?: string) {
        if (ident) {
            if (ident.startsWith("fallen")) {
                ident = "fallen";
            }
        }
        return this.player.publicBattle.get(
            "conditions",
            ident,
        ) as Partial<Condition>;
    }

    getSide(ident: Protocol.Side) {
        return this.player.publicBattle.getSide(ident);
    }

    _preprocessEdge(edge: Edge) {
        edge.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__REQUEST_COUNT,
            value: this.player.requestCount,
        });
        edge.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__VALID,
            value: 1,
        });
        edge.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__INDEX,
            value: this.edgeBuffer.numEdges,
        });
        edge.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__TURN_ORDER_VALUE,
            value: this.turnOrder,
        });
        this.turnOrder += 1;
        edge.setFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__TURN_VALUE,
            value: this.turnNum,
        });
        return edge;
    }

    addEdge(edge: Edge) {
        const preprocessedEdge = this._preprocessEdge(edge);
        this.edgeBuffer.addEdge(preprocessedEdge);
    }

    "|move|"(args: Args["|move|"], kwArgs: KWArgs["|move|"]) {
        const [argName, pokeIdent, moveId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent as PokemonIdent);
        const edge = new Edge(this.player);

        const move = this.getMove(moveId);

        edge.addMajorArg({ argName, edgeIndex });
        edge.setEntityEdgeFeature({
            edgeIndex,
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN,
            value: IndexValueFromEnum(MovesEnum, move.id),
        });
        edge.setEntityEdgeFeature({
            edgeIndex,
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN,
            value: IndexValueFromEnum(MovesEnum, move.id),
        });
        if (kwArgs.from) {
            const fromEffect = this.getCondition(kwArgs.from);
            edge.updateEdgeFromOf({ effect: fromEffect, edgeIndex });
        }
        edge.setEntityEdgeFeature({
            edgeIndex,
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__HIT_COUNT,
            value: 1,
        });
        this.addEdge(edge);
    }

    "|player|"(args: Args["|player|"]) {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const [argName, playerId, userName] = args;
        const playerIndex = playerId.at(1);
        if (playerIndex !== undefined && this.player.userName === userName) {
            this.player.playerIndex = parseInt(playerIndex) - 1;
        }
    }

    "|drag|"(args: Args["|drag|"]) {
        this.handleSwitch(args, {});
    }

    "|switch|"(args: Args["|switch|"], kwArgs: KWArgs["|switch|"]) {
        this.handleSwitch(args, kwArgs);
    }

    handleSwitch(
        args: Args["|switch|" | "|drag|"],
        kwArgs: KWArgs["|switch|" | "|drag|"],
    ) {
        const [argName, pokeIdent, pokeDetails] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        let switchInEdgeIndex = this.getPokemon(pokeIdent).index;
        let switchOutEdgeIndex = switchInEdgeIndex;

        const switchedOut = this.player.publicBattle.getSwitchedOutPokemon(
            pokeIdent,
            pokeDetails,
        );
        if (switchedOut !== undefined) {
            switchInEdgeIndex = switchOutEdgeIndex;
            switchOutEdgeIndex = this.getPokemon(
                switchedOut.originalIdent,
            ).index;
        }
        const edge = new Edge(this.player);

        edge.addMajorArg({ argName, edgeIndex: switchOutEdgeIndex });

        edge.setEntityEdgeFeature({
            edgeIndex: switchOutEdgeIndex,
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN,
            value: MovesEnum.MOVES_ENUM___SWITCH_OUT,
        });
        edge.setEntityEdgeFeature({
            edgeIndex: switchInEdgeIndex,
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN,
            value: MovesEnum.MOVES_ENUM___SWITCH_IN,
        });

        if (argName !== "switch") {
            const from = (kwArgs as KWArgs["|switch|"]).from;
            if (from) {
                const effect = this.getCondition(from);
                edge.updateEdgeFromOf({
                    effect,
                    edgeIndex: switchOutEdgeIndex,
                });
            }
        }
        this.addEdge(edge);
    }

    "|cant|"(args: Args["|cant|"]) {
        const [argName, pokeIdent, conditionId, moveId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;
        const edge = new Edge(this.player);

        if (moveId) {
            const move = this.getMove(moveId);
            edge.setEntityEdgeFeature({
                edgeIndex,
                featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__MOVE_TOKEN,
                value: IndexValueFromEnum(MovesEnum, move.id),
            });
        }

        const condition = this.getCondition(conditionId);

        edge.addMajorArg({ argName, edgeIndex });
        edge.updateEdgeFromOf({ effect: condition, edgeIndex });

        this.addEdge(edge);
    }

    "|faint|"(args: Args["|faint|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;
        const edge = new Edge(this.player);

        edge.addMajorArg({ argName, edgeIndex });
        this.addEdge(edge);
    }

    "|-fail|"(args: Args["|-fail|"], kwArgs: KWArgs["|-fail|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect,
            edgeIndex,
        });
    }

    "|-block|"(args: Args["|-block|"], kwArgs: KWArgs["|-block|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
    }

    "|-notarget|"(args: Args["|-notarget|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        if (pokeIdent !== undefined) {
            const { index: edgeIndex } = this.getPokemon(pokeIdent)!;
            this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        } else {
            throw new Error(
                `Pokemon identifier is required for |-notarget| event: ${args}`,
            );
        }
    }

    "|-miss|"(args: Args["|-miss|"], kwArgs: KWArgs["|-miss|"]) {
        const [argName, pokeIdent, poke2Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const idents = [pokeIdent];
        if (poke2Ident !== undefined) {
            idents.push(poke2Ident);
        }
        for (const ident of idents) {
            const { index: edgeIndex } = this.getPokemon(ident)!;
            const effect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
            this.edgeBuffer.updateLatestEdgeFromOf({
                effect,
                edgeIndex,
            });
        }
    }

    "|-damage|"(
        args: Args["|-damage|"] | Args["|-sethp|"],
        kwArgs: KWArgs["|-damage|"] | KWArgs["|-sethp|"],
    ) {
        const [argName, pokeIdent, hpStatus] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon, index: edgeIndex } = this.getPokemon(pokeIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        if (kwArgs.from) {
            const effect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
        }

        const damage = pokemon.healthParse(hpStatus);
        if (damage === null) {
            throw new Error(`Invalid damage value: ${damage}`);
        }

        const addedDamageToken = Math.abs(
            Math.floor((MAX_RATIO_TOKEN * damage[0]) / damage[1]),
        );

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        const currentDamageToken =
            this.edgeBuffer.getLatestEntityEdgeFeature({
                featureIndex:
                    EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO,
                edgeIndex,
            }) ?? 0;
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__DAMAGE_RATIO,
            edgeIndex,
            value: Math.min(
                MAX_RATIO_TOKEN,
                currentDamageToken + addedDamageToken,
            ),
        });
    }

    "|-heal|"(
        args: Args["|-heal|"] | Args["|-sethp|"],
        kwArgs: KWArgs["|-heal|"] | KWArgs["|-sethp|"],
    ) {
        const [argName, pokeIdent, hpStatus] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { pokemon, index: edgeIndex } = this.getPokemon(pokeIdent);
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }

        if (kwArgs.from) {
            const effect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestEdgeFromOf({
                effect,
                edgeIndex,
            });
        }

        const damage = pokemon.healthParse(hpStatus);
        if (damage === null) {
            throw new Error(`Invalid damage value: ${damage}`);
        }

        const addedHealToken = Math.abs(
            Math.floor((MAX_RATIO_TOKEN * damage[0]) / damage[1]),
        );

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        const currentHealToken = this.edgeBuffer.getLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO,
            edgeIndex,
        });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__HEAL_RATIO,
            edgeIndex,
            value: Math.min(MAX_RATIO_TOKEN, currentHealToken + addedHealToken),
        });
    }

    "|-sethp|"(args: Args["|-sethp|"], kwArgs: KWArgs["|-sethp|"]) {
        const [argName, pokeIdent, hpStatus] = args;
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }
        const { pokemon, index: edgeIndex } = this.getPokemon(pokeIdent)!;
        if (pokemon === null) {
            throw new Error(`Pokemon ${pokeIdent} not found`);
        }
        const damage = pokemon.healthParse(hpStatus);
        if (damage === null) {
            throw new Error(`Invalid damage value: ${damage}`);
        }
        if (damage[0] < 0) {
            this["|-damage|"](
                ["-damage", args[1], args[2]] as Args["|-damage|"],
                kwArgs,
            );
        } else if (damage[0] > 0) {
            this["|-heal|"](
                ["-heal", args[1], args[2]] as Args["|-heal|"],
                kwArgs,
            );
        }
        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    }

    "|-status|"(args: Args["|-status|"], kwArgs: KWArgs["|-status|"]) {
        const [argName, pokeIdent, statusId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        const fromEffect = this.getCondition(kwArgs.from);
        const statusToken = IndexValueFromEnum(StatusEnum, statusId);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect: fromEffect,
            edgeIndex,
        });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__STATUS_TOKEN,
            edgeIndex,
            value: statusToken,
        });
    }

    "|-curestatus|"(args: Args["|-curestatus|"]) {
        const [argName, pokeIdent, statusId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        const statusToken = IndexValueFromEnum(StatusEnum, statusId);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__STATUS_TOKEN,
            edgeIndex,
            value: statusToken,
        });
    }

    "|-cureteam|"(args: Args["|-cureteam|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    }

    static getStatBoostEdgeFeatureIndex(stat: BoostID) {
        return EntityEdgeFeature[
            `ENTITY_EDGE_FEATURE__BOOST_${stat.toLocaleUpperCase()}_VALUE` as `ENTITY_EDGE_FEATURE__BOOST_${Uppercase<BoostID>}_VALUE`
        ];
    }

    "|-boost|"(args: Args["|-boost|"]) {
        const [argName, pokeIdent, stat, value] = args;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex,
            edgeIndex,
            value: parseInt(value),
        });
    }

    "|-unboost|"(args: Args["|-unboost|"]) {
        const [argName, pokeIdent, stat, value] = args;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex,
            edgeIndex,
            value: -parseInt(value),
        });
    }

    "|-setboost|"(args: Args["|-setboost|"], kwArgs: KWArgs["|-setboost|"]) {
        const [argName, pokeIdent, stat, value] = args;

        const featureIndex = EventHandler.getStatBoostEdgeFeatureIndex(stat);
        const effect = this.getCondition(kwArgs.from);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex,
            edgeIndex,
            value: parseInt(value),
        });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
    }

    "|-swapboost|"(args: Args["|-swapboost|"], kwArgs: KWArgs["|-swapboost|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
    }

    "|-invertboost|"(
        args: Args["|-invertboost|"],
        kwArgs: KWArgs["|-invertboost|"],
    ) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
    }

    "|-clearboost|"(
        args: Args["|-clearboost|"],
        kwArgs: KWArgs["|-clearboost|"],
    ) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
    }

    "|-clearnegativeboost|"(
        args: Args["|-clearnegativeboost|"],
        kwArgs: KWArgs["|-clearnegativeboost|"],
    ) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent);

        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
    }

    "|-copyboost|"() {}

    "|-weather|"(args: Args["|-weather|"]) {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const [argName, weatherId] = args;

        const weatherIndex =
            weatherId === "none"
                ? WeatherEnum.WEATHER_ENUM___NULL
                : IndexValueFromEnum(WeatherEnum, weatherId);

        this.edgeBuffer.setLatestFieldFeature({
            featureIndex: FieldFeature.FIELD_FEATURE__WEATHER_ID,
            value: weatherIndex,
        });
    }

    "|-fieldstart|"() {
        // kwArgs: KWArgs["|-fieldstart|"], // args: Args["|-fieldstart|"],
        // const [argName] = args;
    }

    "|-fieldend|"() {
        // args: Args["|-fieldend|"], kwArgs: KWArgs["|-fieldend|"]
        // const [argName] = args;
    }

    "|-sidestart|"(args: Args["|-sidestart|"]) {
        const [argName, sideId, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const side = this.getSide(sideId);
        const effect = this.getCondition(conditionId);

        for (const pokemon of side.team) {
            const ident = pokemon.originalIdent;
            const { index: edgeIndex } = this.getPokemon(ident);
            this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
            this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
        }
    }

    "|-sideend|"(args: Args["|-sideend|"]) {
        const [argName, sideId, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const side = this.getSide(sideId);
        const effect = this.getCondition(conditionId);

        for (const pokemon of side.team) {
            const ident = pokemon.originalIdent;
            const { index: edgeIndex } = this.getPokemon(ident);
            this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
            this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
        }
    }

    "|-swapsideconditions|"() {}

    "|-start|"(args: Args["|-start|"], kwArgs: KWArgs["|-start|"]) {
        const [argName, pokeIdent, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect: this.getCondition(kwArgs.from),
            edgeIndex,
        });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect: this.getCondition(
                conditionId.startsWith("perish") ? "perishsong" : conditionId,
            ),
            edgeIndex,
        });
    }

    "|-end|"(args: Args["|-end|"], kwArgs: KWArgs["|-end|"]) {
        const [argName, pokeIdent, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect: this.getCondition(kwArgs.from),
            edgeIndex,
        });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect: this.getCondition(conditionId),
            edgeIndex,
        });
    }

    "|-crit|"(args: Args["|-crit|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent);
        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    }

    "|-supereffective|"(args: Args["|-supereffective|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent);
        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    }

    "|-resisted|"(args: Args["|-resisted|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent);
        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    }

    "|-immune|"(args: Args["|-immune|"], kwArgs: KWArgs["|-immune|"]) {
        const [argName, pokeIdent, conditionId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent);
        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect: this.getCondition(kwArgs.from),
            edgeIndex,
        });
        this.edgeBuffer.updateLatestEdgeFromOf({
            effect: this.getCondition(conditionId),
            edgeIndex,
        });
    }

    "|-item|"(args: Args["|-item|"], kwArgs: KWArgs["|-item|"]) {
        const [argName, pokeIdent, itemId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent as PokemonIdent);
        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });

        const itemIndex = IndexValueFromEnum(ItemsEnum, itemId);
        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__ITEM_TOKEN,
            edgeIndex,
            value: itemIndex,
        });
    }

    "|-enditem|"(args: Args["|-enditem|"], kwArgs: KWArgs["|-enditem|"]) {
        const [argName, pokeIdent, itemId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent);
        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });

        const itemIndex = IndexValueFromEnum(ItemsEnum, itemId);
        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__ITEM_TOKEN,
            edgeIndex,
            value: itemIndex,
        });
    }

    "|-ability|"(args: Args["|-ability|"], kwArgs: KWArgs["|-ability|"]) {
        const [argName, pokeIdent, abilityId] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent);

        const abilityIndex = IndexValueFromEnum(AbilitiesEnum, abilityId);

        if (kwArgs.from) {
            const effect = this.getCondition(kwArgs.from);
            this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
        }

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__ABILITY_TOKEN,
            edgeIndex,
            value: abilityIndex,
        });
    }

    "|-endability|"(
        args: Args["|-endability|"],
        kwArgs: KWArgs["|-endability|"],
    ) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        const effect = this.getCondition(kwArgs.from);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
    }

    rememberTransformed(
        pokeIdent: Protocol.PokemonIdent,
        poke2Ident: Protocol.PokemonIdent,
    ) {
        const { pokemon: srcPokemon } = this.getPokemon(pokeIdent, false)!;
        const { pokemon: tgtPokemon } = this.getPokemon(poke2Ident)!;

        if (srcPokemon !== null && tgtPokemon !== null) {
            const transformedPokemon =
                srcPokemon?.volatiles?.transform?.pokemon;
            if (transformedPokemon === undefined) {
                return;
            }
            const currentRememberedMoves = new Set(
                tgtPokemon.moveSlots.map((x) => x.id),
            );
            for (const { id } of srcPokemon.moveSlots.slice(0, 4)) {
                if (!currentRememberedMoves.has(id)) {
                    tgtPokemon.rememberMove(id);
                }
            }
            tgtPokemon.rememberAbility(transformedPokemon.ability);
        }
    }

    "|-transform|"(args: Args["|-transform|"]) {
        const [argName, pokeIdent, poke2Ident] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent);

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.rememberTransformed(pokeIdent, poke2Ident);
    }

    "|-mega|"() {}

    "|-primal|"() {}

    "|-burst|"() {}

    "|-zpower|"() {}

    "|-zbroken|"() {}

    // Suprisingly not needed?

    // "|-terastallize|"(args: Args["|-terastallize|"]) {
    // const [argName, pokeIdent] = args;
    // const playerIndex = this.player.getPlayerIndex();
    // if (playerIndex === undefined) {
    //     throw new Error();
    // }
    // const { index: edgeIndex } = this.getPokemon(pokeIdent)!;
    // this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    // }

    "|-activate|"(args: Args["|-activate|"], kwArgs: KWArgs["|-activate|"]) {
        const [argName, pokeIdent, conditionId1] = args;

        if (pokeIdent) {
            const playerIndex = this.player.getPlayerIndex();
            if (playerIndex === undefined) {
                throw new Error();
            }

            const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

            this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
            for (const effect of [
                this.getCondition(kwArgs.from),
                this.getCondition(conditionId1),
                // this.getCondition(conditionId2),
            ]) {
                this.edgeBuffer.updateLatestEdgeFromOf({ effect, edgeIndex });
            }
        }
    }

    "|-mustrecharge|"(args: Args["|-mustrecharge|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    }

    "|-prepare|"(args: Args["|-prepare|"]) {
        const [argName, pokeIdent] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
    }

    "|-hitcount|"(args: Args["|-hitcount|"]) {
        const [argName, pokeIdent, numHits] = args;

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error();
        }

        const { index: edgeIndex } = this.getPokemon(pokeIdent)!;

        this.edgeBuffer.updateLatestMinorArgs({ argName, edgeIndex });
        this.edgeBuffer.setLatestEntityEdgeFeature({
            featureIndex: EntityEdgeFeature.ENTITY_EDGE_FEATURE__HIT_COUNT,
            edgeIndex,
            value: parseInt(numHits),
        });
    }

    "|done|"(args: Args["|done|"]) {
        const [argName] = args;

        let edge = undefined;
        for (const side of this.player.publicBattle.sides) {
            for (const active of side.active) {
                if (active !== null) {
                    const { index: edgeIndex } = this.getPokemon(
                        active.originalIdent,
                    );
                    if (edge === undefined) {
                        edge = new Edge(this.player);
                    }
                    if (edgeIndex >= 0) {
                        edge.addMajorArg({ argName, edgeIndex });
                    }
                }
            }
        }
        if (edge !== undefined && this.turnOrder > 0) {
            this.addEdge(edge);
        }
    }

    "|start|"() {
        this.turnOrder = 0;

        const edge = new Edge(this.player);
        this.addEdge(edge);
    }

    "|teampreview|"() {
        this.turnOrder = 0;

        const edge = new Edge(this.player);
        for (let i = 0; i < 12; i++)
            edge.addMajorArg({ argName: "poke", edgeIndex: i });
        this.addEdge(edge);
    }

    "|t:|"(args: Args["|t:|"]) {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const [_, timestamp] = args;
        this.timestamp = parseInt(timestamp);
    }

    "|turn|"(args: Args["|turn|"]) {
        const turnNum = (args.at(1) ?? "").toString();

        this.turnOrder = 0;
        this.turnNum = parseInt(turnNum);
    }

    "|win|"() {
        this.player.done = true;
    }

    "|tie|"() {
        this.player.done = true;
    }
}

class PrivateActionHandler {
    player: TrainablePlayerAI;
    request: AnyObject;
    actionBuffer: Int16Array;

    constructor(player: TrainablePlayerAI) {
        this.player = player;
        this.request = player.getRequest();

        // const numActive = (this.request?.active ?? []).length;
        this.actionBuffer = new Int16Array(1 * 4 * numMoveFeatures);
    }

    assignActionBuffer(args: { offset: number; index: number; value: number }) {
        const { offset, index, value } = args;
        this.actionBuffer[offset + index] = value;
    }

    pushMoveAction(
        actionOffset: number,
        move:
            | { name: "Recharge"; id: "recharge" }
            | { name: Protocol.MoveName; id: ID }
            | {
                  name: Protocol.MoveName;
                  id: ID;
                  pp: number;
                  maxpp: number;
                  target: MoveTarget;
                  disabled?: boolean;
              },
    ) {
        if ("pp" in move) {
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__HAS_PP,
                value: MovesetHasPP.MOVESET_HAS_PP__YES,
            });
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__PP,
                value: move.pp,
            });
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__MAXPP,
                value: move.maxpp,
            });
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__PP_RATIO,
                value: MAX_RATIO_TOKEN * (move.pp / move.maxpp),
            });
        } else {
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__HAS_PP,
                value: MovesetHasPP.MOVESET_HAS_PP__NO,
            });
        }
        let moveId = move.id;
        if (moveId.startsWith("return")) {
            moveId = "return" as ID;
        } else if (moveId.startsWith("frustration")) {
            moveId = "frustration" as ID;
        } else if (moveId.startsWith("hiddenpower")) {
            const power = parseInt(moveId.slice(-2));
            if (isNaN(power)) {
                moveId = "hiddenpower" as ID;
            } else {
                moveId = moveId.slice(0, -2) as ID;
            }
        }
        this.assignActionBuffer({
            offset: actionOffset,
            index: MovesetFeature.MOVESET_FEATURE__MOVE_ID,
            value: IndexValueFromEnum(MovesEnum, moveId),
        });
        this.assignActionBuffer({
            offset: actionOffset,
            index: MovesetFeature.MOVESET_FEATURE__ACTION_TYPE,
            value: ActionType.ACTION_TYPE__MOVE,
        });
    }

    build() {
        const actives = (this.request?.active ?? [
            null,
        ]) as Protocol.MoveRequest["active"];
        const switches = (this.request?.side?.pokemon ??
            []) as Protocol.Request.SideInfo["pokemon"];

        let actionOffset = 0;

        for (const [activeIndex, activePokemon] of actives.entries()) {
            let moves = [];
            if (activePokemon !== null) {
                const { pokemon, index: entityIndex } =
                    this.player.eventHandler.getPokemon(
                        switches[activeIndex].ident,
                        false,
                    );
                if (pokemon === null) {
                    throw new Error(
                        `Pokemon ${switches[activeIndex].ident} not found`,
                    );
                }
                moves = activePokemon.moves;
                for (const action of moves) {
                    this.pushMoveAction(actionOffset, action);
                    this.assignActionBuffer({
                        offset: actionOffset,
                        index: MovesetFeature.MOVESET_FEATURE__ENTITY_IDX,
                        value: entityIndex,
                    });
                    actionOffset += numMoveFeatures;
                }
            }
            actionOffset += (4 - moves.length) * numMoveFeatures;
        }

        return new Uint8Array(this.actionBuffer.buffer);
    }
}

class PublicActionHandler {
    player: TrainablePlayerAI;
    actionBuffer: Int16Array;

    constructor(player: TrainablePlayerAI) {
        this.player = player;

        const request = player.getRequest();
        const numActive = (request?.active ?? []).length;
        this.actionBuffer = new Int16Array(
            (numActive * 4 + 6) * numMoveFeatures,
        );
    }

    assignActionBuffer(args: { offset: number; index: number; value: number }) {
        const { offset, index, value } = args;
        this.actionBuffer[offset + index] = value;
    }

    pushMoveAction(actionOffset: number, move: Pokemon["moveSlots"][number]) {
        if ("ppUsed" in move) {
            const moveData = this.player.eventHandler.getMove(move.id);
            if (moveData === undefined || moveData.pp === undefined) {
                throw new Error(`Move ${move.id} not found`);
            }
            const maxpp = (8 / 5) * moveData.pp;
            const pp = maxpp - move.ppUsed;
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__HAS_PP,
                value: MovesetHasPP.MOVESET_HAS_PP__YES,
            });
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__PP,
                value: pp,
            });
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__MAXPP,
                value: maxpp,
            });
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__PP_RATIO,
                value: MAX_RATIO_TOKEN * (pp / maxpp),
            });
        } else {
            this.assignActionBuffer({
                offset: actionOffset,
                index: MovesetFeature.MOVESET_FEATURE__HAS_PP,
                value: MovesetHasPP.MOVESET_HAS_PP__NO,
            });
        }
        this.assignActionBuffer({
            offset: actionOffset,
            index: MovesetFeature.MOVESET_FEATURE__MOVE_ID,
            value: IndexValueFromEnum(MovesEnum, move.id),
        });
        this.assignActionBuffer({
            offset: actionOffset,
            index: MovesetFeature.MOVESET_FEATURE__ACTION_TYPE,
            value: ActionType.ACTION_TYPE__MOVE,
        });
    }

    pushSwitchAction(actionOffset: number, switchIndex: number) {
        this.assignActionBuffer({
            offset: actionOffset,
            index: MovesetFeature.MOVESET_FEATURE__MOVE_ID,
            value: MovesEnum.MOVES_ENUM___SWITCH_IN,
        });
        this.assignActionBuffer({
            offset: actionOffset,
            index: MovesetFeature.MOVESET_FEATURE__ACTION_TYPE,
            value: ActionType.ACTION_TYPE__SWITCH,
        });
        this.assignActionBuffer({
            offset: actionOffset,
            index: MovesetFeature.MOVESET_FEATURE__HAS_PP,
            value: MovesetHasPP.MOVESET_HAS_PP__NO,
        });
        this.assignActionBuffer({
            offset: actionOffset,
            index: MovesetFeature.MOVESET_FEATURE__ENTITY_IDX,
            value: switchIndex,
        });
    }

    build() {
        const side =
            this.player.publicBattle.sides[1 - this.player.getPlayerIndex()!];
        const actives = side.active;
        const switches = side.team;

        let actionOffset = 0;

        for (const [activeIndex, activePokemon] of actives.entries()) {
            let moves = [];
            if (activePokemon !== null) {
                const { pokemon, index: entityIndex } =
                    this.player.eventHandler.getPokemon(
                        switches[activeIndex].ident,
                        false,
                    );
                if (pokemon === null) {
                    throw new Error(
                        `Pokemon ${switches[activeIndex].ident} not found`,
                    );
                }
                moves = activePokemon.moveSlots.slice(0, 4);
                for (const move of moves) {
                    this.pushMoveAction(actionOffset, move);
                    this.assignActionBuffer({
                        offset: actionOffset,
                        index: MovesetFeature.MOVESET_FEATURE__ENTITY_IDX,
                        value: entityIndex,
                    });
                    actionOffset += numMoveFeatures;
                }
                for (let i = 0; i < 4 - moves.length; i++) {
                    this.pushMoveAction(actionOffset, {
                        ppUsed: 0,
                        name: "" as Protocol.MoveName,
                        id: "" as ID,
                    });
                    this.assignActionBuffer({
                        offset: actionOffset,
                        index: MovesetFeature.MOVESET_FEATURE__ENTITY_IDX,
                        value: entityIndex,
                    });
                    actionOffset += numMoveFeatures;
                }
                for (const [switchIndex] of switches.entries()) {
                    this.pushSwitchAction(actionOffset, switchIndex);
                    actionOffset += numMoveFeatures;
                }
                actionOffset += (6 - switches.length) * numMoveFeatures;
            } else {
                actionOffset += moves.length * numMoveFeatures;
                actionOffset += switches.length * numMoveFeatures;
            }
        }

        return new Uint8Array(this.actionBuffer.buffer);
    }
}

export class RewardTracker {
    prevFaintedCount: [number, number];
    currFaintedCount: [number, number];

    funcCache: Map<number, number>;

    constructor() {
        this.prevFaintedCount = [0, 0];
        this.currFaintedCount = [0, 0];

        const B = (x: number) => {
            return x ** 2;
        };
        const B_int = (x: number) => {
            return (1 / 3) * x ** 3;
        };
        // Integrate B from 0 - 6
        const A = B_int(6);
        const func = (x: number) => {
            return (2 / A) * B(x);
        };
        this.funcCache = new Map();
        [0, 1, 2, 3, 4, 5, 6].forEach((x) => {
            this.funcCache.set(x, func(x));
        });
    }

    updateFaintedCount(battle: Battle) {
        this.prevFaintedCount = this.currFaintedCount;
        this.currFaintedCount = battle.sides.map(
            (side) => side.team.filter((poke) => poke.fainted).length,
        ) as [number, number];
    }

    getFibReward(playerIndex: number) {
        const [prevp1, prevp2] = this.prevFaintedCount.map(
            (x) => this.funcCache.get(x)!,
        );
        const [currp1, currp2] = this.currFaintedCount.map(
            (x) => this.funcCache.get(x)!,
        );
        const reward = currp2 - prevp2 - (currp1 - prevp1);
        const sign = playerIndex === 0 ? 1 : -1;
        return sign * reward;
    }
}

export class StateHandler {
    player: TrainablePlayerAI;

    constructor(player: TrainablePlayerAI) {
        this.player = player;
    }

    static getActionMask(args: {
        request?: AnyObject | null;
        maskMoves?: boolean | null;
        maskTera?: boolean | null;
    }): {
        actionMask: OneDBoolean;
        isStruggling: boolean;
    } {
        const { request } = args;
        const maskMoves = args.maskMoves ?? false;
        const maskTera = args.maskTera ?? false;

        const actionMask = new OneDBoolean(numActionMaskFeatures, Uint8Array);
        let isStruggling = false;

        actionMask.set(ActionMaskFeature.ACTION_MASK_FEATURE__CAN_NORMAL, true);

        const setAllTrue = () => {
            for (let i = 0; i < numActionMaskFeatures; i++) {
                actionMask.set(i, true);
            }
        };

        if (request === undefined || request === null) {
            setAllTrue();
        } else {
            if (request.wait) {
                setAllTrue();
            } else if (request.forceSwitch) {
                actionMask.set(
                    ActionMaskFeature.ACTION_MASK_FEATURE__CAN_MOVE,
                    false,
                );
                actionMask.set(
                    ActionMaskFeature.ACTION_MASK_FEATURE__CAN_SWITCH,
                    true,
                );

                const pokemon = request.side
                    .pokemon as Protocol.Request.SideInfo["pokemon"];
                const forceSwitchLength = request.forceSwitch.length;
                const isReviving = !!pokemon[0].reviving;

                for (let j = 0; j < 6; j++) {
                    const currentPokemon = pokemon[j];
                    if (
                        currentPokemon &&
                        j >= forceSwitchLength &&
                        (isReviving ? 1 : 0) ^
                            (currentPokemon.condition.endsWith(" fnt") ? 0 : 1)
                    ) {
                        const switchIndex =
                            ActionMaskFeature.ACTION_MASK_FEATURE__SWITCH_SLOT_1 +
                            j;
                        actionMask.set(switchIndex, true);
                    }
                }
            } else if (request.active) {
                const pokemon = request.side.pokemon;
                const active = request.active[0];

                const { canMegaEvo, canDynamax, canTerastallize } = active;
                actionMask.set(
                    ActionMaskFeature.ACTION_MASK_FEATURE__CAN_MEGA,
                    !!canMegaEvo,
                );
                actionMask.set(
                    ActionMaskFeature.ACTION_MASK_FEATURE__CAN_MAX,
                    !!canDynamax,
                );

                const noOtherTeras = !pokemon.some(
                    (x: { terastallized?: string }) =>
                        (x?.terastallized ?? "") !== "",
                );
                actionMask.set(
                    ActionMaskFeature.ACTION_MASK_FEATURE__CAN_TERA,
                    !!canTerastallize && noOtherTeras && !maskTera,
                );

                const possibleMoves = active.moves ?? [];
                const canSwitch = [];

                for (let j = 0; j < 6; j++) {
                    const currentPokemon = pokemon[j];
                    if (
                        currentPokemon &&
                        !currentPokemon.active &&
                        !currentPokemon.condition.endsWith(" fnt")
                    ) {
                        const switchIndex =
                            ActionMaskFeature.ACTION_MASK_FEATURE__SWITCH_SLOT_1 +
                            j;
                        canSwitch.push(switchIndex);
                    }
                }

                const switches = active.trapped ? [] : canSwitch;
                const canAddMove = !maskMoves || switches.length === 0;
                let canMove = false;

                for (let j = 0; j < possibleMoves.length; j++) {
                    const currentMove = possibleMoves[j];
                    if (currentMove.id === "struggle") {
                        isStruggling = true;
                    }
                    if ((!currentMove.disabled && canAddMove) || isStruggling) {
                        const moveIndex =
                            ActionMaskFeature.ACTION_MASK_FEATURE__MOVE_SLOT_1 +
                            j;
                        actionMask.set(moveIndex, true);
                        canMove = true;
                    }
                }
                if (canMove) {
                    actionMask.set(
                        ActionMaskFeature.ACTION_MASK_FEATURE__CAN_MOVE,
                        true,
                    );
                }

                if (switches.length > 0) {
                    actionMask.set(
                        ActionMaskFeature.ACTION_MASK_FEATURE__CAN_SWITCH,
                        true,
                    );
                }
                for (const switchIndex of switches) {
                    actionMask.set(switchIndex, true);
                }
            } else if (request.teamPreview) {
                const pokemon = request.side.pokemon;

                actionMask.set(
                    ActionMaskFeature.ACTION_MASK_FEATURE__CAN_MOVE,
                    false,
                );
                actionMask.set(
                    ActionMaskFeature.ACTION_MASK_FEATURE__CAN_SWITCH,
                    false,
                );
                actionMask.set(
                    ActionMaskFeature.ACTION_MASK_FEATURE__CAN_TEAMPREVIEW,
                    true,
                );

                for (let j = 0; j < 6; j++) {
                    const currentPokemon = pokemon[j];
                    if (currentPokemon) {
                        const switchIndex =
                            ActionMaskFeature.ACTION_MASK_FEATURE__SWITCH_SLOT_1 +
                            j;
                        actionMask.set(switchIndex, true);
                    }
                }
            }
        }
        return { actionMask, isStruggling };
    }

    getMoveset(): Uint8Array {
        const actionHandler = new PrivateActionHandler(this.player);
        return actionHandler.build();
    }

    getOppMoveset(): Uint8Array {
        const actionHandler = new PublicActionHandler(this.player);
        return actionHandler.build();
    }

    getPublicTeamFromSide(playerIndex: number): {
        publicBuffer: Int16Array;
        revealedBuffer: Int16Array;
    } {
        const side = this.player.publicBattle.sides[playerIndex];
        const publicBuffer = new Int16Array(6 * numPublicEntityNodeFeatures);
        const revealedBuffer = new Int16Array(
            6 * numRevealedEntityNodeFeatures,
        );

        let publicOffset = 0;
        let revealedOffset = 0;
        let team = side.team.slice(0, 6);

        const relativeSide = isMySide(side.n, this.player.getPlayerIndex());

        try {
            for (const member of team) {
                const { publicData, revealedData } = getArrayFromPublicPokemon(
                    member,
                    relativeSide,
                );
                publicBuffer.set(publicData, publicOffset);
                revealedBuffer.set(revealedData, revealedOffset);
                publicOffset += numPublicEntityNodeFeatures;
                revealedOffset += numRevealedEntityNodeFeatures;
            }

            const { publicData, revealedData } = relativeSide
                ? unkPokemon1
                : unkPokemon0;
            for (let i = team.length; i < side.totalPokemon; i++) {
                publicBuffer.set(publicData, publicOffset);
                revealedBuffer.set(revealedData, revealedOffset);
                publicOffset += numPublicEntityNodeFeatures;
                revealedOffset += numRevealedEntityNodeFeatures;
            }

            for (let i = side.totalPokemon; i < 6; i++) {
                revealedBuffer.set(revealedData, revealedOffset);
                revealedOffset += numRevealedEntityNodeFeatures;
            }
        } catch (error) {
            console.log(error);
            console.log(team);
            return { publicBuffer, revealedBuffer };
        }

        if (publicOffset !== publicBuffer.length) {
            throw new Error(
                `Buffer length mismatch: expected ${publicBuffer.length}, got ${publicOffset}`,
            );
        }
        return { publicBuffer, revealedBuffer };
    }

    getPrivateTeam(playerIndex: number): Int16Array {
        let sets =
            this.player.privateBattle.sides[playerIndex].sets ??
            this.player.privateBattle.sides[1 - playerIndex].sets;
        if (sets === undefined) {
            throw new Error("Team is undefined");
        }

        const request = this.player.getRequest();
        if (request === undefined) {
            throw new Error("Request is undefined");
        }
        const requestPokemon = request.side?.pokemon as
            | Protocol.Request.SideInfo["pokemon"]
            | undefined;

        let offset = 0;
        const buffer = new Int16Array(6 * numPrivateEntityNodeFeatures);

        if (requestPokemon === undefined) {
            throw new Error("Request pokemon is undefined");
        } else {
            for (const member of requestPokemon) {
                const name = toID(member.speciesForme);
                const matchedSet = sets.find((set) => {
                    const setSpecies = toID(set.species);
                    return (
                        setSpecies === name ||
                        setSpecies.includes(name) ||
                        name.includes(setSpecies)
                    );
                });
                const matchedTeamMate = this.player.privateBattle.sides[
                    playerIndex
                ].team.find((teamMate) => {
                    const setSpecies = toID(
                        teamMate.baseSpecies.baseSpecies.toLowerCase(),
                    );
                    return (
                        setSpecies === name ||
                        setSpecies.includes(name) ||
                        name.includes(setSpecies)
                    );
                });
                if (matchedSet === undefined) {
                    throw new Error(`Pokemon ${name} not found in team`);
                }

                buffer.set(
                    getArrayFromPrivatePokemon(matchedTeamMate, matchedSet),
                    offset,
                );
                offset += numPrivateEntityNodeFeatures;
            }
        }

        return buffer;
    }

    getPublicTeam(playerIndex: number): {
        publicData: Int16Array;
        revealedData: Int16Array;
    } {
        const publicDataArr = [];
        const revealedDataArr = [];
        for (const idx of [playerIndex, 1 - playerIndex]) {
            const { publicBuffer, revealedBuffer } =
                this.getPublicTeamFromSide(idx);
            publicDataArr.push(publicBuffer);
            revealedDataArr.push(revealedBuffer);
        }

        return {
            publicData: concatenateArrays(publicDataArr),
            revealedData: concatenateArrays(revealedDataArr),
        };
    }

    getHistory(numHistory: number = NUM_HISTORY) {
        return this.player.eventHandler.edgeBuffer.getHistory(numHistory);
    }

    getWinReward() {
        if (this.player.done) {
            if (this.player.finishedEarly) {
                // Prevent reward hacking by stalling
                return Math.floor(MAX_RATIO_TOKEN * -0.5);
            }
            for (let i = this.player.log.length - 1; i >= 0; i--) {
                const line = this.player.log.at(i) ?? "";
                // eslint-disable-next-line @typescript-eslint/no-unused-vars
                const [_, cmd, winner] = line.split("|");
                if (cmd === "win") {
                    return this.player.userName === winner
                        ? MAX_RATIO_TOKEN
                        : -MAX_RATIO_TOKEN;
                } else if (cmd === "tie") {
                    return 0;
                }
            }
        }
        return 0;
    }

    getFibReward() {
        return Math.floor(
            MAX_RATIO_TOKEN *
                this.player.rewardTracker.getFibReward(
                    this.player.getPlayerIndex()!,
                ),
        );
    }

    getInfo() {
        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error("Player index is undefined");
        }

        const infoBuffer = new Int16Array(numInfoFeatures);

        infoBuffer[InfoFeature.INFO_FEATURE__PLAYER_INDEX] = playerIndex;
        infoBuffer[InfoFeature.INFO_FEATURE__TURN] =
            this.player.privateBattle.turn;
        infoBuffer[InfoFeature.INFO_FEATURE__DONE] = +this.player.done;
        infoBuffer[InfoFeature.INFO_FEATURE__REQUEST_COUNT] =
            this.player.requestCount;

        infoBuffer[InfoFeature.INFO_FEATURE__WIN_REWARD] = this.getWinReward();
        infoBuffer[InfoFeature.INFO_FEATURE__FIB_REWARD] = this.getFibReward();

        const getHpRatio = (member: Pokemon) => {
            const isHpBug = !member.fainted && member.hp === 0;
            const hp = isHpBug ? 100 : member.hp;
            const maxHp = isHpBug ? 100 : member.maxhp;
            return hp / maxHp;
        };

        let [myFaintedCount, myHpCount] = [0, 0];
        const mySide = this.player.privateBattle.sides[playerIndex];
        for (const member of mySide.team) {
            if (member.fainted) {
                myFaintedCount += 1;
            } else {
                myHpCount += getHpRatio(member);
            }
        }
        myHpCount += mySide.totalPokemon - mySide.team.length;

        let [oppFaintedCount, oppHpCount] = [0, 0];
        const oppSide = this.player.privateBattle.sides[1 - playerIndex];
        for (const member of oppSide.team) {
            if (member.fainted) {
                oppFaintedCount += 1;
            } else {
                oppHpCount += getHpRatio(member);
            }
        }
        oppHpCount += oppSide.totalPokemon - oppSide.team.length;

        infoBuffer[InfoFeature.INFO_FEATURE__MY_FAINTED_COUNT] = Math.floor(
            (MAX_RATIO_TOKEN * myFaintedCount) / mySide.totalPokemon,
        );
        infoBuffer[InfoFeature.INFO_FEATURE__OPP_FAINTED_COUNT] = Math.floor(
            (MAX_RATIO_TOKEN * oppFaintedCount) / oppSide.totalPokemon,
        );
        infoBuffer[InfoFeature.INFO_FEATURE__MY_HP_COUNT] = Math.floor(
            (MAX_RATIO_TOKEN * myHpCount) / mySide.totalPokemon,
        );
        infoBuffer[InfoFeature.INFO_FEATURE__OPP_HP_COUNT] = Math.floor(
            (MAX_RATIO_TOKEN * oppHpCount) / oppSide.totalPokemon,
        );

        return new Uint8Array(infoBuffer.buffer);
    }

    static toReadablePrivate(buffer: Uint8Array) {
        const teamBuffer = new Int16Array(buffer.buffer);
        const entityDatums = [];
        const numEntites = teamBuffer.length / numPrivateEntityNodeFeatures;
        for (let entityIndex = 0; entityIndex < numEntites; entityIndex++) {
            const start = entityIndex * numPrivateEntityNodeFeatures;
            const end = (entityIndex + 1) * numPrivateEntityNodeFeatures;
            entityDatums.push(
                entityPrivateArrayToObject(teamBuffer.slice(start, end)),
            );
        }
        return entityDatums;
    }

    static toReadablePublic(buffer: Uint8Array) {
        const teamBuffer = new Int16Array(buffer.buffer);
        const entityDatums = [];
        const numEntites = teamBuffer.length / numPublicEntityNodeFeatures;
        for (let entityIndex = 0; entityIndex < numEntites; entityIndex++) {
            const start = entityIndex * numPublicEntityNodeFeatures;
            const end = (entityIndex + 1) * numPublicEntityNodeFeatures;
            entityDatums.push(
                entityPublicArrayToObject(teamBuffer.slice(start, end)),
            );
        }
        return entityDatums;
    }

    static toReadableRevealed(buffer: Uint8Array) {
        const teamBuffer = new Int16Array(buffer.buffer);
        const entityDatums = [];
        const numEntites = teamBuffer.length / numRevealedEntityNodeFeatures;
        for (let entityIndex = 0; entityIndex < numEntites; entityIndex++) {
            const start = entityIndex * numRevealedEntityNodeFeatures;
            const end = (entityIndex + 1) * numRevealedEntityNodeFeatures;
            entityDatums.push(
                entityRevealedArrayToObject(teamBuffer.slice(start, end)),
            );
        }
        return entityDatums;
    }

    static toReadableMoveset(buffer: Uint8Array) {
        const movesetBuffer = new Int16Array(buffer.buffer);
        const entityDatums = [];
        const numMoves = movesetBuffer.length / numMoveFeatures;
        for (let moveIndex = 0; moveIndex < numMoves; moveIndex++) {
            const start = moveIndex * numMoveFeatures;
            const end = (moveIndex + 1) * numMoveFeatures;
            entityDatums.push(
                moveArrayToObject(movesetBuffer.slice(start, end)),
            );
        }
        return entityDatums;
    }

    getField() {
        const fieldBuffer = new Int16Array(numFieldFeatures);
        const playerIndex = this.player.getPlayerIndex()!;
        for (const side of this.player.privateBattle.sides) {
            const relativeSide = isMySide(side.n, playerIndex);
            const sideOffset = relativeSide
                ? FieldFeature.FIELD_FEATURE__MY_SIDECONDITIONS0
                : FieldFeature.FIELD_FEATURE__OPP_SIDECONDITIONS0;
            const spikesOffset = relativeSide
                ? FieldFeature.FIELD_FEATURE__MY_SPIKES
                : FieldFeature.FIELD_FEATURE__OPP_SPIKES;
            const toxisSpikesOffset = relativeSide
                ? FieldFeature.FIELD_FEATURE__MY_TOXIC_SPIKES
                : FieldFeature.FIELD_FEATURE__OPP_TOXIC_SPIKES;

            let sideConditionBuffer = BigInt(0b0);
            for (const [id] of Object.entries(side.sideConditions)) {
                const featureIndex = IndexValueFromEnum(SideconditionEnum, id);
                sideConditionBuffer |= BigInt(1) << BigInt(featureIndex);
            }
            fieldBuffer.set(
                bigIntToInt16Array(sideConditionBuffer),
                sideOffset,
            );
            if (side.sideConditions.spikes) {
                fieldBuffer[spikesOffset] = side.sideConditions.spikes.level;
            }
            if (side.sideConditions.toxicspikes) {
                fieldBuffer[toxisSpikesOffset] =
                    side.sideConditions.toxicspikes.level;
            }
        }
        const field = this.player.publicBattle.field;
        const weatherIndex = field.weatherState.id
            ? IndexValueFromEnum(
                  WeatherEnum,
                  WEATHERS[field.weatherState.id as never],
              )
            : WeatherEnum.WEATHER_ENUM___NULL;

        fieldBuffer[FieldFeature.FIELD_FEATURE__WEATHER_ID] = weatherIndex;
        fieldBuffer[FieldFeature.FIELD_FEATURE__WEATHER_MAX_DURATION] =
            field.weatherState.maxDuration;
        fieldBuffer[FieldFeature.FIELD_FEATURE__WEATHER_MIN_DURATION] =
            field.weatherState.minDuration;
        return new Uint8Array(fieldBuffer.buffer);
    }

    build(): EnvironmentState {
        this.player.rewardTracker.updateFaintedCount(this.player.privateBattle);

        const request = this.player.getRequest();
        if (!this.player.done && request === undefined) {
            throw new Error("Need Request");
        }

        const state = new EnvironmentState();
        const info = this.getInfo();
        state.setInfo(info);

        const maskTera = false;
        // const maskTera =
        //     this.player.publicBattle.turn < this.player.earliestTeraTurn;
        const { actionMask } = StateHandler.getActionMask({
            request,
            maskTera,
        });
        state.setActionMask(actionMask.buffer);

        const {
            historyEntityPublic,
            historyEntityRevealed,
            historyEntityEdges,
            historyField,
            historyLength,
        } = this.getHistory(NUM_HISTORY);
        state.setHistoryEntityPublic(historyEntityPublic);
        state.setHistoryEntityRevealed(historyEntityRevealed);
        state.setHistoryEntityEdges(historyEntityEdges);
        state.setHistoryField(historyField);
        state.setHistoryLength(historyLength);

        const playerIndex = this.player.getPlayerIndex();
        if (playerIndex === undefined) {
            throw new Error("Player index is undefined");
        }

        const privateTeam = this.getPrivateTeam(playerIndex);
        state.setPrivateTeam(new Uint8Array(privateTeam.buffer));

        const { publicData, revealedData } = this.getPublicTeam(playerIndex);
        state.setPublicTeam(new Uint8Array(publicData.buffer));
        state.setRevealedTeam(new Uint8Array(revealedData.buffer));

        state.setMoveset(this.getMoveset());

        state.setField(this.getField());

        return state;
    }
}
