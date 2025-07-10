// import {
//     Dex,
//     Format,
//     ModdedDex,
//     PRNG,
//     PRNGSeed,
//     PlayerOptions,
//     PokemonSet,
//     RandomTeamsTypes,
//     Species,
//     toID,
// } from "@pkmn/sim";
// import { Teams } from "@pkmn/sim";

// class TeamGenerator {
//     format: Format;
//     readonly dex: ModdedDex;
//     gen: number;
//     prng: PRNG;
//     readonly maxSize: number;
//     readonly maxMoveCount: number;

//     constructor(formatId: string, prng: PRNG | PRNGSeed | null) {
//         this.format = Dex.formats.get(formatId);
//         this.dex = Dex.mod(this.format.mod);
//         this.gen = this.dex.gen;
//         this.prng = PRNG.get(prng);
//         this.maxSize = 6;
//         this.maxMoveCount = 4;
//     }

//     randomChance(numerator: number, denominator: number) {
//         return this.prng.randomChance(numerator, denominator);
//     }

//     sample<T>(items: readonly T[]): T {
//         return this.prng.sample(items);
//     }

//     random(m?: number, n?: number) {
//         return this.prng.random(m, n);
//     }

//     getPokemonPool(
//         type: string,
//         pokemonToExclude: PokemonSet[] = [],
//         isMonotype = false,
//         pokemonList: string[],
//     ): [{ [k: string]: string[] }, string[]] {
//         const exclude = pokemonToExclude.map((p) => toID(p.species));
//         const pokemonPool: { [k: string]: string[] } = {};
//         const baseSpeciesPool = [];
//         for (const pokemon of pokemonList) {
//             let species = this.dex.species.get(pokemon);
//             if (exclude.includes(species.id)) continue;
//             if (isMonotype) {
//                 if (!species.types.includes(type)) continue;
//                 if (typeof species.battleOnly === "string") {
//                     species = this.dex.species.get(species.battleOnly);
//                     if (!species.types.includes(type)) continue;
//                 }
//             }

//             if (species.baseSpecies in pokemonPool) {
//                 pokemonPool[species.baseSpecies].push(pokemon);
//             } else {
//                 pokemonPool[species.baseSpecies] = [pokemon];
//             }
//         }
//         // Include base species 1x if 1-3 formes, 2x if 4-6 formes, 3x if 7+ formes
//         for (const baseSpecies of Object.keys(pokemonPool)) {
//             // Squawkabilly has 4 formes, but only 2 functionally different formes, so only include it 1x
//             const weight =
//                 baseSpecies === "Squawkabilly"
//                     ? 1
//                     : Math.min(
//                           Math.ceil(pokemonPool[baseSpecies].length / 3),
//                           3,
//                       );
//             for (let i = 0; i < weight; i++) baseSpeciesPool.push(baseSpecies);
//         }
//         return [pokemonPool, baseSpeciesPool];
//     }

//     fastPop(list: unknown[], index: number) {
//         // If an array doesn't need to be in order, replacing the
//         // element at the given index with the removed element
//         // is much, much faster than using list.splice(index, 1).
//         const length = list.length;
//         if (index < 0 || index >= list.length) {
//             // sanity check
//             throw new Error(`Index ${index} out of bounds for given array`);
//         }

//         const element = list[index];
//         list[index] = list[length - 1];
//         list.pop();
//         return element;
//     }

//     sampleNoReplace(list: unknown[]) {
//         const length = list.length;
//         if (length === 0) return null;
//         const index = this.random(length);
//         return this.fastPop(list, index);
//     }

//     getForme(species: Species): string {
//         if (typeof species.battleOnly === "string") {
//             // Only change the forme. The species has custom moves, and may have different typing and requirements.
//             return species.battleOnly;
//         }
//         if (species.cosmeticFormes)
//             return this.sample([species.name].concat(species.cosmeticFormes));
//         if (species.name.endsWith("-Gmax")) return species.name.slice(0, -5);

//         // Consolidate mostly-cosmetic formes, at least for the purposes of Random Battles
//         if (
//             ["Magearna", "Polteageist", "Zarude"].includes(species.baseSpecies)
//         ) {
//             return this.sample([species.name].concat(species.otherFormes!));
//         }
//         if (species.baseSpecies === "Basculin")
//             return "Basculin" + this.sample(["", "-Blue-Striped"]);
//         if (species.baseSpecies === "Keldeo" && this.gen <= 7)
//             return "Keldeo" + this.sample(["", "-Resolute"]);
//         if (
//             species.baseSpecies === "Pikachu" &&
//             this.dex.currentMod === "gen8"
//         ) {
//             return (
//                 "Pikachu" +
//                 this.sample([
//                     "",
//                     "-Original",
//                     "-Hoenn",
//                     "-Sinnoh",
//                     "-Unova",
//                     "-Kalos",
//                     "-Alola",
//                     "-Partner",
//                     "-World",
//                 ])
//             );
//         }
//         return species.name;
//     }

//     randomSet(
//         species: string | Species,
//         teamDetails: RandomTeamsTypes.TeamDetails = {},
//         isLead = false,
//         isDoubles = false,
//         isNoDynamax = false,
//     ): RandomTeamsTypes.RandomSet {
//         species = this.dex.species.get(species);
//         const forme = this.getForme(species);
//         const gmax = species.name.endsWith("-Gmax");

//         const data = this.randomData[species.id];

//         const randMoves =
//             (isDoubles && data.doublesMoves) ||
//             (isNoDynamax && data.noDynamaxMoves) ||
//             data.moves;
//         const movePool: string[] = [
//             ...(randMoves || this.dex.species.getMovePool(species.id)),
//         ];

//         const rejectedPool = [];
//         let ability = "";
//         let item = undefined;

//         const evs = { hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85 };
//         const ivs = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };

//         const types = new Set(species.types);
//         const abilitiesSet = new Set(Object.values(species.abilities));
//         if (species.unreleasedHidden) abilitiesSet.delete(species.abilities.H);
//         const abilities = Array.from(abilitiesSet);

//         const moves = new Set<string>();
//         let counter: MoveCounter;
//         // This is just for BDSP Unown;
//         // it can be removed from this file if BDSP gets its own random-teams file in the future.
//         let hasHiddenPower = false;

//         do {
//             // Choose next 4 moves from learnset/viable moves and add them to moves list:
//             const pool = movePool.length ? movePool : rejectedPool;
//             while (moves.size < this.maxMoveCount && pool.length) {
//                 const moveid = this.sampleNoReplace(pool);
//                 if (moveid.startsWith("hiddenpower")) {
//                     if (hasHiddenPower) continue;
//                     hasHiddenPower = true;
//                 }
//                 moves.add(moveid);
//             }

//             counter = this.queryMoves(
//                 moves,
//                 species.types,
//                 abilities,
//                 movePool,
//             );
//             const runEnforcementChecker = (checkerName: string) => {
//                 if (!this.moveEnforcementCheckers[checkerName]) return false;
//                 return this.moveEnforcementCheckers[checkerName](
//                     movePool,
//                     moves,
//                     abilities,
//                     types,
//                     counter,
//                     species,
//                     teamDetails,
//                 );
//             };

//             // Iterate through the moves again, this time to cull them:
//             for (const moveid of moves) {
//                 const move = this.dex.moves.get(moveid);
//                 let { cull, isSetup } = this.shouldCullMove(
//                     move,
//                     types,
//                     moves,
//                     abilities,
//                     counter,
//                     movePool,
//                     teamDetails,
//                     species,
//                     isLead,
//                     isDoubles,
//                     isNoDynamax,
//                 );

//                 if (
//                     move.id !== "photongeyser" &&
//                     ((move.category === "Physical" &&
//                         counter.setupType === "Special") ||
//                         (move.category === "Special" &&
//                             counter.setupType === "Physical"))
//                 ) {
//                     // Reject STABs last in case the setup type changes later on
//                     const stabs =
//                         counter.get(species.types[0]) +
//                         (species.types[1] ? counter.get(species.types[1]) : 0);
//                     if (
//                         !types.has(move.type) ||
//                         stabs > 1 ||
//                         counter.get(move.category) < 2
//                     )
//                         cull = true;
//                 }

//                 // Pokemon should have moves that benefit their types, stats, or ability
//                 const isLowBP = move.basePower && move.basePower < 50;

//                 // Genesect-Douse should never reject Techno Blast
//                 const moveIsRejectable =
//                     !(
//                         species.id === "genesectdouse" &&
//                         move.id === "technoblast"
//                     ) &&
//                     !(species.id === "togekiss" && move.id === "nastyplot") &&
//                     !(
//                         species.id === "shuckle" &&
//                         ["stealthrock", "stickyweb"].includes(move.id)
//                     ) &&
//                     (move.category === "Status" ||
//                         (!types.has(move.type) && move.id !== "judgment") ||
//                         (isLowBP &&
//                             !move.multihit &&
//                             !abilities.includes("Technician")));
//                 // Setup-supported moves should only be rejected under specific circumstances
//                 const notImportantSetup =
//                     !counter.setupType ||
//                     counter.setupType === "Mixed" ||
//                     (counter.get(counter.setupType) + counter.get("Status") >
//                         3 &&
//                         !counter.get("hazards")) ||
//                     (move.category !== counter.setupType &&
//                         move.category !== "Status");

//                 if (
//                     moveIsRejectable &&
//                     !cull &&
//                     !isSetup &&
//                     !move.weather &&
//                     !move.stallingMove &&
//                     notImportantSetup &&
//                     !move.damage &&
//                     (isDoubles
//                         ? this.unrejectableMovesInDoubles(move)
//                         : this.unrejectableMovesInSingles(move))
//                 ) {
//                     // There may be more important moves that this Pokemon needs
//                     if (
//                         // Pokemon should have at least one STAB move
//                         (!counter.get("stab") &&
//                             counter.get("physicalpool") +
//                                 counter.get("specialpool") >
//                                 0 &&
//                             move.id !== "stickyweb") ||
//                         // Swords Dance Mew should have Brave Bird
//                         (moves.has("swordsdance") &&
//                             species.id === "mew" &&
//                             runEnforcementChecker("Flying")) ||
//                         // Dhelmise should have Anchor Shot
//                         (abilities.includes("Steelworker") &&
//                             runEnforcementChecker("Steel")) ||
//                         // Check for miscellaneous important moves
//                         (!isDoubles &&
//                             runEnforcementChecker("recovery") &&
//                             move.id !== "stickyweb") ||
//                         runEnforcementChecker("screens") ||
//                         runEnforcementChecker("misc") ||
//                         ((isLead || species.id === "shuckle") &&
//                             runEnforcementChecker("lead")) ||
//                         (moves.has("leechseed") &&
//                             runEnforcementChecker("leechseed"))
//                     ) {
//                         cull = true;
//                         // Pokemon should have moves that benefit their typing
//                         // Don't cull Sticky Web in type-based enforcement, and make sure Azumarill always has Aqua Jet
//                     } else if (
//                         move.id !== "stickyweb" &&
//                         !(species.id === "azumarill" && move.id === "aquajet")
//                     ) {
//                         for (const type of types) {
//                             if (runEnforcementChecker(type)) {
//                                 cull = true;
//                             }
//                         }
//                     }
//                 }

//                 // Sleep Talk shouldn't be selected without Rest
//                 if (move.id === "rest" && cull) {
//                     const sleeptalk = movePool.indexOf("sleeptalk");
//                     if (sleeptalk >= 0) {
//                         if (movePool.length < 2) {
//                             cull = false;
//                         } else {
//                             this.fastPop(movePool, sleeptalk);
//                         }
//                     }
//                 }

//                 // Remove rejected moves from the move list
//                 if (cull && movePool.length) {
//                     if (moveid.startsWith("hiddenpower"))
//                         hasHiddenPower = false;
//                     if (move.category !== "Status" && !move.damage)
//                         rejectedPool.push(moveid);
//                     moves.delete(moveid);
//                     break;
//                 }
//                 if (cull && rejectedPool.length) {
//                     if (moveid.startsWith("hiddenpower"))
//                         hasHiddenPower = false;
//                     moves.delete(moveid);
//                     break;
//                 }
//             }
//         } while (
//             moves.size < this.maxMoveCount &&
//             (movePool.length || rejectedPool.length)
//         );

//         ability = this.getAbility(
//             types,
//             moves,
//             abilities,
//             counter,
//             movePool,
//             teamDetails,
//             species,
//             "",
//             "",
//             isDoubles,
//             isNoDynamax,
//         );

//         if (species.requiredItems) {
//             item = this.sample(species.requiredItems);
//             // First, the extra high-priority items
//         } else {
//             item = this.getHighPriorityItem(
//                 ability,
//                 types,
//                 moves,
//                 counter,
//                 teamDetails,
//                 species,
//                 isLead,
//                 isDoubles,
//             );
//             if (item === undefined && isDoubles) {
//                 item = this.getDoublesItem(
//                     ability,
//                     types,
//                     moves,
//                     abilities,
//                     counter,
//                     teamDetails,
//                     species,
//                 );
//             }
//             if (item === undefined) {
//                 item = this.getMediumPriorityItem(
//                     ability,
//                     moves,
//                     counter,
//                     species,
//                     isLead,
//                     isDoubles,
//                     isNoDynamax,
//                 );
//             }
//             if (item === undefined) {
//                 item = this.getLowPriorityItem(
//                     ability,
//                     types,
//                     moves,
//                     abilities,
//                     counter,
//                     teamDetails,
//                     species,
//                     isLead,
//                     isDoubles,
//                     isNoDynamax,
//                 );
//             }

//             // fallback
//             if (item === undefined)
//                 item = isDoubles ? "Sitrus Berry" : "Leftovers";
//         }

//         // For Trick / Switcheroo
//         if (item === "Leftovers" && types.has("Poison")) {
//             item = "Black Sludge";
//         }

//         const level: number = this.getLevel(species, isDoubles, isNoDynamax);

//         // Prepare optimal HP
//         const srImmunity =
//             ability === "Magic Guard" || item === "Heavy-Duty Boots";
//         const srWeakness = srImmunity
//             ? 0
//             : this.dex.getEffectiveness("Rock", species);
//         while (evs.hp > 1) {
//             const hp = Math.floor(
//                 (Math.floor(
//                     2 * species.baseStats.hp +
//                         ivs.hp +
//                         Math.floor(evs.hp / 4) +
//                         100,
//                 ) *
//                     level) /
//                     100 +
//                     10,
//             );
//             const multipleOfFourNecessary =
//                 moves.has("substitute") &&
//                 !["Leftovers", "Black Sludge"].includes(item) &&
//                 (item === "Sitrus Berry" ||
//                     item === "Salac Berry" ||
//                     ability === "Power Construct");
//             if (multipleOfFourNecessary) {
//                 // Two Substitutes should activate Sitrus Berry
//                 if (hp % 4 === 0) break;
//             } else if (
//                 moves.has("bellydrum") &&
//                 (item === "Sitrus Berry" || ability === "Gluttony")
//             ) {
//                 // Belly Drum should activate Sitrus Berry
//                 if (hp % 2 === 0) break;
//             } else if (moves.has("substitute") && moves.has("reversal")) {
//                 // Reversal users should be able to use four Substitutes
//                 if (hp % 4 > 0) break;
//             } else {
//                 // Maximize number of Stealth Rock switch-ins
//                 if (srWeakness <= 0 || hp % (4 / srWeakness) > 0) break;
//             }
//             evs.hp -= 4;
//         }

//         if (moves.has("shellsidearm") && item === "Choice Specs") evs.atk -= 8;

//         // Minimize confusion damage
//         const noAttackStatMoves = [...moves].every((m) => {
//             const move = this.dex.moves.get(m);
//             if (move.damageCallback || move.damage) return true;
//             return move.category !== "Physical" || move.id === "bodypress";
//         });
//         if (
//             noAttackStatMoves &&
//             !moves.has("transform") &&
//             (!moves.has("shellsidearm") || !counter.get("Status"))
//         ) {
//             evs.atk = 0;
//             ivs.atk = 0;
//         }

//         // Ensure Nihilego's Beast Boost gives it Special Attack boosts instead of Special Defense
//         if (forme === "Nihilego") evs.spd -= 32;

//         if (moves.has("gyroball") || moves.has("trickroom")) {
//             evs.spe = 0;
//             ivs.spe = 0;
//         }

//         return {
//             name: species.baseSpecies,
//             species: forme,
//             gender: species.gender,
//             shiny: this.randomChance(1, 1024),
//             gigantamax: gmax,
//             level,
//             moves: Array.from(moves),
//             ability,
//             evs,
//             ivs,
//             item,
//         };
//     }

//     getTeam(options?: PlayerOptions | null): PokemonSet[] {
//         const typePool = this.dex.types.names();
//         const type = this.sample(typePool);

//         const pokemon: PokemonSet[] = [];
//         const pokemonList: string[] = [];

//         const baseFormes: { [k: string]: number } = {};
//         const teamDetails: RandomTeamsTypes.TeamDetails = {};

//         const [pokemonPool, baseSpeciesPool] = this.getPokemonPool(
//             type,
//             pokemon,
//             false,
//             pokemonList,
//         );
//         while (baseSpeciesPool.length < this.maxSize) {
//             const baseSpecies = this.sampleNoReplace(baseSpeciesPool) as string;
//             const species = this.dex.species.get(
//                 this.sample(pokemonPool[baseSpecies]),
//             );
//             if (!species.exists) continue;

//             if (baseFormes[species.baseSpecies]) continue;

//             const set = this.randomSet(
//                 species,
//                 teamDetails,
//                 pokemon.length === 0,
//                 isDoubles,
//                 this.dex.formats.getRuleTable(this.format).has("dynamaxclause"),
//             );

//             // Okay, the set passes, add it to our team
//             pokemon.push(set);
//         }
//     }
// }

// const generator = new TeamGenerator("gen3ou");

// while (true) {
//     const team = generator.getTeam();
//     const packed = Teams.pack(generate);
//     console.log(team);
// }
