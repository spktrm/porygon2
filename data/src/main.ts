import * as fs from "fs";
import * as path from "path";

import { Dex, Format, RuleTable } from "@pkmn/sim";
import {
    Ability,
    Item,
    Move,
    Species,
    Type,
    toID,
    Dex as dexDex,
    TypeName,
} from "@pkmn/dex";
import { Generations } from "@pkmn/data";

const PS_DIRECTORY = "ps/";
const BATCH_SIZE = 128;
const PARENT_DATA_DIR = `data`;

const PAD_TOKEN = "_PAD";
const UNK_TOKEN = "_UNK";
const NULL_TOKEN = "_NULL";
const SWITCH_TOKEN = "_SWITCH";
const EXTRA_TOKENS = [NULL_TOKEN, PAD_TOKEN, UNK_TOKEN];

type CustomScrapingFunction = (content: string, file: string) => string[];

type Enums =
    | "pseudoWeather"
    | "status"
    | "volatileStatus"
    | "genderName"
    | "effectTypes"
    | "sideCondition"
    | "weather"
    | "terrain"
    | "battleMajorArgs"
    | "battleMinorArgs"
    | "boosts"
    | "itemEffectTypes"
    | "lastItemEffectTypes";

const customScrapingFunctions: {
    [key in Enums]: CustomScrapingFunction;
} = {
    pseudoWeather: (content: string, file: string): string[] => {
        const patterns = [
            /pseudoWeather:\s*['"](\w+)['"]/g,
            /addPseudoWeather\(['"](\w+)['"]/g,
            /removePseudoWeather\(['"](\w+)['"]/g,
            /field\.addPseudoWeather\(['"](\w+)['"]/g,
            /field\.removePseudoWeather\(['"](\w+)['"]/g,
            /field\.hasPseudoWeather\(['"](\w+)['"]/g,
            /field\.getPseudoWeather\(['"](\w+)['"]/g,
            /this\.field\.addPseudoWeather\(['"](\w+)['"]/g,
            /this\.field\.removePseudoWeather\(['"](\w+)['"]/g,
            /this\.field\.hasPseudoWeather\(['"](\w+)['"]/g,
            /this\.field\.getPseudoWeather\(['"](\w+)['"]/g,
        ];
        return applyPatterns(content, patterns);
    },
    status: (content: string, file: string): string[] => {
        const patterns = [
            /status:\s*['"](\w+)['"]/g,
            /\bstatusData\s*\[\s*['"](\w+)['"]\s*\]/g,
            /\bcure(?:Target|Ally)?Status\s*\(\s*['"](\w+)['"]/g,
            /\bset(?:Status|Condition)\s*\(\s*['"](\w+)['"]/g,
            /\btry(?:Set|Add)Status\s*\(\s*['"](\w+)['"]/g,
            /StatusCondition\.(\w+)/g,
            /status\.id\s*===?\s*['"](\w+)['"]/g,
            /status\.name\s*===?\s*['"](\w+)['"]/g,
            /status\.effectType\s*===?\s*['"](\w+)['"]/g,
            /pokemon\.hasStatus\(['"](\w+)['"]/g,
            /pokemon\.setStatus\(['"](\w+)['"]/g,
            /pokemon\.cureStatus\(['"](\w+)['"]/g,
            /this\.hasStatus\(['"](\w+)['"]/g,
            /this\.setStatus\(['"](\w+)['"]/g,
            /this\.cureStatus\(['"](\w+)['"]/g,
        ];
        return applyPatterns(content, patterns);
    },
    volatileStatus: (content: string, file: string): string[] => {
        const patterns = [
            /volatileStatus:\s*['"](\w+)['"]/g,
            /addVolatile\(['"](\w+)['"]/g,
            /removeVolatile\(['"](\w+)['"]/g,
            /pokemon\.addVolatile\(['"](\w+)['"]/g,
            /pokemon\.removeVolatile\(['"](\w+)['"]/g,
            /pokemon\.hasVolatile\(['"](\w+)['"]/g,
            /pokemon\.getVolatile\(['"](\w+)['"]/g,
            /this\.addVolatile\(['"](\w+)['"]/g,
            /this\.removeVolatile\(['"](\w+)['"]/g,
            /this\.hasVolatile\(['"](\w+)['"]/g,
            /this\.getVolatile\(['"](\w+)['"]/g,
            /volatileStatuses\[['"](\w+)['"]\]/g,
            /volatileStatusData\[['"](\w+)['"]\]/g,
        ];
        const matchedVolatiles = applyPatterns(content, patterns);

        // Look for the VOLATILES constant
        const volatileMatch = content.match(
            /const\s+VOLATILES\s*=\s*\[([\s\S]*?)\]/,
        );
        if (volatileMatch) {
            const volatileContent = volatileMatch[1];
            const volatileItems = volatileContent.match(/['"](\w+)['"]/g);
            if (volatileItems) {
                matchedVolatiles.push(
                    ...volatileItems.map((item) => item.replace(/['"]/g, "")),
                );
            }
        }

        // Look for the STARTABLE constant
        const startableMatch = content.match(
            /const\s+STARTABLE\s*=\s*new\s+Set\(\[\s*([\s\S]*?)\s*\]\)/,
        );
        if (startableMatch) {
            const startableContent = startableMatch[1];
            const startableItems = startableContent.match(/['"](\w+)['"]/g);
            if (startableItems) {
                matchedVolatiles.push(
                    ...startableItems.map((item) => item.replace(/['"]/g, "")),
                );
            }
        }

        return [...new Set(matchedVolatiles)]; // Remove duplicates
    },
    genderName: (content: string, file: string): string[] => {
        const match = content.match(
            /export\s+type\s+GenderName\s*=\s*([\s\S]*?);/,
        );
        if (match) {
            const genderContent = match[1];
            const genders = genderContent.match(/['"](\w*)['"]/g);
            if (genders) {
                return genders.map((gender) => gender.replace(/['"]/g, ""));
            }
        }
        return [];
    },
    effectTypes: (content: string, file: string): string[] => {
        const match = content.match(
            /export\s+type\s+EffectType\s*=\s*([\s\S]*?);/,
        );
        if (match) {
            const effectTypeContent = match[1];
            const effectTypes = effectTypeContent.match(/['"](\w+)['"]/g);
            if (effectTypes) {
                return effectTypes.map((effectType) =>
                    effectType.replace(/['"]/g, ""),
                );
            }
        }
        return [];
    },
    sideCondition: (content: string, file: string): string[] => {
        const patterns = [
            /sideCondition:\s*['"](\w+)['"]/g,
            /addSideCondition\(['"](\w+)['"]/g,
            /removeSideCondition\(['"](\w+)['"]/g,
            /field\.addSideCondition\(['"](\w+)['"]/g,
            /field\.removeSideCondition\(['"](\w+)['"]/g,
            /field\.hasSideCondition\(['"](\w+)['"]/g,
            /field\.getSideCondition\(['"](\w+)['"]/g,
            /this\.field\.addSideCondition\(['"](\w+)['"]/g,
            /this\.field\.removeSideCondition\(['"](\w+)['"]/g,
            /this\.field\.hasSideCondition\(['"](\w+)['"]/g,
            /this\.field\.getSideCondition\(['"](\w+)['"]/g,
            /sideConditions\[['"](\w+)['"]\]/g,
        ];
        return applyPatterns(content, patterns);
    },
    weather: (content: string, file: string): string[] => {
        const patterns = [
            /weather:\s*['"](\w+)['"]/g,
            /setWeather\(['"](\w+)['"]/g,
            /clearWeather\(['"](\w+)['"]/g,
            /battle\.setWeather\(['"](\w+)['"]/g,
            /battle\.clearWeather\(['"](\w+)['"]/g,
            /battle\.weather\.id\s*===?\s*['"](\w+)['"]/g,
            /this\.setWeather\(['"](\w+)['"]/g,
            /this\.clearWeather\(['"](\w+)['"]/g,
            /this\.weather\.id\s*===?\s*['"](\w+)['"]/g,
            /field\.setWeather\(['"](\w+)['"]/g,
            /field\.clearWeather\(['"](\w+)['"]/g,
            /field\.weather\.id\s*===?\s*['"](\w+)['"]/g,
        ];
        return applyPatterns(content, patterns);
    },
    terrain: (content: string, file: string): string[] => {
        const patterns = [
            /terrain:\s*['"](\w+)['"]/g,
            /setTerrain\(['"](\w+)['"]/g,
            /clearTerrain\(['"](\w+)['"]/g,
            /battle\.setTerrain\(['"](\w+)['"]/g,
            /battle\.clearTerrain\(['"](\w+)['"]/g,
            /battle\.terrain\.id\s*===?\s*['"](\w+)['"]/g,
            /this\.setTerrain\(['"](\w+)['"]/g,
            /this\.clearTerrain\(['"](\w+)['"]/g,
            /this\.terrain\.id\s*===?\s*['"](\w+)['"]/g,
            /field\.setTerrain\(['"](\w+)['"]/g,
            /field\.clearTerrain\(['"](\w+)['"]/g,
            /field\.terrain\.id\s*===?\s*['"](\w+)['"]/g,
        ];
        const terrainMap: { [key: string]: string } = {
            electric: "electricterrain",
            grassy: "grassyterrain",
            psychic: "psychicterrain",
            misty: "mistyterrain",
            electricterrain: "electricterrain",
            mistyterrain: "mistyterrain",
            grassyterrain: "grassyterrain",
            psychicterrain: "psychicterrain",
        };
        const terrains = applyPatterns(content, patterns);
        return terrains.map((t) => terrainMap[t.toLowerCase()] || t);
    },
    battleMajorArgs: (content: string, file: string): string[] => {
        if (content.includes("interface BattleMajorArgs")) {
            const interfaceMatch = content.match(
                /interface\s+BattleMajorArgs\s*{[\s\S]*?}/,
            );
            if (interfaceMatch) {
                const interfaceContent = interfaceMatch[0];
                const argMatches =
                    interfaceContent.match(/['"]?\|(\w+)\|['"]?:/g) || [];
                return [
                    "turn",
                    ...argMatches.map((match) =>
                        match
                            .replace(/['"]?\|(\w+)\|['"]?:/, "$1")
                            .toLowerCase(),
                    ),
                ];
            }
        }
        return [];
    },
    battleMinorArgs: (content: string, file: string): string[] => {
        if (content.includes("interface BattleMinorArgs")) {
            const interfaceMatch = content.match(
                /interface\s+BattleMinorArgs\s*{[\s\S]*?}/,
            );
            if (interfaceMatch) {
                const interfaceContent = interfaceMatch[0];
                const argMatches =
                    interfaceContent.match(/['"]?\|-?(\w+)\|['"]?:/g) || [];
                return argMatches.map((match) =>
                    match.replace(/['"]?\|-?(\w+)\|['"]?:/, "$1").toLowerCase(),
                );
            }
        }
        return [];
    },
    boosts: (content: string, file: string): string[] => {
        const boostMatch = content.match(
            /const\s+BOOSTS\s*:\s*BoostID\[\]\s*=\s*\[([\s\S]*?)\]/,
        );
        if (boostMatch) {
            const boostContent = boostMatch[1];
            const boostItems = boostContent.match(/['"](\w+)['"]/g);
            if (boostItems) {
                return boostItems.map((item) => item.replace(/['"]/g, ""));
            }
        }
        return [];
    },
    itemEffectTypes: (content: string, file: string): string[] => {
        const patterns = [
            /export\s+type\s+ItemEffect\s*=\s*([\s\S]*?);/g,
            /item\.effectType\s*===?\s*['"]([\w\s]+)['"]/g,
            /item\.hasEffect\(['"]([\w\s]+)['"]\)/g,
            /item\.addEffect\(['"]([\w\s]+)['"]\)/g,
            /item\.removeEffect\(['"]([\w\s]+)['"]\)/g,
            /this\.addItemEffect\(['"]([\w\s]+)['"]\)/g,
            /this\.removeItemEffect\(['"]([\w\s]+)['"]\)/g,
            /item\.effects\[['"]([\w\s]+)['"]\]/g,
            /itemEffect\(['"]([\w\s]+)['"]\)/g,
            /['"]effectType['"]\s*:\s*['"]([\w\s]+)['"]/g,
            /['"]on\w+['"]\s*:\s*['"]([\w\s]+)['"]/g,
            /['"]([\w\s]+)['"]\s*:\s*function\s*\(/g,
            /poke\.itemEffect\s*=\s*['"]([\w\s]+)['"]/g, // Added pattern
            /this\.itemEffect\s*=\s*['"]([\w\s]+)['"]/g, // Added pattern
        ];

        const effects: string[] = [];

        // Match the export type ItemEffect
        const typeMatch = content.match(
            /export\s+type\s+ItemEffect\s*=\s*([\s\S]*?);/,
        );
        if (typeMatch) {
            const effectsContent = typeMatch[1];
            const typeEffects = effectsContent.match(/['"]([\w\s]+)['"]/g);
            if (typeEffects) {
                effects.push(
                    ...typeEffects.map((effect) => effect.replace(/['"]/g, "")),
                );
            }
        }

        // Apply other patterns
        const otherEffects = applyPatterns(content, patterns.slice(1));
        effects.push(...otherEffects);

        return effects;
    },
    lastItemEffectTypes: (content: string, file: string): string[] => {
        const patterns = [
            /export\s+type\s+LastItemEffect\s*=\s*([\s\S]*?);/g,
            /poke\.lastItemEffect\s*=\s*['"]([\w\s]+)['"]/g, // Added pattern
            /this\.lastItemEffect\s*=\s*['"]([\w\s]+)['"]/g, // Added pattern
            /['"]lastItemEffect['"]\s*:\s*['"]([\w\s]+)['"]/g,
            /item\.lastEffect\s*=\s*['"]([\w\s]+)['"]/g,
            /item\.lastItemEffect\s*=\s*['"]([\w\s]+)['"]/g,
            /item\.lastEffectType\s*=\s*['"]([\w\s]+)['"]/g,
        ];

        const effects: string[] = [];

        // Match the export type LastItemEffect
        const typeMatch = content.match(
            /export\s+type\s+LastItemEffect\s*=\s*([\s\S]*?);/,
        );
        if (typeMatch) {
            const effectsContent = typeMatch[1];
            const typeEffects = effectsContent.match(/['"]([\w\s]+)['"]/g);
            if (typeEffects) {
                effects.push(
                    ...typeEffects.map((effect) => effect.replace(/['"]/g, "")),
                );
            }
        }

        // Apply other patterns
        const otherEffects = applyPatterns(content, patterns.slice(1));
        effects.push(...otherEffects);

        return effects;
    },
};

function applyPatterns(content: string, patterns: RegExp[]): string[] {
    const matches: string[] = [];
    for (const pattern of patterns) {
        const found = [...content.matchAll(pattern)].map((match) =>
            toID(match[1]),
        );
        matches.push(...found);
    }
    return matches;
}

function getFiles(dir: string): string[] {
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    const files = entries
        .filter((file) => !file.isDirectory())
        .filter((file) => {
            const fileName = file.name.toLowerCase();
            // Exclude test files
            return (
                !fileName.includes(".test.") &&
                !fileName.includes(".spec.") &&
                !fileName.endsWith(".test") &&
                !fileName.endsWith(".spec")
            );
        })
        .map((file) => path.join(dir, file.name));

    const folders = entries
        .filter((folder) => folder.isDirectory())
        .filter((folder) => {
            const folderName = folder.name.toLowerCase();
            // Exclude test directories
            return (
                folderName !== "test" &&
                folderName !== "__tests__" &&
                !folderName.endsWith(".test") &&
                !folderName.endsWith(".spec")
            );
        });

    for (const folder of folders) {
        files.push(...getFiles(path.join(dir, folder.name)));
    }

    return files;
}

async function processInBatches<T>(
    items: T[],
    batchSize: number,
    processFn: (item: T) => Promise<void>,
) {
    for (let i = 0; i < items.length; i += batchSize) {
        const batch = items.slice(i, i + batchSize);
        console.log(
            `Processing batch ${i / batchSize + 1}/${Math.ceil(
                items.length / batchSize,
            )}`,
        );
        await Promise.all(batch.map(processFn));
    }
}

type GenData = {
    species: Species[];
    moves: Move[];
    abilities: Ability[];
    items: Item[];
    typechart: Type[];
    //     learnsets: {
    //         species: SpeciesName;
    //         [k: string]: any;
    //     }[];
};

async function getGenData(genNo: number): Promise<GenData> {
    const dex = new Dex.ModdedDex(`gen${genNo}`);
    const gens = new Generations(dexDex).get(genNo);
    const species = dex.species.all();
    const promises = species.map((species: { id: string }) =>
        dex.learnsets.get(species.id),
    );
    const abilities = dex.abilities.all();
    const learnsets = await Promise.all(promises);
    const moves = dex.moves.all();
    const items = dex.items.all();
    const typechart = dex.types
        .all()
        .filter((type) => type.isNonstandard === null);

    const data = {
        species: species.map((x) => {
            const effectiveness = Object.fromEntries(
                typechart.map((type) => [
                    type.name,
                    gens.types.totalEffectiveness(
                        type.name as TypeName,
                        x.types as TypeName[],
                    ),
                ]),
            );
            return {
                ...x,
                effectiveness,
            };
        }),
        moves,
        abilities,
        items,
        typechart,
        learnsets,
    };

    return data as unknown as GenData;
}

function formatData(data: GenData) {
    const moveIds = [
        ...data.moves.map((item) => {
            let name = toID(item.name);
            if (name === "return") {
                return `${item.id}102`;
            } else if (
                name.startsWith("hiddenpower") &&
                !name.endsWith("hiddenpower")
            ) {
                name += `${item.basePower}`;
                return name;
            } else {
                return name;
            }
        }),
        "return",
    ];
    moveIds
        .filter((x) => x.startsWith("hiddenpower") && x !== "hiddenpower")
        .map((x) => {
            moveIds.push(x.slice(0, -2));
            moveIds.push(x.slice(0, -2) + `70`);
        });

    const getId = (item: any) => {
        return toID(item.id);
    };

    return {
        species: data.species.map(getId),
        moves: [SWITCH_TOKEN, ...moveIds, "recharge"],
        abilities: data.abilities.map(getId),
        items: data.items.map(getId),
        typechart: data.typechart.map(getId),
    };
}

function standardize(values: string[], extraTokens?: string[]) {
    return Object.fromEntries(
        [
            ...(extraTokens ?? []),
            ...Array.from(values).sort((a, b) => a.localeCompare(b)),
        ].map((value, index) => [value, index]),
    );
}

async function scrapeRepo() {
    const allFiles = getFiles(PS_DIRECTORY);
    const jsAndTsFiles = allFiles.filter(
        (file) => file.endsWith(".js") || file.endsWith(".ts"),
    );

    console.log(
        `Found ${jsAndTsFiles.length} JavaScript/TypeScript files to process.`,
    );

    const keywords: { [key: string]: Set<string> } = {};
    for (const category of Object.keys(customScrapingFunctions)) {
        keywords[category as Enums] = new Set<string>();
    }

    await processInBatches(jsAndTsFiles, BATCH_SIZE, async (file) => {
        const content = fs.readFileSync(file, "utf-8");

        // Process all custom scraping functions
        for (const [category, scrapeFn] of Object.entries(
            customScrapingFunctions,
        )) {
            const extracted = scrapeFn(content, file);
            extracted.forEach((value: string) => {
                const formattedValue = toID(value);
                if (formattedValue) {
                    keywords[category].add(formattedValue);
                }
            });
        }
    });

    const genData = await getGenData(9);
    const allFormats = Dex.formats.all().map((format) => ({
        format,
        formatId: format.id,
        ruletable: Dex.formats.getRuleTable(format),
    }));
    const data: { [k: string]: { [k: string]: number } } = {};

    const genFormatData = formatData(genData);
    for (const [category, values] of Object.entries({
        ...genFormatData,
        ...keywords,
    })) {
        data[category] = standardize(Array.from(values), EXTRA_TOKENS);
    }
    const conditions = [
        ...keywords.sideCondition,
        ...keywords.volatileStatus,
        "recoil",
        "drain",
    ];
    data["Condition"] = standardize(conditions, EXTRA_TOKENS);

    data["Effect"] = standardize(
        [
            ...genData.abilities.map((x) => `ability_${x.id}`),
            ...genData.items.map((x) => `item_${x.id}`),
            ...genData.moves.map((x) => `move_${x.id}`),
            ...[...keywords.status].map((x) => `status_${x}`),
            ...[...keywords.weather].map((x) => `weather_${x}`),
            ...conditions.map((x) => `condition_${x}`),
        ],
        EXTRA_TOKENS,
    );

    data["Actions"] = standardize([
        ...[...genFormatData.species, ...EXTRA_TOKENS].map(
            (x) => `switch_${x}`,
        ),
        ...[...genFormatData.moves, ...EXTRA_TOKENS].map((x) => `move_${x}`),
    ]);

    fs.writeFileSync(
        `${PARENT_DATA_DIR}/data.json`,
        JSON.stringify(data, null, 2),
    );

    const datas = [];
    const parentDirs = [];

    for (const genNo of [1, 2, 3, 4, 5, 6, 7, 8, 9]) {
        const parentDir = `${PARENT_DATA_DIR}/gen${genNo}/`;

        if (!fs.existsSync(parentDir)) {
            fs.mkdirSync(parentDir, { recursive: true });
        }

        const genDataPromise = getGenData(genNo);

        datas.push(genDataPromise);
        parentDirs.push(parentDir);
    }

    const results = await Promise.all(datas);

    for (const [index, result] of results.entries()) {
        for (const [key, values] of Object.entries(result)) {
            const ourPath = `${parentDirs.at(index) ?? ""}/${key}.json`;
            console.log(`writing ${ourPath}`);
            fs.writeFileSync(ourPath, JSON.stringify(values, null, 2));
        }
    }
}

scrapeRepo().catch((err) => console.error("Error during scraping:", err));
