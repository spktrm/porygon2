import * as fs from "fs";
import {
    Ability,
    Dex,
    GenID,
    ID,
    Item,
    Move,
    Species,
    Type,
    toID,
} from "@pkmn/dex";
import { Generations } from "@pkmn/data";
import dotenv from "dotenv";
import { Protocol } from "@pkmn/protocol";

dotenv.config();
const accessToken = process.env.ACCESS_TOKEN;

const owner = "pkmn";
const repo = "ps";
const apiUrl = `https://api.github.com/repos/${owner}/${repo}/contents/`;

async function getFilenames(path: string): Promise<string[]> {
    const response = await fetch(`${apiUrl}${path}`, {
        headers: {
            Authorization: `token ${accessToken}`,
            Accept: "application/vnd.github.v3+json",
        },
    });

    if (!response.ok) {
        console.error(
            `GitHub API responded with ${response.status} for path ${path}`,
        );
        return [];
    }

    const data = await response.json();
    const paths = [];

    for (const item of data) {
        if (item.type === "file") {
            console.log(item.path);
            paths.push(item.download_url);
        } else if (item.type === "dir") {
            paths.push(...(await getFilenames(item.path)));
        }
    }
    return paths;
}

// Helper function to convert a string to an identifier
function toId(string: string): string {
    return string.toLowerCase().replace(/[^a-z0-9]/g, "");
}

// Helper function to reduce an array to unique, sorted identifiers
function reduce(arr: string[]): string[] {
    return Array.from(new Set(arr.map(toId).filter((x) => !!x))).sort();
}

// function findDuplicates(arr: string[]): string[] {
//     return arr.filter((item, index) => {
//         return arr.indexOf(item) !== index;
//     });
// }

// Helper function to create an enumeration from an array
function enumerate(arr: string[]): { [key: string]: number } {
    const enumeration: { [key: string]: number } = {};
    let count = 0;
    for (const item of arr.sort()) {
        if (!Object.keys(enumeration).includes(item)) {
            enumeration[item] = count;
        }
        count += 1;
    }
    return enumeration;
}

// Function to fetch text content from a URL
async function fetchText(url: string): Promise<string> {
    const response = await fetch(url);
    return response.text();
}

// Function to download all files and concatenate their content
async function downloadAll(): Promise<string[]> {
    const urls = [
        ...(await getFilenames("sim")),
        ...(await getFilenames("client")),
    ];
    const requests = urls.map((url) => fetchText(url));
    return Promise.all(requests);
}

// Function to extract unique strings from source text based on a regular expression
function extractPatterns(src: string, pattern: RegExp): string[] {
    return [...src.matchAll(pattern)].map((match) => match[1]);
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

async function getGenData(gen: number): Promise<GenData> {
    const format = `gen${gen}` as ID;
    const generations = new Generations(Dex);
    const dex = generations.dex.mod(format as GenID);
    const species = (dex.species as any).all();
    const promises = species.map((species: { id: string }) =>
        dex.learnsets.get(species.id),
    );
    const learnsets = await Promise.all(promises);
    const moves = (dex.moves as any).all();
    const data = {
        species: species,
        moves: moves,
        abilities: (dex.abilities as any).all(),
        items: (dex.items as any).all(),
        typechart: (dex.types as any).all(),
        learnsets: learnsets,
    };
    return data;
}

function mapId<T extends { id: string; [key: string]: any }>(
    arr: T[],
): string[] {
    return arr.map((item) => item.id);
}

const padToken = "!PAD!";
const unkToken = "!UNK!";
const nullToken = "!NULL!";
const noneToken = "!NONE!";
const switchToken = "!SWITCH!";
const extraTokens = [noneToken, unkToken, padToken];

// function formatKey(key: string): string {
//     return key.startsWith("<") ? key : key.toLowerCase().replace(/[\W_]+/g, "");
// }

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
    return {
        species: enumerate([...extraTokens, ...mapId(data.species)]),
        moves: enumerate([...extraTokens, switchToken, ...moveIds, "recharge"]),
        abilities: enumerate([...extraTokens, ...mapId(data.abilities)]),
        items: enumerate([...extraTokens, nullToken, ...mapId(data.items)]),
    };
}

// The main function that executes the download and processing
async function main(): Promise<void> {
    const sources = await downloadAll();
    const src = sources.join("\n");

    // Extract patterns for different categories
    const weathersPattern = /['|"]-weather['|"],\s*['|"](.*)['|"],/g;

    const terrainPattern = /terrain:\s*['|"](.*)['|"],/g;
    const pseudoWeatherPattern = /pseudoWeather:\s['|"](.*?)['|"]/g;

    const itemEffectPatterns = [
        /itemEffect = ['|"](.*?)['|"]/g,
        /lastItemEffect = ['|"](.*?)['|"]/g,
    ];

    // Define patterns for volatile status
    const volatileStatusPatterns = [
        /removeVolatile\('([^']+)'/g,
        /hasVolatile\('([^']+)'/g,
        /volatiles\??\.([a-zA-Z_]\w*)/g,
        /volatileStatus:\s*['|"](.*)['|'],/g,
        /this\.add\('-start',\s*[^,]+,\s*([^,)]+)/g,
        /addVolatile\('([^']+)'/g,
    ];

    // Use a Set to ensure uniqueness
    let volatileStatusSet = new Set<string>();

    // Process each pattern and add the results to the Set
    volatileStatusPatterns.forEach((pattern) => {
        const matches = src.match(pattern);
        if (matches) {
            matches.forEach((match) => {
                match = match.replace(pattern, "$1"); // Extract the captured group
                match = match
                    .replace(/['|"|\[|\]|\(|\)|,]/g, "") // Clean up any extra characters
                    .trim();
                match = match.startsWith("move: ")
                    ? match.slice("move: ".length)
                    : match;
                match = match.startsWith("ability: ")
                    ? match.slice("ability: ".length)
                    : match;
                if (match) {
                    volatileStatusSet.add(match);
                }
            });
        }
    });

    // Convert the Set to an array and reduce it to unique, sorted identifiers
    let volatileStatus = reduce(["wrap", ...Array.from(volatileStatusSet)]);

    let weathers = extractPatterns(src, weathersPattern);
    weathers = reduce(weathers).map((t) => t.replace("raindance", "rain"));
    weathers = reduce(weathers).map((t) => t.replace("sandstorm", "sand"));
    weathers = reduce(weathers).map((t) => t.replace("sunnyday", "sun"));

    const sideConditionPattern =
        /this\.add\('-sidestart',\s*[^,]+,\s*'([^']+)'/g;
    let sideConditions = reduce(
        extractPatterns(src, sideConditionPattern).map((sideCondition) =>
            sideCondition.startsWith("move: ")
                ? sideCondition.slice("move: ".length)
                : sideCondition,
        ),
    );

    let terrain = extractPatterns(src, terrainPattern);
    terrain = reduce(terrain).map((t) => t.replace("terrain", ""));

    let pseudoweather = extractPatterns(src, pseudoWeatherPattern);
    pseudoweather = reduce(pseudoweather);

    let itemEffects = [];
    for (const pattern of itemEffectPatterns) {
        itemEffects.push(...extractPatterns(src, pattern));
    }
    itemEffects = reduce([
        "eaten",
        "popped",
        "consumed",
        "held up",
        ...itemEffects,
    ]);

    const genData = await getGenData(9);

    // Create the data object
    const data = {
        pseudoWeather: enumerate([nullToken, ...pseudoweather.sort()]),
        volatileStatus: enumerate([nullToken, ...volatileStatus.sort()]),
        itemEffect: enumerate([nullToken, ...itemEffects.sort()]),
        weathers: enumerate([nullToken, ...weathers.sort()]),
        terrain: enumerate([nullToken, ...terrain.sort()]),
        sideConditions: enumerate([nullToken, ...sideConditions.sort()]),
        ...formatData(genData),
        statuses: enumerate([
            nullToken,
            "slp",
            "psn",
            "brn",
            "frz",
            "par",
            "tox",
        ]),
        boosts: enumerate([
            "atk",
            "def",
            "spa",
            "spd",
            "spe",
            "accuracy",
            "evasion",
        ]),
        types: enumerate([
            ...extraTokens,
            ...genData.typechart.flatMap((type) =>
                type.isNonstandard === "Future" ? [] : type.id,
            ),
        ]),
        genders: enumerate([unkToken, "M", "F", "N"]),
        hyphenArgs: enumerate([
            unkToken,
            ...Object.keys(Protocol.ARGS)
                .filter((x) => x.startsWith("|-"))
                .map((str) => str.replace(/\|/g, "")),
        ]),
    };

    const parentDataDir = `data/`;

    if (!fs.existsSync(parentDataDir)) {
        fs.mkdirSync(parentDataDir, { recursive: true });
    }

    // Write the data to a JSON file
    fs.writeFileSync(
        `${parentDataDir}/data.json`,
        JSON.stringify(data, null, 2),
    );

    for (const genNo of [1, 2, 3, 4, 5, 6, 7, 8, 9]) {
        const parentDir = `${parentDataDir}/gen${genNo}/`;
        if (!fs.existsSync(parentDir)) {
            fs.mkdirSync(parentDir, { recursive: true });
        }
        const genData = await getGenData(genNo);
        for (const [key, value] of Object.entries(genData)) {
            const ourPath = `${parentDir}/${key}.json`;
            console.log(`writing ${ourPath}`);
            fs.writeFileSync(ourPath, JSON.stringify(value, null, 2));
        }
    }
}

// Execute the main function
main().catch((error) => {
    console.error("An error occurred:", error);
});
