import { Teams, TeamValidator } from "@pkmn/sim";
import * as fs from "fs";
import * as path from "path";

function processFile(filePath: string) {
    const fileName = path.basename(filePath);
    const match = fileName.match(/^(.*)_packed\.json$/);
    const format = match ? match[1] : "gen3ou"; // Fallback if regex fails

    const teamsJson = JSON.parse(
        fs.readFileSync(filePath, "utf-8"),
    ) as string[];
    console.log(`\nProcessing ${fileName} — ${teamsJson.length} teams found`);

    const validator = new TeamValidator(format);

    const validTeams = teamsJson.filter((packed) => {
        const unpacked = Teams.unpack([packed].join("]"));
        const errors = validator.validateTeam(unpacked);
        if (errors !== null) {
            console.error(
                `Invalid team in ${fileName}: ${packed} — Errors: ${errors}`,
            );
        }
        return errors === null;
    });

    console.log(`Valid teams for ${fileName}: ${validTeams.length}`);
    fs.writeFileSync(filePath, JSON.stringify(validTeams, null, 2));
}

function main() {
    const dataDir = path.resolve(__dirname, "../data");
    const packedFiles = fs
        .readdirSync(dataDir)
        .filter((f) => f.includes("packed") && f.endsWith(".json"));

    if (packedFiles.length === 0) {
        console.warn("No packed JSON files found in ../data");
        return;
    }

    packedFiles.forEach((file) => processFile(path.join(dataDir, file)));
}

main();
