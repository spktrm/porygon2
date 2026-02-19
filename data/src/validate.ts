import { Teams, TeamValidator } from "@pkmn/sim";

// Get arguments from command line
// Usage: npx ts-node validate.ts <TEAM_STRING> <FORMAT>
const args = process.argv.slice(2);
const teamString = args[0];
const format = args[1] || "gen6nu"; // Defaults to gen6nu if not specified

if (!teamString) {
    console.error("❌ Error: Please provide a team string.");
    console.log("Usage: npx ts-node validate.ts 'TEAM_STRING' [format]");
    process.exit(1);
}

console.log(`\nValidating team for format: ${format}...`);

// Unpack the team
const team = Teams.unpack(teamString);

if (!team) {
    console.error("❌ Error: Could not parse/unpack the team string.");
    process.exit(1);
}

// Validate
const validator = new TeamValidator(format);
const errors = validator.validateTeam(team);

// Output results
if (errors && errors.length > 0) {
    console.log(`\n🚫 Team is INVALID. Found ${errors.length} error(s):`);
    errors.forEach((err) => console.log(`   - ${err}`));
    process.exit(1);
} else {
    console.log("\n✅ Team is VALID!");
    process.exit(0);
}
