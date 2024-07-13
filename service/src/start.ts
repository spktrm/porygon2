import { exec } from "child_process";

const runCommand = (command: string) => {
    const process = exec(command);

    process.stdout?.on("data", (data) => {
        console.log(`stdout: ${data}`);
    });

    process.stderr?.on("data", (data) => {
        console.error(`stderr: ${data}`);
    });

    process.on("close", (code) => {
        console.log(`process exited with code ${code}`);
    });
};

runCommand("npm run start-training-server");
runCommand("npm run start-evaluation-server");
