import { exec } from "child_process";

const runCommand = (command: string, processName: string) => {
    const process = exec(command);

    process.stdout?.on("data", (data) => {
        data.toString()
            .split("\n")
            .forEach((line: any) => {
                if (line) {
                    console.log(`${processName} stdout: ${line}`);
                }
            });
    });

    process.stderr?.on("data", (data) => {
        data.toString()
            .split("\n")
            .forEach((line: any) => {
                if (line) {
                    console.error(`${processName} stderr: ${line}`);
                }
            });
    });

    process.on("close", (code) => {
        console.log(`${processName} process exited with code ${code}`);
    });
};

runCommand("npm run start-training-server", "TrainingServer");
runCommand("npm run start-evaluation-server", "EvaluationServer");
