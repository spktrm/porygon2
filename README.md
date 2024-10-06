# Porygon2

<img src="porygon2.png" alt="porygon2" display="block" margin-left="auto" margin-right="auto" width="%50"/>

Porygon2 provides:

-   A UNIX server written in Node.js wrapped around the `sim` and `client` packages from [pkmn](https://github.com/pkmn).
-   A reinforcement learning framework for interacting with this server. Currently supports [R-NaD from DeepMind](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/rnad/rnad.py).

## Overview

Porygon2 is a platform that simulates Pokémon battles and provides an environment for training reinforcement learning agents. It leverages the `pkmn` library for accurate game mechanics and offers a server-client architecture to facilitate interactions between agents and the simulation environment.

## Installation

To set up the project, you can use the provided `setup.sh` script, which automates the installation process.

### Prerequisites

-   **Python 3** installed on your system.
-   **Node.js** and **npm** installed for running the server components.

### Steps

1.  **Clone the Repository**

```bash
git clone https://github.com/yourusername/porygon2.git
cd porygon2
```

2.  **Run the Setup Script**

Make sure the `setup.sh` script is executable:

```bash
chmod +x setup.sh
```

Then run the script:

```bash
./setup.sh
```

This script will:

-   Create a Python virtual environment in the `venv` directory.
-   Install all Python dependencies from `requirements.txt` files located in the root directory and immediate subdirectories, excluding any directories named `env`.
-   Install all Node.js dependencies by running `npm install` in directories (one layer deep) containing a `package.json` file, excluding any directories named `env` or `node_modules`.

3.  **Activate the Python Virtual Environment**

After running the script, your Python virtual environment is activated. If you open a new terminal session, reactivate it using:

```bash
source venv/bin/activate
```

## Training

### Server Setup

1.  **Navigate to the Server Directory**

```bash
cd service
```

2.  **Install Dependencies**

```bash
npm install
```

3.  **Run Tests (Optional)**

```bash
npm run test
```

4.  **Start the Training Server**

```bash
npm run start
```

### Client Setup

Open a new terminal and navigate to the root directory of the repository.

1.  **Activate the Virtual Environment**

```bash
source venv/bin/activate
```

2.  **Run the Training Client**

```bash
python ml/rl.py
```

## Evaluation

### Server Setup

1.  **Activate the Virtual Environment**

```bash
source venv/bin/activate
```

2.  **Start the Inference Server**

```bash
python inference/server.py
```

### Client Setup

1.  **Navigate to the Server Directory**

```bash
cd server
```

2.  **Install Dependencies**

```bash
npm install
```

3.  **Start the Evaluation Client**

```bash
npm run start-evaluation-client
```

## Scripts

The `scripts/` directory contains helper scripts for various tasks:

-   `compile_protos.sh`: Compiles protocol buffer definitions.
-   `generate_requirements.sh`: Generates `requirements.txt` files.
-   `lint.sh`: Runs code linters to ensure code quality.
-   `make_data.sh`: Generates necessary data for the project.

## Development Notes

-   **Directory Structure**: The project contains several subdirectories with their own `npm` installs. The `setup.sh` script handles these installations automatically.
-   **Python Virtual Environment**: A virtual environment named `env` is created in the root directory. Always activate it before running Python scripts.

## Additional Information

-   **License**: This project is licensed under the MIT License.
