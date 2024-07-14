## Installation

```bash
make datas
make protos
```

Everything below wil assume that the above commands have been run already

## Training

### Server Installation

1. install

```bash
cd service
npm install
```

2. run tests

```bash
npm run test
```

3. run training servers

```bash
npm run start
```

### Client Installation

Then run the below commands in a new terminal

```bash
pip install -r ml/requirements.txt
python ml/main.py
```

## Evaluation

### Server Installation

```bash
pip install -r inference/requirements.txt
python inference/server.py
```

### Client Installation

```bash
cd server
npm install
npm run start-evaluation-client
```
