## Installation

```bash
make datas
make protos
```

### Server Installation

1.

```bash
cd server
npm install
```

then

2.

```bash
npm run test
```

### Client Installation

Run step 1. of server installion in a separate terminal.

Then run the below commands in a new terminal

```bash
cd client
pip install virtualenv
virtualenv env
source env/bin/activate
pip install -r requirements.txt
python test.py
```
