### Server Installation

```bash
cd server
npm install
npm run compile-proto
npm start
```

### Client Installation

```bash
cd client
pip install virtualenv
virtualenv env
source env/bin/activate
pip install -r requirements.txt
python -m client.py
```
