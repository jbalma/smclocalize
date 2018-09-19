## Sensor localization

These are various files related to the problem of localizing sensors based on sensor data.

## Installation

To install:

1. git clone git@github.com:WildflowerSchools/smclocalize.git 
2. pip install -e ./smclocalize

## bin/smclocalize_worker

This runs the locations model, reading in sensor observation frames from redis,
and posting locations to firebase.  Must be run on a GPU accelerated machine.

A number of environment variables must be set. In a production environment, the
environment will likely provide a way to set this.  Outside of a production
environment, you will need to manage them yourself. One option is to create
a file that you can source with your shell. Here is an example with the keys
you will need to set, but the values are made up:

```
# .myenv
export FIREBASE_PRIVATE_KEY_ID="3028a9384023a0580239b0234023948f03998"
export FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----
MIIEvQIBAD *** This is not a real key ********* AQC7fEy6S4XDdwja
688ix02FI1VqRZtlYeORXBXk1lHAJ+EG4Xl5VXPLXCUhHfysyhLEjGqfefjcypTO
fMDUv2f2Ncy1s+125zVEEd4XCO+9eZ4+bOcZpSzzyJWdmvyywBeBKzGs2iO42X2Y
2MF7BlPD90XKxfnB2ErmVT4=
-----END PRIVATE KEY-----
"
export FIREBASE_CLIENT_EMAIL="some-email@something.iam.gserviceaccount.com"
export FIREBASE_PROJECT_ID="my-project-3249283"
export FIREBASE_CLIENT_ID="2054719480194382340"
export FIREBASE_CLIENT_X509_CERT_URL="https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-dn9oq%40project.iam.gserviceaccount.com"
export REDIS_URL="redis://localhost:6379"
export FIREBASE_URL="https://myproject.firebaseio.com"
export SENSEI_USERNAME="someusername"
export SENSEI_PASSWORD="somepassword"
```

To set the environment variables from the file, `source .myenv`

### Running the worker

`./bin/smclocalize_worker -c 735`
