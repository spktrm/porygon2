// package: 
// file: features.proto

import * as jspb from "google-protobuf";

export interface FeatureEntityMap {
  SPECIES: 0;
  ITEM: 1;
  ITEM_EFFECT: 2;
  ABILITY: 3;
  GENDER: 4;
  ACTIVE: 5;
  FAINTED: 6;
  HP: 7;
  MAXHP: 8;
  STATUS: 9;
  TOXIC_TURNS: 10;
  SLEEP_TURNS: 11;
  BEING_CALLED_BACK: 12;
  TRAPPED: 13;
  NEWLY_SWITCHED: 14;
  LEVEL: 15;
  MOVEID0: 16;
  MOVEID1: 17;
  MOVEID2: 18;
  MOVEID3: 19;
  MOVEPP0: 20;
  MOVEPP1: 21;
  MOVEPP2: 22;
  MOVEPP3: 23;
}

export const FeatureEntity: FeatureEntityMap;

export interface FeatureMovesetMap {
  MOVEID: 0;
  PPLEFT: 1;
  PPMAX: 2;
}

export const FeatureMoveset: FeatureMovesetMap;

export interface FeatureTurnContextMap {
  VALID: 0;
  IS_MY_TURN: 1;
  ACTION: 2;
  MOVE: 3;
  SWITCH_COUNTER: 4;
  MOVE_COUNTER: 5;
  TURN: 6;
}

export const FeatureTurnContext: FeatureTurnContextMap;

export interface FeatureWeatherMap {
  WEATHER_ID: 0;
  MIN_DURATION: 1;
  MAX_DURATION: 2;
}

export const FeatureWeather: FeatureWeatherMap;

